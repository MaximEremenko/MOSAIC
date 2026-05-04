"""Regression test: finalize-stage preallocate-and-slot output is byte-identical
to the previous list-plus-np.concatenate path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.residual_field.backend import (
    _allocate_finalize_output,
    _finalize_scratch_dir,
)


# ---- Legacy reference (reconstructed inline) -----------------------------

def _legacy_concat_from_snapshots(snapshots: list[dict]) -> dict:
    point_ids_blocks: list[np.ndarray] = []
    delta_blocks: list[np.ndarray] = []
    average_blocks: list[np.ndarray] = []
    grid_shape_blocks: list[np.ndarray] = []
    point_offset = 0
    for snap in snapshots:
        d = np.asarray(snap["amplitudes_delta"], dtype=np.complex128).reshape(-1)
        a = np.asarray(snap["amplitudes_average"], dtype=np.complex128).reshape(-1)
        block_size = int(d.shape[0])
        point_ids_blocks.append(
            np.arange(point_offset, point_offset + block_size, dtype=np.int64)
        )
        point_offset += block_size
        delta_blocks.append(d)
        average_blocks.append(a)
        grid_shape_blocks.append(np.asarray(snap["grid_shape_nd"], dtype=np.int64))
    return {
        "point_ids": np.concatenate(point_ids_blocks),
        "grid_shape_nd": np.vstack(grid_shape_blocks),
        "amplitudes_delta": np.concatenate(delta_blocks),
        "amplitudes_average": np.concatenate(average_blocks),
    }


def _make_snapshots(n_partitions=4, n_points=100, seed0=9000):
    snapshots = []
    for p in range(n_partitions):
        rng = np.random.default_rng(seed0 + p)
        snapshots.append({
            "amplitudes_delta": (
                rng.standard_normal(n_points)
                + 1j * rng.standard_normal(n_points)
            ).astype(np.complex128),
            "amplitudes_average": (
                rng.standard_normal(n_points)
                + 1j * rng.standard_normal(n_points)
            ).astype(np.complex128),
            "grid_shape_nd": np.array([[326, 326, 326]], dtype=np.int64),
            "reciprocal_point_count": n_points,
            "total_reciprocal_points": 4 * n_points,
            "incorporated_interval_ids": (p, p + 1000),
        })
    return snapshots


# ---- Tests ---------------------------------------------------------------

def test_allocate_finalize_output_ram_and_memmap_match(tmp_path):
    shape = (50,)
    rng = np.random.default_rng(0)
    payload = (
        rng.standard_normal(shape[0]) + 1j * rng.standard_normal(shape[0])
    ).astype(np.complex128)

    ram = _allocate_finalize_output(
        shape=shape, dtype=np.complex128, scratch_dir=None, name="ram"
    )
    ram[:] = payload

    mm = _allocate_finalize_output(
        shape=shape, dtype=np.complex128,
        scratch_dir=tmp_path / "scratch", name="mm",
    )
    mm[:] = payload
    mm.flush()

    assert np.array_equal(np.asarray(ram), payload)
    assert np.array_equal(np.asarray(mm), payload)
    assert np.array_equal(np.asarray(ram), np.asarray(mm))


def test_finalize_scratch_dir_routing(tmp_path):
    assert _finalize_scratch_dir(None, chunk_id=0, parameter_digest="abc") is None
    out = _finalize_scratch_dir(str(tmp_path), chunk_id=3, parameter_digest="abc123")
    assert out is not None
    assert out.name == "chunk_3_params_abc123"


def test_slot_assignment_matches_legacy_concat_bytewise():
    """Simulate the patched site-B path; prove it produces the same bytes."""
    snapshots = _make_snapshots()
    legacy = _legacy_concat_from_snapshots(snapshots)

    total_points = sum(
        int(np.asarray(s["amplitudes_delta"]).reshape(-1).shape[0])
        for s in snapshots
    )
    grid_cols = int(np.asarray(snapshots[0]["grid_shape_nd"]).shape[1])
    total_rows = sum(
        int(np.asarray(s["grid_shape_nd"]).shape[0]) for s in snapshots
    )

    out_point_ids = _allocate_finalize_output(
        shape=(total_points,), dtype=np.int64, scratch_dir=None, name="pids"
    )
    out_delta = _allocate_finalize_output(
        shape=(total_points,), dtype=np.complex128, scratch_dir=None, name="d"
    )
    out_average = _allocate_finalize_output(
        shape=(total_points,), dtype=np.complex128, scratch_dir=None, name="a"
    )
    out_grid = _allocate_finalize_output(
        shape=(total_rows, grid_cols), dtype=np.int64,
        scratch_dir=None, name="g",
    )

    p_off = 0
    r_off = 0
    for snap in snapshots:
        d = np.asarray(snap["amplitudes_delta"], dtype=np.complex128).reshape(-1)
        a = np.asarray(snap["amplitudes_average"], dtype=np.complex128).reshape(-1)
        n = int(d.shape[0])
        out_delta[p_off:p_off + n] = d
        out_average[p_off:p_off + n] = a
        out_point_ids[p_off:p_off + n] = np.arange(p_off, p_off + n, dtype=np.int64)
        g = np.asarray(snap["grid_shape_nd"], dtype=np.int64)
        r = int(g.shape[0])
        out_grid[r_off:r_off + r] = g
        p_off += n
        r_off += r

    assert np.array_equal(out_delta, legacy["amplitudes_delta"])
    assert np.array_equal(out_average, legacy["amplitudes_average"])
    assert np.array_equal(out_point_ids, legacy["point_ids"])
    assert np.array_equal(out_grid, legacy["grid_shape_nd"])
    # Byte-for-byte serialization equality (strictly identical bytes).
    assert out_delta.tobytes() == legacy["amplitudes_delta"].tobytes()
    assert out_average.tobytes() == legacy["amplitudes_average"].tobytes()
    assert out_point_ids.tobytes() == legacy["point_ids"].tobytes()
    assert out_grid.tobytes() == legacy["grid_shape_nd"].tobytes()
