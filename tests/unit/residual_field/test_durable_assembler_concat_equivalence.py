"""C1-VALIDATE: the streaming durable reduce equals a naive concatenate reference.

``assemble_durable_generation_chunk_payload`` streams each shard block into a
preallocated/memmap slot (peak RAM ~ one block; no giant ``np.concatenate`` / ``np.vstack``).
This test pins that streaming assembler as BYTE-IDENTICAL to a naive reference that
concatenates the delta/average/point_ids blocks and vstacks the grid_shape blocks over the
SAME ordered blocks.

We fixture the durable path by monkeypatching the loader the assembler calls --
``core.residual_field.backend.load_residual_field_generation_payload`` -- to return a set
of controlled in-memory blocks. Each "manifest" is an opaque sentinel keyed to its block,
so no on-disk shard fixture is required. Both the 1-D and 2-D ``grid_shape_nd`` block cases
are covered.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.residual_field.backend as backend
from core.residual_field.backend import assemble_durable_generation_chunk_payload


class _BlockManifest:
    """Opaque per-block manifest sentinel; only used as the loader lookup key."""

    __slots__ = ("token",)

    def __init__(self, token: int) -> None:
        self.token = token


def _make_blocks(*, grid_2d: bool, seed0: int = 4242):
    """Build a few controlled shard blocks with contiguous absolute point IDs.

    The durable assembler enforces that each block's ``point_ids`` equal the absolute
    chunk-row IDs ``arange(point_offset, point_offset + n_points)``. We honour that here so
    the streaming path validates against the reference rather than raising.
    """
    block_point_counts = [50, 0, 137, 1, 64]  # includes an empty block and a singleton
    blocks: list[dict] = []
    point_offset = 0
    for index, n_points in enumerate(block_point_counts):
        rng = np.random.default_rng(seed0 + index)
        point_ids = np.arange(point_offset, point_offset + n_points, dtype=np.int64)
        point_offset += n_points
        if grid_2d:
            # 2-D grid block: a few rows per shard (distinct values per row).
            grid_block = (
                np.arange(index * 12, index * 12 + 6, dtype=np.int64).reshape(2, 3)
            )
        else:
            # 1-D grid block: shape (k,) -> reference vstack treats it as one row.
            grid_block = np.array(
                [300 + index, 310 + index, 320 + index], dtype=np.int64
            )
        blocks.append(
            {
                "point_ids": point_ids,
                "grid_shape_nd": grid_block,
                "amplitudes_delta": (
                    rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)
                ).astype(np.complex128),
                "amplitudes_average": (
                    rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)
                ).astype(np.complex128),
                "reciprocal_point_count": int(7 * (index + 1)),
                "total_reciprocal_points": 999,
                "incorporated_interval_ids": (index, index + 500),
            }
        )
    return blocks


def _reference_concat(blocks: list[dict]) -> dict:
    """Naive reference: concatenate the 1-D arrays, vstack the grid blocks."""
    point_ids = np.concatenate(
        [np.asarray(b["point_ids"], dtype=np.int64).reshape(-1) for b in blocks]
    )
    delta = np.concatenate(
        [np.asarray(b["amplitudes_delta"], dtype=np.complex128).reshape(-1) for b in blocks]
    )
    average = np.concatenate(
        [np.asarray(b["amplitudes_average"], dtype=np.complex128).reshape(-1) for b in blocks]
    )
    grid_rows = []
    for b in blocks:
        g = np.asarray(b["grid_shape_nd"], dtype=np.int64)
        grid_rows.append(g.reshape(1, -1) if g.ndim == 1 else g)
    grid_shape_nd = np.vstack(grid_rows)
    reciprocal_point_count = sum(int(b["reciprocal_point_count"]) for b in blocks)
    interval_ids: set[int] = set()
    for b in blocks:
        interval_ids.update(int(i) for i in b["incorporated_interval_ids"])
    return {
        "point_ids": point_ids,
        "grid_shape_nd": grid_shape_nd,
        "amplitudes_delta": delta,
        "amplitudes_average": average,
        "reciprocal_point_count": reciprocal_point_count,
        "incorporated_interval_ids": tuple(sorted(interval_ids)),
    }


def _install_loader(monkeypatch, manifests, blocks):
    by_token = {m.token: block for m, block in zip(manifests, blocks)}

    def _fake_loader(manifest):
        # Return a fresh copy so the assembler's per-pass `del` cannot disturb the source.
        return dict(by_token[manifest.token])

    monkeypatch.setattr(backend, "load_residual_field_generation_payload", _fake_loader)


@pytest.mark.parametrize("grid_2d", [False, True], ids=["grid_1d", "grid_2d"])
@pytest.mark.parametrize("scratch", [False, True], ids=["ram", "memmap"])
def test_durable_assembler_byte_identical_to_concat(monkeypatch, tmp_path, grid_2d, scratch):
    blocks = _make_blocks(grid_2d=grid_2d)
    manifests = [_BlockManifest(token=i) for i in range(len(blocks))]
    _install_loader(monkeypatch, manifests, blocks)

    scratch_dir = (tmp_path / "finalize") if scratch else None
    total_reciprocal_points = 999

    payload = assemble_durable_generation_chunk_payload(
        sorted_manifests=manifests,
        total_reciprocal_points=total_reciprocal_points,
        scratch_dir=scratch_dir,
        chunk_id=0,
    )
    reference = _reference_concat(blocks)

    out_point_ids = np.asarray(payload["point_ids"])
    out_delta = np.asarray(payload["amplitudes_delta"])
    out_average = np.asarray(payload["amplitudes_average"])
    out_grid = np.asarray(payload["grid_shape_nd"])

    # Byte-for-byte identity of every array the reference concatenate produces.
    assert out_point_ids.dtype == reference["point_ids"].dtype
    assert out_delta.dtype == reference["amplitudes_delta"].dtype
    assert out_average.dtype == reference["amplitudes_average"].dtype
    assert out_grid.dtype == reference["grid_shape_nd"].dtype

    assert out_point_ids.shape == reference["point_ids"].shape
    assert out_grid.shape == reference["grid_shape_nd"].shape

    assert out_point_ids.tobytes() == reference["point_ids"].tobytes()
    assert out_delta.tobytes() == reference["amplitudes_delta"].tobytes()
    assert out_average.tobytes() == reference["amplitudes_average"].tobytes()
    assert out_grid.tobytes() == reference["grid_shape_nd"].tobytes()

    # Scalar / aggregate fields match the reference reductions.
    assert int(payload["reciprocal_point_count"]) == reference["reciprocal_point_count"]
    assert int(payload["total_reciprocal_points"]) == total_reciprocal_points
    assert payload["incorporated_interval_ids"] == reference["incorporated_interval_ids"]

    # point_ids are the contiguous absolute chunk-row IDs (the durable contract).
    total = int(out_point_ids.shape[0])
    assert np.array_equal(out_point_ids, np.arange(total, dtype=np.int64))
