"""The tiled/streaming type-2 path must reproduce the materialized path.

``tile_consumer`` exists so huge target sets never materialize the full
``(n_trans, n_tgt)`` result on the host; correctness demands the streamed
tiles reassemble to exactly what the full-array path returns (same transform,
same accumulation, only the delivery differs).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adapters.cunufft_wrapper import execute_type2_on_lattice


def _small_problem(seed: int = 7, n_tgt: int = 4321):
    rng = np.random.default_rng(seed)
    dims = (6, 7, 8)
    meta = {
        "dims": dims,
        "dq": np.array([0.0625, 0.0625, 0.0625]),
        "origin": np.array([-0.2, -0.15, -0.25]),
        "snap_dev": 0.0,
    }
    grids = (
        rng.standard_normal((2,) + dims) + 1j * rng.standard_normal((2,) + dims)
    ).astype(np.complex128)
    targets = rng.uniform(-3.0, 3.0, size=(n_tgt, 3)).astype(np.float64)
    return meta, grids, targets


def test_tile_consumer_matches_full_output_cpu(monkeypatch):
    monkeypatch.setenv("MOSAIC_NUFFT_CPU_TILE_TARGETS", "500")  # force ~9 tiles
    meta, grids, targets = _small_problem()

    reference = execute_type2_on_lattice(
        meta, grids, targets, eps=1e-12, prefer_cpu=True
    )

    streamed = np.zeros_like(reference)
    seen: list[tuple[int, int]] = []

    def consume(t0: int, t1: int, tile_out) -> None:
        seen.append((t0, t1))
        streamed[:, t0:t1] += tile_out

    result = execute_type2_on_lattice(
        meta, grids, targets, eps=1e-12, prefer_cpu=True, tile_consumer=consume
    )

    assert result is None
    assert len(seen) > 1, "tile bound not honored; consumer saw a single tile"
    covered = sorted(seen)
    assert covered[0][0] == 0 and covered[-1][1] == targets.shape[0]
    for (_, prev_end), (next_start, _) in zip(covered, covered[1:]):
        assert prev_end == next_start
    np.testing.assert_allclose(streamed, reference, rtol=1e-9, atol=1e-9)


def test_tile_consumer_matches_full_output_gpu():
    wrapper = pytest.importorskip("cupy")
    try:
        if wrapper.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        import cufinufft  # noqa: F401
    except Exception:
        pytest.skip("GPU NUFFT stack unavailable")

    meta, grids, targets = _small_problem(seed=11)
    reference = execute_type2_on_lattice(meta, grids, targets, eps=1e-12)
    streamed = np.zeros_like(reference)

    def consume(t0: int, t1: int, tile_out) -> None:
        streamed[:, t0:t1] += tile_out

    result = execute_type2_on_lattice(
        meta, grids, targets, eps=1e-12, tile_consumer=consume
    )
    assert result is None
    np.testing.assert_allclose(streamed, reference, rtol=1e-9, atol=1e-9)


def test_lattice_groups_fold_matches_manual_reconstruction(monkeypatch, tmp_path):
    """_execute_lattice_groups streams tiles into the (delta, average) pair;
    the fold plus per-tile conjugate reconstruction must equal the manual
    full-array computation it replaced."""
    monkeypatch.setenv("MOSAIC_NUFFT_CPU_TILE_TARGETS", "600")
    monkeypatch.setenv("MOSAIC_RESIDUAL_LATTICE_SCRATCH", str(tmp_path))

    from core.residual_field import tasks as residual_tasks
    from core.scattering.accumulation import (
        apply_half_space_conjugate_reconstruction,
    )

    meta_a, grids_a, targets = _small_problem(seed=3)
    meta_b, grids_b, _ = _small_problem(seed=5)
    groups = [
        ("positive_half", meta_a, grids_a),
        ("full", meta_b, grids_b),
    ]

    delta, average = residual_tasks._execute_lattice_groups(
        groups,
        rifft_grid=targets,
        nufft_eps=1e-12,
        nufft_prefer_cpu=True,
        nufft_gpu_only=False,
    )

    expected_delta = np.zeros(targets.shape[0], dtype=np.complex128)
    expected_average = np.zeros(targets.shape[0], dtype=np.complex128)
    empty_q = residual_tasks._LATTICE_EMPTY_Q
    for role, meta, grids in groups:
        outs = execute_type2_on_lattice(
            meta, grids, targets, eps=1e-12, prefer_cpu=True
        )
        expected_delta += apply_half_space_conjugate_reconstruction(
            outs[0], empty_q, role
        )
        expected_average += apply_half_space_conjugate_reconstruction(
            outs[1], empty_q, role
        )

    np.testing.assert_allclose(delta, expected_delta, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(average, expected_average, rtol=1e-9, atol=1e-9)
