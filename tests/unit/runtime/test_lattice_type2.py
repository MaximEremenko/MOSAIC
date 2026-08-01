"""Wrapper-level tests for the lattice scatter + type-2 path.

Covers what the tasks-level lattice tests cannot: the plan/scatter/type-2
primitives themselves — exactness against a direct DFT, the LSQ step
refinement that sets the reconstruction floor for noisy stored coordinates,
masked (sparse) occupancy, and degenerate axes.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.adapters.cunufft_wrapper import (
    execute_type2_on_lattice,
    plan_lattice,
)


def scatter_on_lattice(meta: dict, weights: np.ndarray) -> np.ndarray:
    """Reference scatter: plain np.add.at over the whole grid (duplicates sum,
    matching type-3 linearity exactly). Returns ``(n_trans, *dims)``.

    Test-only reference implementation. Production uses the striped,
    sort-based scatter in core/residual_field/tasks.py — this serial
    whole-grid np.add.at must never be reused on production-sized grids."""
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    dims = meta["dims"]
    flat = meta["flat_index"]
    grids = np.zeros((weights.shape[0], int(np.prod(dims))), dtype=np.complex128)
    for row in range(weights.shape[0]):
        np.add.at(grids[row], flat, weights[row])
    return grids.reshape((weights.shape[0],) + tuple(dims))


def execute_lattice_type2_batch(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
):
    """Reference plan + scatter + type-2 pipeline; ``None`` when the q-points
    are not lattice-eligible (production falls back to the type-3 path)."""
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    meta = plan_lattice(q_coords, n_trans=int(weights.shape[0]))
    if meta is None:
        return None
    grids = scatter_on_lattice(meta, weights)
    return execute_type2_on_lattice(
        meta, grids, real_coords, eps=eps, prefer_cpu=prefer_cpu, gpu_only=gpu_only
    )


def _direct_reference(q, w, tgt):
    # F(r) = sum_q w(q) exp(-i r.q)
    phases = np.exp(-1j * (tgt @ q.T))
    return w @ phases.T


def test_lattice_type2_matches_direct_dft_on_exact_lattice():
    rng = np.random.default_rng(3)
    step = 2 * np.pi / 16
    idx = rng.integers(-40, 41, size=(1500, 3))
    idx = np.unique(idx, axis=0)
    q = idx.astype(np.float64) * step
    w = (
        rng.standard_normal((2, len(q))) + 1j * rng.standard_normal((2, len(q)))
    ).astype(np.complex128)
    tgt = rng.uniform(0.0, 20.0, size=(200, 3))

    out = execute_lattice_type2_batch(q, w, tgt, eps=1e-12, prefer_cpu=True)
    assert out is not None
    ref = _direct_reference(q, w, tgt)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-10


def test_plan_lattice_lsq_refinement_beats_min_gap_floor():
    """Production q-coordinates are exact float64 lattice products, but the
    min-gap step estimate reads gaps quantized by round(col, 9), which biases
    the step by ~1e-9 relative; the index error grows with |index| and set the
    observed ~4e-6 hkl32 reconstruction floor. The LSQ refinement regresses
    the step over all points and recovers machine precision."""
    rng = np.random.default_rng(11)
    step = 2 * np.pi / 32
    idx = rng.integers(-200, 201, size=(3000, 3))
    idx = np.unique(idx, axis=0)
    q = idx.astype(np.float64) * step

    meta = plan_lattice(q)
    assert meta is not None
    # pre-refinement this deviation is ~1e-6 (min-gap bias x index range)
    assert meta["snap_dev"] < 1e-10
    assert abs(meta["dq"][0] - step) / step < 1e-12

    w = (
        rng.standard_normal((1, len(q))) + 1j * rng.standard_normal((1, len(q)))
    ).astype(np.complex128)
    tgt = rng.uniform(0.0, 30.0, size=(150, 3))
    out = execute_lattice_type2_batch(q, w, tgt, eps=1e-12, prefer_cpu=True)
    assert out is not None
    ref = _direct_reference(q, w, tgt)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-10


def test_scatter_on_lattice_sums_duplicates_and_keeps_masked_zero():
    step = 0.5
    q = np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.0], [1.0, 1.5]],
        dtype=np.float64,
    )
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    w = np.array([[1 + 1j, 2.0, 3.0, 4 - 1j]], dtype=np.complex128)
    grids = scatter_on_lattice(meta, w)
    # x axis: {0, 0.5, 1.0} -> 3 modes; y axis: {0, 1.5} -> 2 modes
    assert grids.shape == (1, 3, 2)
    assert grids[0, 0, 0] == 1 + 1j
    assert grids[0, 1, 0] == 5.0  # duplicates sum, matching type-3 linearity
    assert grids[0, 2, 1] == 4 - 1j
    # everything the mask removed stays exactly zero
    assert int(np.count_nonzero(grids[0])) == 3


def test_plan_lattice_degenerate_axis_gets_single_mode():
    rng = np.random.default_rng(5)
    step = 0.25
    idx = rng.integers(0, 30, size=(300, 3))
    q = idx.astype(np.float64) * step
    q[:, 2] = 1.75  # zero-plane-like: one l value for the whole role
    meta = plan_lattice(q)
    assert meta is not None
    assert meta["dims"][2] == 1
    assert meta["dq"][2] == 0.0
    assert meta["origin"][2] == pytest.approx(1.75)

    w = (
        rng.standard_normal((1, len(q))) + 1j * rng.standard_normal((1, len(q)))
    ).astype(np.complex128)
    tgt = rng.uniform(0.0, 10.0, size=(100, 3))
    out = execute_lattice_type2_batch(q, w, tgt, eps=1e-12, prefer_cpu=True)
    assert out is not None
    ref = _direct_reference(q, w, tgt)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-10


def test_plan_lattice_rejects_off_lattice_points():
    rng = np.random.default_rng(9)
    q = rng.uniform(-3.0, 3.0, size=(400, 3))  # genuinely scattered points
    assert plan_lattice(q) is None


def test_plan_lattice_rejects_grid_beyond_host_budget():
    step = 0.1
    q = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]) * step
    # dims (2, 2, 2) -> 8 modes x 16 B x n_trans=2 = 256 B > 100 B budget
    meta = plan_lattice(q, host_budget_bytes=100)
    assert meta is None


def test_lattice_type2_gpu_tiles_targets_when_pool_cap_is_small(caplog):
    """Streaming subchunks carry the chunk's FULL point range, so n_tgt is
    points x rifft-grid (hkl40: 2.1e8 targets = ~5 GiB of coordinates against
    a 2.4 GiB pool cap). The target axis must therefore be tiled: this pins a
    pool cap small enough that one monolithic upload cannot fit, and requires
    (a) the tiling branch actually engages and (b) the tiled GPU result stays
    at parity with the untiled CPU reference."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cufinufft")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA runtime unavailable")

    rng = np.random.default_rng(7)
    step = 2 * np.pi / 24
    idx = rng.integers(-30, 31, size=(4000, 3))
    idx = np.unique(idx, axis=0)
    q = idx.astype(np.float64) * step
    w = (
        rng.standard_normal((2, len(q))) + 1j * rng.standard_normal((2, len(q)))
    ).astype(np.complex128)
    meta = plan_lattice(q, n_trans=2)
    assert meta is not None
    grids = scatter_on_lattice(meta, w)
    tgt = rng.uniform(0.0, 25.0, size=(600_000, 3))

    ref = execute_type2_on_lattice(meta, grids, tgt, eps=1e-11, prefer_cpu=True)

    pool = cp.get_default_memory_pool()
    old_limit = int(pool.get_limit() or 0)
    pool.free_all_blocks()
    # 600k targets x 128 B/target needs ~77 MiB in flight; a 64 MiB cap forces
    # >=2 tiles while each tile (~41 MiB peak) still fits.
    pool.set_limit(size=64 << 20)
    try:
        with caplog.at_level(logging.INFO, logger="core.adapters.cunufft_wrapper"):
            out = execute_type2_on_lattice(meta, grids, tgt, eps=1e-11, gpu_only=True)
    finally:
        pool.set_limit(size=old_limit)
        pool.free_all_blocks()

    assert any("target tiles" in record.message for record in caplog.records)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-8


def test_lattice_type2_gpu_tile_sizing_sees_concurrent_siblings(caplog, monkeypatch):
    """Target-tile sizing runs AFTER _transform_enter() and divides the
    half-pool budget by the live in-flight count. Sized before entering (the
    old order), two slot siblings both read used_bytes()~0 and each claimed
    0.5x the whole pool -- 1.0x combined before a single slab ran. With a
    sibling pinned in flight the same pool cap must now force >=2 tiles where
    a lone transform runs monolithic, at unchanged numerical parity."""
    import core.adapters.cunufft_wrapper as wrapper

    cp = pytest.importorskip("cupy")
    pytest.importorskip("cufinufft")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA runtime unavailable")

    rng = np.random.default_rng(17)
    step = 2 * np.pi / 24
    idx = rng.integers(-30, 31, size=(4000, 3))
    idx = np.unique(idx, axis=0)
    q = idx.astype(np.float64) * step
    w = (
        rng.standard_normal((2, len(q))) + 1j * rng.standard_normal((2, len(q)))
    ).astype(np.complex128)
    meta = plan_lattice(q, n_trans=2)
    assert meta is not None
    grids = scatter_on_lattice(meta, w)
    tgt = rng.uniform(0.0, 25.0, size=(600_000, 3))

    ref = execute_type2_on_lattice(meta, grids, tgt, eps=1e-11, prefer_cpu=True)

    pool = cp.get_default_memory_pool()
    old_limit = int(pool.get_limit() or 0)
    pool.free_all_blocks()
    # 600k targets x 128 B/target: a 256 MiB cap leaves the lone transform's
    # half-pool budget (~128 MiB -> ~970k targets) monolithic, while a sibling
    # halves it again (~470k targets -> 2 tiles).
    pool.set_limit(size=256 << 20)
    try:
        with caplog.at_level(logging.INFO, logger="core.adapters.cunufft_wrapper"):
            alone = execute_type2_on_lattice(meta, grids, tgt, eps=1e-11, gpu_only=True)
        assert not any("target tiles" in r.message for r in caplog.records)
        caplog.clear()
        monkeypatch.setattr(wrapper, "_gpu_inflight", 1, raising=False)
        with caplog.at_level(logging.INFO, logger="core.adapters.cunufft_wrapper"):
            shared = execute_type2_on_lattice(meta, grids, tgt, eps=1e-11, gpu_only=True)
        assert any("target tiles" in r.message for r in caplog.records)
    finally:
        monkeypatch.setattr(wrapper, "_gpu_inflight", 0, raising=False)
        pool.set_limit(size=old_limit)
        pool.free_all_blocks()

    assert np.abs(alone - ref).max() / np.abs(ref).max() < 1e-8
    assert np.abs(shared - ref).max() / np.abs(ref).max() < 1e-8
