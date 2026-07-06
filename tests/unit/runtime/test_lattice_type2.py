"""Wrapper-level tests for the lattice scatter + type-2 path.

Covers what the tasks-level lattice tests cannot: the plan/scatter/type-2
primitives themselves — exactness against a direct DFT, the LSQ step
refinement that sets the reconstruction floor for noisy stored coordinates,
masked (sparse) occupancy, and degenerate axes.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adapters.cunufft_wrapper import (
    execute_lattice_type2_batch,
    plan_lattice,
    scatter_on_lattice,
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
