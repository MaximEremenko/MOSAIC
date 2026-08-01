"""Lattice type-1 forward path (Stage-1 scattering acceleration).

``execute_type1_on_lattice`` must match ``execute_cunufft`` (forward type-3,
isign=+1) to NUFFT eps on lattice-eligible q-points, reuse one plan across
intervals that differ only in box origin, and hand back ``None`` (type-3
fallback) when it cannot run.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adapters import cunufft_wrapper as wrapper
from core.adapters.cunufft_wrapper import (
    clear_lattice_type1_plan_cache,
    execute_cunufft,
    execute_type1_on_lattice,
    plan_lattice,
)
from core.scattering.kernels import (
    build_interval_lattice_meta,
    forward_interval_amplitudes,
)


@pytest.fixture(autouse=True)
def _fresh_plan_cache(monkeypatch):
    # the forward lattice path is opt-in; these tests exercise it directly
    monkeypatch.setenv("MOSAIC_SCATTERING_LATTICE_FFT", "1")
    clear_lattice_type1_plan_cache()
    yield
    clear_lattice_type1_plan_cache()


def _interval(seed, *, box=12, m_lo=-40, m_hi=40, keep=0.7):
    rng = np.random.default_rng(seed)
    step = 2 * np.pi / 16
    m0 = rng.integers(m_lo, m_hi - box, size=3)
    idx = np.stack(
        np.meshgrid(*[np.arange(box)] * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    idx = idx[rng.random(len(idx)) < keep]
    q = (m0[None, :] + idx) * step
    return q.astype(np.float64)


def _sources(seed, n=800):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 16.0, size=(n, 3))
    weights = (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex128)
    return coords, weights


def test_type1_matches_type3_on_lattice_interval():
    q = _interval(0)
    coords, weights = _sources(1)
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    out = execute_type1_on_lattice(meta, coords, weights, eps=1e-12)
    assert out is not None
    ref = execute_cunufft(coords, weights, q, eps=1e-12)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-9


def test_type1_matches_direct_dft():
    q = _interval(2, box=6, keep=1.0)
    coords, weights = _sources(3, n=200)
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    out = execute_type1_on_lattice(meta, coords, weights, eps=1e-12)
    assert out is not None
    ref = np.exp(1j * (q @ coords.T)) @ weights
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-9


def test_type1_cpu_path_matches_direct_dft():
    q = _interval(4, box=6, keep=1.0)
    coords, weights = _sources(5, n=200)
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    out = execute_type1_on_lattice(meta, coords, weights, eps=1e-12, prefer_cpu=True)
    assert out is not None
    ref = np.exp(1j * (q @ coords.T)) @ weights
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-9


def test_type1_plan_reused_across_interval_origins():
    """Two intervals with the same box dims and pitch but different origins
    must share one cached plan (the wrapped source coords do not depend on
    the interval)."""
    coords, weights = _sources(6)
    q_a = _interval(7, keep=1.0)
    q_b = _interval(8, keep=1.0)
    meta_a = plan_lattice(q_a, n_trans=1)
    meta_b = plan_lattice(q_b, n_trans=1)
    assert meta_a is not None and meta_b is not None
    assert tuple(meta_a["dims"]) == tuple(meta_b["dims"])

    out_a = execute_type1_on_lattice(meta_a, coords, weights, eps=1e-12)
    out_b = execute_type1_on_lattice(meta_b, coords, weights, eps=1e-12)
    if out_a is None or out_b is None:
        pytest.skip("GPU unavailable and CPU path does not cache plans")
    with wrapper._TYPE1_PLAN_CACHE_LOCK:
        cache_size = len(wrapper._TYPE1_PLAN_CACHE_ORDER)
    if wrapper._GPU_AVAILABLE and wrapper.cp is not None:
        assert cache_size == 1
    for q, out in ((q_a, out_a), (q_b, out_b)):
        ref = execute_cunufft(coords, weights, q, eps=1e-12)
        assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-9


def test_type1_degenerate_axis():
    rng = np.random.default_rng(9)
    step = 2 * np.pi / 16
    idx = rng.integers(0, 10, size=(400, 3))
    q = idx.astype(np.float64) * step
    q[:, 2] = 5 * step  # zero-plane-like single l value
    coords, weights = _sources(10, n=300)
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    assert meta["dims"][2] == 1
    out = execute_type1_on_lattice(meta, coords, weights, eps=1e-12)
    assert out is not None
    ref = np.exp(1j * (q @ coords.T)) @ weights
    assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-9


def test_type1_rejects_oversized_fine_grid():
    q = _interval(11)
    coords, weights = _sources(12)
    meta = plan_lattice(q, n_trans=1)
    assert meta is not None
    giant = dict(meta, dims=(4096, 4096, 4096))
    assert execute_type1_on_lattice(giant, coords, weights, eps=1e-12) is None


def test_type1_empty_sources_and_targets():
    q = _interval(13, box=6, keep=1.0)
    meta = plan_lattice(q, n_trans=1)
    out = execute_type1_on_lattice(
        meta, np.zeros((0, 3)), np.zeros(0, dtype=np.complex128), eps=1e-12
    )
    assert out is not None
    assert out.shape == (len(q),)
    assert not out.any()


class TestPlanCacheLeasing:
    """Eviction/clear must never destroy a plan another thread has leased —
    that would hand a freed cuFINUFFT handle to execute (use-after-free)."""

    def _entry(self):
        return wrapper._Type1PlanEntry(object(), [], None)

    def test_clear_defers_destroy_while_leased(self, monkeypatch):
        destroyed = []
        monkeypatch.setattr(
            wrapper, "_destroy_plan_quietly", lambda plan: destroyed.append(plan)
        )
        entry, cached = wrapper._type1_cache_acquire_or_build(
            ("k1",), lambda: (object(), [], None, 64, b"")
        )
        assert cached and entry.leases == 1
        clear_lattice_type1_plan_cache()
        assert entry.doomed
        assert destroyed == []          # lease outstanding: not destroyed yet
        wrapper._type1_entry_release(entry)
        assert len(destroyed) == 1      # last release destroys

    def test_eviction_defers_destroy_of_leased_entry(self, monkeypatch):
        destroyed = []
        monkeypatch.setattr(
            wrapper, "_destroy_plan_quietly", lambda plan: destroyed.append(plan)
        )
        monkeypatch.setenv("MOSAIC_SCATTERING_TYPE1_PLAN_CACHE_MAX", "1")
        leased, _ = wrapper._type1_cache_acquire_or_build(
            ("k1",), lambda: (object(), [], None, 64, b"")
        )
        other, _ = wrapper._type1_cache_acquire_or_build(
            ("k2",), lambda: (object(), [], None, 64, b"")
        )
        # inserting k2 evicted k1, but k1 is leased -> doomed, not destroyed
        assert leased.doomed
        assert destroyed == []
        wrapper._type1_entry_release(leased)
        assert len(destroyed) == 1
        wrapper._type1_entry_release(other)

    def test_unleased_eviction_destroys_immediately(self, monkeypatch):
        destroyed = []
        monkeypatch.setattr(
            wrapper, "_destroy_plan_quietly", lambda plan: destroyed.append(plan)
        )
        monkeypatch.setenv("MOSAIC_SCATTERING_TYPE1_PLAN_CACHE_MAX", "1")
        first, _ = wrapper._type1_cache_acquire_or_build(
            ("k1",), lambda: (object(), [], None, 64, b"")
        )
        wrapper._type1_entry_release(first)   # no lease outstanding
        second, _ = wrapper._type1_cache_acquire_or_build(
            ("k2",), lambda: (object(), [], None, 64, b"")
        )
        assert len(destroyed) == 1            # k1 destroyed at eviction
        wrapper._type1_entry_release(second)

    def test_cache_disabled_returns_uncached_entry(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_SCATTERING_TYPE1_PLAN_CACHE_MAX", "0")
        entry, cached = wrapper._type1_cache_acquire_or_build(
            ("k1",), lambda: (object(), [], None, 64, b"")
        )
        assert not cached
        with wrapper._TYPE1_PLAN_CACHE_LOCK:
            assert len(wrapper._TYPE1_PLAN_CACHE_ORDER) == 0


class TestNearLatticeRejection:
    """NEAR-lattice q must fall back to type-3, not be silently snapped:
    snapping data that is genuinely off-lattice by ~1e-6 of a step corrupts
    Stage-1 amplitudes at ~1e-5 relative with no warning."""

    def _sheared_interval(self, shear):
        rng = np.random.default_rng(21)
        step = 2 * np.pi / 16
        idx = rng.integers(-40, 40, size=(3000, 3))
        idx = np.unique(idx, axis=0).astype(np.float64)
        q = idx * step
        q[:, 0] += q[:, 2] * shear        # off-orthogonal cell mixes axes
        return q

    def test_exact_lattice_accepted(self):
        q = self._sheared_interval(0.0)
        assert build_interval_lattice_meta(q) is not None

    def test_near_lattice_rejected(self):
        q = self._sheared_interval(2e-7)
        meta = build_interval_lattice_meta(q)
        assert meta is None

    def test_plan_lattice_max_snap_dev_gate(self):
        q = self._sheared_interval(2e-7)
        loose = plan_lattice(q, n_trans=1)
        assert loose is not None          # legacy tolerance accepts it...
        assert loose["snap_dev"] > 1e-7   # ...with a deviation the gate sees
        assert plan_lattice(q, n_trans=1, max_snap_dev=1e-7) is None


class TestKernelsRouting:
    def test_forward_interval_amplitudes_lattice_matches_type3(self):
        q = _interval(14)
        coords, weights = _sources(15)
        meta = build_interval_lattice_meta(q)
        assert meta is not None
        via_lattice = forward_interval_amplitudes(
            coords, weights, q, lattice_meta=meta, nufft_eps=1e-12
        )
        via_type3 = forward_interval_amplitudes(
            coords, weights, q, lattice_meta=None, nufft_eps=1e-12
        )
        assert np.abs(via_lattice - via_type3).max() / np.abs(via_type3).max() < 1e-9

    def test_flag_off_disables_lattice_meta(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_SCATTERING_LATTICE_FFT", "0")
        assert build_interval_lattice_meta(_interval(16)) is None

    def test_flag_defaults_off(self, monkeypatch):
        monkeypatch.delenv("MOSAIC_SCATTERING_LATTICE_FFT", raising=False)
        assert build_interval_lattice_meta(_interval(18)) is None

    def test_off_lattice_points_get_no_meta(self):
        rng = np.random.default_rng(17)
        q = rng.uniform(-3.0, 3.0, size=(500, 3))
        assert build_interval_lattice_meta(q) is None
