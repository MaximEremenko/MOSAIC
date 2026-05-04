"""T1: bounded cuFINUFFT plan cache — numerical identity + LRU bound."""
from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest

from core.adapters import cunufft_wrapper as wrapper


class _SpyPlan:
    """Mock of cufinufft.Plan that records ctor + destruction counts."""
    instances_built = 0
    instances_destroyed = 0
    _lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        with _SpyPlan._lock:
            _SpyPlan.instances_built += 1
        self._destroyed = False
        self._plan = object()
        self._references = [object()]
        self._seed = hash((args, tuple(sorted(kwargs.items())))) & 0xFFFFFFFF

    def _destroy_plan(self, handle):
        if self._plan is None:
            return 0
        self._plan = None
        self._references = []
        with _SpyPlan._lock:
            _SpyPlan.instances_destroyed += 1
        return 0

    def setpts(self, *a, **kw):
        pass

    def execute(self, weights):
        rng = np.random.default_rng(self._seed)
        return rng.standard_normal(4) + 1j * rng.standard_normal(4)

    def __del__(self):
        if getattr(self, "_plan", None) is not None:
            self._destroy_plan(self._plan)

    @classmethod
    def reset(cls):
        with cls._lock:
            cls.instances_built = 0
            cls.instances_destroyed = 0


@pytest.fixture(autouse=True)
def _reset_module_state():
    wrapper._clear_plan_cache()
    wrapper._SUCCESSFUL_SUBPROB.clear()
    _SpyPlan.reset()
    yield
    wrapper._clear_plan_cache()
    wrapper._SUCCESSFUL_SUBPROB.clear()


def test_cache_hit_reuses_plan_object():
    # Cache key is just the Plan() ctor args after the redesign — different
    # point sets all reuse the same cached Plan via setpts().
    key = (1, 3, -1, 1, 1e-9, 32)

    def _builder():
        return _SpyPlan(3, 1)

    original_max = wrapper._PLAN_CACHE_MAX
    try:
        wrapper._PLAN_CACHE_MAX = 4
        plan_a, lock_a, cached_a = wrapper._plan_cache_get_or_build(key, _builder)
        plan_b, lock_b, cached_b = wrapper._plan_cache_get_or_build(key, _builder)

        assert plan_a is plan_b
        assert lock_a is lock_b
        assert cached_a is True and cached_b is True
        assert _SpyPlan.instances_built == 1
    finally:
        wrapper._PLAN_CACHE_MAX = original_max


def test_cache_disabled_builds_fresh_each_call_and_caller_destroys():
    """With MAX<=0 the cache is off: each call gets a fresh Plan and the
    caller is signalled to destroy it (`cached=False`)."""
    key = (1, 3, -1, 1, 1e-9, 32)
    original_max = wrapper._PLAN_CACHE_MAX
    try:
        wrapper._PLAN_CACHE_MAX = 0
        p1, _l1, c1 = wrapper._plan_cache_get_or_build(key, lambda: _SpyPlan(3, 1))
        p2, _l2, c2 = wrapper._plan_cache_get_or_build(key, lambda: _SpyPlan(3, 1))
        assert p1 is not p2
        assert c1 is False and c2 is False
        assert wrapper._plan_cache_stats()["size"] == 0
        assert _SpyPlan.instances_built == 2
    finally:
        wrapper._PLAN_CACHE_MAX = original_max


def test_distinct_keys_build_distinct_plans():
    # Plans differ ONLY when ctor args differ (e.g. n_trans or subprob).
    k1 = (1, 3, -1, 1, 1e-9, 32)
    k2 = (1, 3, -1, 2, 1e-9, 32)  # n_trans=2

    original_max = wrapper._PLAN_CACHE_MAX
    try:
        wrapper._PLAN_CACHE_MAX = 4
        p1, _, _c1 = wrapper._plan_cache_get_or_build(k1, lambda: _SpyPlan(3, 1))
        p2, _, _c2 = wrapper._plan_cache_get_or_build(k2, lambda: _SpyPlan(3, 1))
        assert p1 is not p2
        assert _SpyPlan.instances_built == 2
    finally:
        wrapper._PLAN_CACHE_MAX = original_max


def test_bound_honoured_evicts_oldest():
    original_max = wrapper._PLAN_CACHE_MAX
    try:
        wrapper._PLAN_CACHE_MAX = 4  # small bound for this test
        # Vary the gpu_maxsubprobsize slot to fill the cache with 5 entries.
        for i in range(5):
            key = (1, 3, -1, 1, 1e-9, i)
            wrapper._plan_cache_get_or_build(key, lambda: _SpyPlan(3, 1))
        stats = wrapper._plan_cache_stats()
        assert stats["size"] == 4
        assert _SpyPlan.instances_built == 5
        assert _SpyPlan.instances_destroyed == 1
    finally:
        wrapper._PLAN_CACHE_MAX = original_max


def test_subprob_memoization():
    assert wrapper._subprob_order(3, 4) == wrapper._DEFAULT_SUBPROBS
    wrapper._record_successful_subprob(3, 4, 8)
    order = wrapper._subprob_order(3, 4)
    assert order[0] == 8
    assert set(order) == set(wrapper._DEFAULT_SUBPROBS)


def test_destroy_plan_quietly_swallows_exceptions():
    class _Exploder:
        def __init__(self):
            self._plan = object()
            self._references = [object()]

        def _destroy_plan(self, handle):
            raise RuntimeError("Error destroying plan")

    plan = _Exploder()
    wrapper._destroy_plan_quietly(plan)


def test_bit_identical_output_cached_vs_rebuilt():
    key = (2, 3, -1, 1, 1e-9, 32)
    weights = np.arange(4, dtype=np.complex128)

    def _builder():
        return _SpyPlan(3, 2, n_trans=1, eps=1e-9, isign=-1, dtype="complex128")

    original_max = wrapper._PLAN_CACHE_MAX
    try:
        wrapper._PLAN_CACHE_MAX = 4
        p1, _, _c1 = wrapper._plan_cache_get_or_build(key, _builder)
        out1 = p1.execute(weights)

        wrapper._clear_plan_cache()
        p2, _, _c2 = wrapper._plan_cache_get_or_build(key, _builder)
        out2 = p2.execute(weights)

        # Same ctor args -> same seed -> bit-identical output.
        np.testing.assert_array_equal(out1, out2)
    finally:
        wrapper._PLAN_CACHE_MAX = original_max


def test_coord_sig_content_stable():
    a = np.arange(16, dtype=np.float64)
    b = np.arange(16, dtype=np.float64)
    assert wrapper._coord_sig(a) == wrapper._coord_sig(b)
    c = a.copy()
    c[0] = 999.0
    assert wrapper._coord_sig(a) != wrapper._coord_sig(c)
