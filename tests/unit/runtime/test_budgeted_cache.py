"""BudgetedLRU / bounded_prefetch semantics.

These pin the shared cache behavior the four per-process payload caches
(scattering interval payloads, streaming stage-1 memo, residual RIFFT
grids, residual lattice grids) rely on.
"""
from __future__ import annotations

import time

from core.runtime.budgeted_cache import BudgetedLRU, bounded_prefetch


def _sized_cache(max_bytes=100, **kwargs):
    return BudgetedLRU(max_bytes=max_bytes, size_fn=lambda value: value[1], **kwargs)


class TestBudgetedLRU:
    def test_budget_evicts_oldest_first(self):
        cache = _sized_cache(max_bytes=100)
        cache.store("a", ("A", 40))
        cache.store("b", ("B", 40))
        cache.store("c", ("C", 40))  # 120 > 100 -> evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == ("B", 40)
        assert cache.get("c") == ("C", 40)
        assert cache.bytes() == 80
        assert len(cache) == 2

    def test_get_refreshes_recency(self):
        cache = _sized_cache(max_bytes=100)
        cache.store("a", ("A", 40))
        cache.store("b", ("B", 40))
        assert cache.get("a") == ("A", 40)  # "a" becomes most recent
        cache.store("c", ("C", 40))  # evicts "b", not "a"
        assert cache.get("b") is None
        assert cache.get("a") == ("A", 40)
        assert cache.get("c") == ("C", 40)

    def test_oversized_value_never_stored(self):
        cache = _sized_cache(max_bytes=100)
        assert cache.store("big", ("BIG", 101)) is False
        assert len(cache) == 0
        assert cache.bytes() == 0

    def test_entry_count_cap(self):
        cache = _sized_cache(max_bytes=1_000_000, max_entries=2)
        cache.store("a", ("A", 1))
        cache.store("b", ("B", 1))
        cache.store("c", ("C", 1))
        assert cache.get("a") is None
        assert len(cache) == 2

    def test_zero_charge_entries_bypass_byte_budget(self):
        # The memmap zero-charge rule: a size_fn may charge disk-backed
        # values 0 so they cache regardless of the RAM budget (the lattice
        # and RIFFT caches charge np.memmap grids nothing).
        cache = BudgetedLRU(max_bytes=10, size_fn=lambda value: 0)
        for index in range(5):
            assert cache.store(index, f"spilled-{index}") is True
        assert cache.bytes() == 0
        assert len(cache) == 5

    def test_same_key_overwrite_recharges(self):
        cache = _sized_cache(max_bytes=100)
        cache.store("a", ("A", 90))
        cache.store("a", ("A2", 10))
        assert cache.bytes() == 10
        assert len(cache) == 1
        assert cache.get("a") == ("A2", 10)

    def test_min_entries_floor_retains_newest(self):
        budget = {"value": 100}
        cache = BudgetedLRU(
            max_bytes=lambda: budget["value"],
            size_fn=lambda value: value[1],
            min_entries=1,
        )
        cache.store("a", ("A", 60))
        budget["value"] = 50  # budget shrank below the resident entry
        cache.store("b", ("B", 40))  # evicts "a"; floor keeps "b" itself
        assert cache.get("a") is None
        assert cache.get("b") == ("B", 40)
        assert len(cache) == 1

    def test_callable_budget_reread_per_store(self):
        budget = {"value": 100}
        cache = BudgetedLRU(
            max_bytes=lambda: budget["value"], size_fn=lambda value: value[1]
        )
        cache.store("a", ("A", 60))
        budget["value"] = 10
        assert cache.store("b", ("B", 60)) is False  # oversized for new budget
        assert cache.get("a") == ("A", 60)

    def test_none_values_and_miss_sentinel(self):
        # The streaming memo stores None (mask-empty interval); get() must
        # distinguish that from a miss via the caller-provided default.
        miss = object()
        cache = BudgetedLRU(max_bytes=100, size_fn=lambda value: 1)
        cache.store("empty", None)
        assert cache.get("empty", miss) is None
        assert cache.get("absent", miss) is miss

    def test_clear(self):
        cache = _sized_cache()
        cache.store("a", ("A", 10))
        cache.clear()
        assert len(cache) == 0
        assert cache.bytes() == 0
        assert cache.get("a") is None


class TestBoundedPrefetch:
    def test_order_preserved_under_concurrency(self):
        def slow_inverse(value):
            # Later items finish first; order must still be input order.
            time.sleep((8 - value) * 0.002)
            return value * 10

        assert list(bounded_prefetch(range(8), slow_inverse, 4)) == [
            value * 10 for value in range(8)
        ]

    def test_window_bounds_lookahead(self):
        pulled: list[int] = []

        def source():
            for index in range(10):
                pulled.append(index)
                yield index

        gen = bounded_prefetch(source(), lambda value: value, 3)
        assert next(gen) == 0
        # Exactly `window` items were consumed from the source before the
        # first result was yielded.
        assert pulled == [0, 1, 2]
        assert next(gen) == 1
        assert pulled == [0, 1, 2, 3]
        assert list(gen) == list(range(2, 10))

    def test_window_one_is_sequential(self):
        pulled: list[int] = []

        def source():
            for index in range(4):
                pulled.append(index)
                yield index

        gen = bounded_prefetch(source(), lambda value: value + 100, 1)
        assert next(gen) == 100
        assert pulled == [0]  # no look-ahead at all
        assert list(gen) == [101, 102, 103]
