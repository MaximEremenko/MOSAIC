"""Shared budgeted-LRU cache and bounded-prefetch primitives.

Consolidates the four hand-rolled per-process payload caches (scattering
interval payloads, streaming stage-1 memo, residual RIFFT grids, residual
lattice grids) and the sliding-window prefetch idiom used by the lattice
builders into one implementation.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Hashable, Iterable, Iterator

__all__ = ["BudgetedLRU", "bounded_prefetch"]


def _resolve(value):
    return value() if callable(value) else value


class BudgetedLRU:
    """Thread-safe byte-budgeted LRU with oldest-first eviction.

    - ``get`` refreshes recency (``move_to_end``), so eviction pops the
      least recently used entry first.
    - ``size_fn`` charges each stored value; caches holding disk-backed
      memmaps charge them 0 so page-cache-resident data never evicts
      RAM-resident data.
    - ``max_bytes``/``max_entries`` may be zero-arg callables so
      env-tunable budgets are re-read on every store.
    - A value whose charge exceeds the byte budget is never stored.
    - ``min_entries`` is the eviction floor: eviction never shrinks the
      cache below it (the just-stored entry is always retained either way,
      because its charge fits the budget and eviction is oldest-first).
    """

    def __init__(
        self,
        *,
        max_bytes: int | Callable[[], int],
        size_fn: Callable[[Any], int],
        max_entries: int | None | Callable[[], int | None] = None,
        min_entries: int = 0,
    ) -> None:
        self._max_bytes = max_bytes
        self._size_fn = size_fn
        self._max_entries = max_entries
        self._min_entries = int(min_entries)
        self._entries: "OrderedDict[Hashable, tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Return the stored value (which may itself be ``None`` for caches
        that memoize negative results — pass a sentinel ``default`` to
        distinguish a miss)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return default
            self._entries.move_to_end(key)
            return entry[0]

    def store(self, key: Hashable, value: Any) -> bool:
        nbytes = int(self._size_fn(value))
        max_bytes = int(_resolve(self._max_bytes))
        if nbytes > max_bytes:
            return False
        raw_max_entries = _resolve(self._max_entries)
        max_entries = None if raw_max_entries is None else int(raw_max_entries)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= int(old[1])
            self._entries[key] = (value, nbytes)
            self._bytes += nbytes
            while (
                self._bytes > max_bytes
                or (max_entries is not None and len(self._entries) > max_entries)
            ) and len(self._entries) > self._min_entries:
                _key, (_value, old_bytes) = self._entries.popitem(last=False)
                self._bytes -= int(old_bytes)
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def bytes(self) -> int:
        with self._lock:
            return self._bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def bounded_prefetch(
    iterable: Iterable[Any],
    fn: Callable[[Any], Any],
    window: int,
) -> Iterator[Any]:
    """Yield ``fn(item)`` in input order, computing up to ``window`` ahead.

    ``window <= 1`` degrades to a plain sequential map (no threads).
    Otherwise a deque + ThreadPoolExecutor sliding window keeps at most
    ``window`` results in flight, bounding peak memory at ``window``
    outputs while overlapping per-item work across threads.
    """
    if window <= 1:
        for item in iterable:
            yield fn(item)
        return
    with ThreadPoolExecutor(max_workers=window) as pool:
        pending: deque = deque()
        for item in iterable:
            pending.append(pool.submit(fn, item))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
