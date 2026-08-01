"""Async durable-snapshot writer thread.

Extracted from the reducer backend god-module: the writer owns only its
queue/threading state, so it is testable without a backend instance.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.residual_field.local_accumulator import LiveLocalAccumulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingLocalSnapshot:
    """A captured, self-contained snapshot awaiting its durable commit."""

    key: tuple
    accumulator: "LiveLocalAccumulator"
    snapshot_seq: int
    payload: dict = field(repr=False)
    captured_interval_ids: tuple
    output_dir: str
    db_path: str
    cleanup_policy: str


class _LocalSnapshotWriter:
    """Per-process writer thread for durable accumulator snapshots.

    An 8.5 GB savez under the per-target fold lock blocked folds for
    4.5-7 s per checkpoint; the writer runs the identical commit sequence
    (savez -> atomic rename -> chunk-mutex manifest -> mark committed ->
    SQLite -> unlink prev) off the fold path. At most one write per
    target key is in flight; errors are stored and re-raised on the next
    fold or drain for that key.

    Lock order contract: never call drain() while holding a per-target
    lock or chunk_mutex — the commit acquires both.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._queue: "queue.Queue[tuple | None]" = queue.Queue()
        self._in_flight: dict[tuple, threading.Event] = {}
        self._errors: dict[tuple, BaseException] = {}
        self._thread: threading.Thread | None = None

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="mosaic-snapshot-writer", daemon=True
            )
            self._thread.start()

    def submit(self, key: tuple, commit_fn) -> bool:
        with self._guard:
            if key in self._in_flight:
                return False
            event = threading.Event()
            self._in_flight[key] = event
            self._ensure_thread()
        self._queue.put((key, commit_fn, event))
        return True

    def in_flight(self, key: tuple) -> bool:
        with self._guard:
            return key in self._in_flight

    def pop_error(self, key: tuple) -> BaseException | None:
        with self._guard:
            return self._errors.pop(key, None)

    def drain(self, key: tuple) -> bool:
        """Wait for the in-flight commit of ``key`` (if any); re-raise its
        stored error. Returns True when a commit finished during the wait."""
        with self._guard:
            event = self._in_flight.get(key)
        if event is None:
            error = self.pop_error(key)
            if error is not None:
                raise error
            return False
        event.wait()
        error = self.pop_error(key)
        if error is not None:
            raise error
        return True

    def drain_matching(self, predicate) -> None:
        with self._guard:
            keys = [key for key in self._in_flight if predicate(key)]
        for key in keys:
            self.drain(key)

    def has_matching(self, predicate) -> bool:
        with self._guard:
            return any(predicate(key) for key in self._in_flight)

    def drain_all(self) -> None:
        self.drain_matching(lambda _key: True)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            key, commit_fn, event = item
            try:
                commit_fn()
            except BaseException as exc:  # stored, surfaced on the fold path
                with self._guard:
                    self._errors[key] = exc
            finally:
                with self._guard:
                    self._in_flight.pop(key, None)
                event.set()
