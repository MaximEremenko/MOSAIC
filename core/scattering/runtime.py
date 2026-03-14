from __future__ import annotations

import atexit
import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterable

from dask.distributed import Client, WorkerPlugin, as_completed
from tqdm import tqdm
from core.adapters.cunufft_wrapper import free_gpu_memory


logger = logging.getLogger(__name__)
TIMER = time.perf_counter
DEFAULT_TASK_RETRIES = 4


def _is_sync_client(client) -> bool:
    try:
        if client is None:
            return True
        cls = type(client).__name__.lower()
        loop = getattr(client, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)
        return ("syncclient" in cls) or (not has_loop)
    except Exception:
        return True


class _NoopLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _safe_chunk_lock(name: str):
    try:
        from distributed import get_client
        from dask.distributed import Lock as _DaskLock

        try:
            client = get_client()
        except Exception:
            return _NoopLock()

        loop = getattr(client, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)
        cls = type(client).__name__.lower()
        if "syncclient" in cls or not has_loop:
            return _NoopLock()
        return _DaskLock(name, client=client)
    except Exception:
        return _NoopLock()


def chunk_mutex(chunk_id: int):
    return _safe_chunk_lock(f"chunk-{chunk_id}")


def _yield_futures_with_results(futs: Iterable, client: Client | None):
    loop = getattr(client, "loop", None)
    for future, result in as_completed(futs, with_results=True, loop=loop):
        ok = False
        try:
            ok = bool(result)
        except Exception:
            ok = False
        yield future, ok


@contextmanager
def _timed(label: str):
    t0 = TIMER()
    try:
        yield
    finally:
        logger.info("%s took %.3f s", label, TIMER() - t0)


def _tqdm(total: int, *, desc: str, unit: str):
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0,
        miniters=1,
        mininterval=0.1,
        leave=True,
        disable=(total <= 0 or not sys.stderr.isatty()),
    )


@contextmanager
def _quiet_db_info():
    names = ("core.storage.database_manager", "DatabaseManager")
    logs = [logging.getLogger(name) for name in names]
    previous_levels = [log.level for log in logs]
    try:
        for log in logs:
            log.setLevel(max(logging.WARNING, log.level))
        yield
    finally:
        for log, level in zip(logs, previous_levels):
            log.setLevel(level)


def _final_cleanup():
    free_gpu_memory()

    try:
        from multiprocessing import resource_tracker, shared_memory

        for shm_name in list(shared_memory._SHARED_MEMORY_BLOCKS):
            try:
                shared_memory.SharedMemory(name=shm_name).unlink()
            except FileNotFoundError:
                pass
            resource_tracker.unregister(shm_name, "shared_memory")
    except Exception:
        pass


class CuPyCleanup(WorkerPlugin):
    name = "cupy-cleanup"

    def teardown(self, worker):
        _final_cleanup()


atexit.register(_final_cleanup)
