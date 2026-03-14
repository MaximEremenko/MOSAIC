from __future__ import annotations

from contextlib import contextmanager

from core.runtime import (
    DEFAULT_TASK_RETRIES,
    TIMER,
    chunk_mutex,
    is_sync_client as _is_sync_client,
    progress_bar as _tqdm,
    quiet_loggers,
    timed as _timed,
    yield_futures_with_results as _yield_futures_with_results,
)
from core.runtime.worker_hooks import CuPyCleanup


@contextmanager
def _quiet_db_info():
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        yield


__all__ = [
    "CuPyCleanup",
    "DEFAULT_TASK_RETRIES",
    "TIMER",
    "_is_sync_client",
    "_quiet_db_info",
    "_timed",
    "_tqdm",
    "_yield_futures_with_results",
    "chunk_mutex",
]
