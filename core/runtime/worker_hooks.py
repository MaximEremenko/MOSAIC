from __future__ import annotations

import atexit
import logging
from contextlib import AbstractContextManager

from core.adapters.cunufft_wrapper import free_gpu_memory, set_cpu_only


logger = logging.getLogger(__name__)


class _NoopLock(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _final_cleanup() -> None:
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


def _safe_chunk_lock(name: str):
    try:
        from distributed import get_client
        from dask.distributed import Lock as dask_lock

        try:
            client = get_client()
        except Exception:
            return _NoopLock()

        loop = getattr(client, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)
        cls = type(client).__name__.lower()
        if "syncclient" in cls or not has_loop:
            return _NoopLock()
        return dask_lock(name, client=client)
    except Exception:
        return _NoopLock()


def chunk_mutex(chunk_id: int):
    return _safe_chunk_lock(f"chunk-{chunk_id}")


def is_gpu_runtime_error(error: Exception | str) -> bool:
    message = str(error).lower()
    return any(
        keyword in message
        for keyword in (
            "cuda",
            "cudart",
            "cufft",
            "cufinufft",
            "cupy",
            "device-side assert",
            "illegal memory access",
            "out of memory",
            "driver shutting down",
        )
    )


def handle_worker_gpu_failure(
    error: Exception,
    *,
    logger: logging.Logger,
) -> bool:
    if not is_gpu_runtime_error(error):
        free_gpu_memory()
        return False

    try:
        from distributed import get_worker

        worker = None
        try:
            set_cpu_only(True)
            worker = get_worker()
            logger.warning("Worker %s set to CPU-only after GPU error", worker.address)
        except Exception:
            worker = None
        if worker is not None:
            try:
                count = getattr(worker, "gpu_fail_count", 0)
                setattr(worker, "gpu_fail_count", int(count) + 1)
            except Exception:
                pass
    finally:
        free_gpu_memory()

    return True


def register_cleanup_plugin(client, *, is_sync_client) -> bool:
    if client is None or is_sync_client(client):
        return False
    try:
        client.register_worker_plugin(CuPyCleanup(), name="cupy-cleanup")
        return True
    except ValueError:
        return False


class CuPyCleanup:
    name = "cupy-cleanup"

    def teardown(self, worker):
        _final_cleanup()


atexit.register(_final_cleanup)

__all__ = [
    "CuPyCleanup",
    "chunk_mutex",
    "handle_worker_gpu_failure",
    "is_gpu_runtime_error",
    "register_cleanup_plugin",
]
