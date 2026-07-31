from __future__ import annotations

import atexit
import ctypes
import logging
import os
import sys
import tempfile
import threading
from contextlib import AbstractContextManager
from pathlib import Path


logger = logging.getLogger(__name__)


def free_gpu_memory() -> None:
    """Release GPU memory only if the NUFFT wrapper has already been loaded.

    Runtime helpers import the NUFFT adapter for non-GPU paths, so cleanup must
    not import it just to discover there is nothing to clean. CUDA initialization
    in that adapter is intentionally deferred until the first GPU operation.
    """
    module = sys.modules.get("core.adapters.cunufft_wrapper")
    cleanup = getattr(module, "free_gpu_memory", None) if module is not None else None
    if callable(cleanup):
        cleanup()


def set_cpu_only(flag: bool) -> None:
    from core.adapters.cunufft_wrapper import set_cpu_only as _set_cpu_only

    _set_cpu_only(flag)


def _malloc_trim() -> None:
    """Return freed glibc arenas to the OS.

    No-op on non-glibc platforms (macOS, musl, Windows). All failures are
    swallowed because this is purely a memory hygiene hint.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return
        trim.argtypes = [ctypes.c_size_t]
        trim(0)
    except Exception:
        return


class _NoopLock(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.RLock] = {}


class _FileChunkLock(AbstractContextManager):
    def __init__(self, path: Path):
        self.path = path
        self._thread_lock: threading.RLock | None = None
        self._handle = None

    def __enter__(self):
        lock_key = str(self.path)
        with _FILE_LOCKS_GUARD:
            self._thread_lock = _FILE_LOCKS.setdefault(lock_key, threading.RLock())
        self._thread_lock.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            # The thread lock still serializes in-process writers on platforms
            # without fcntl; Linux/WSL/HPC shared filesystems use fcntl here.
            pass
        return self

    def __exit__(self, *exc):
        try:
            if self._handle is not None:
                try:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                self._handle.close()
        finally:
            self._handle = None
            if self._thread_lock is not None:
                self._thread_lock.release()
            self._thread_lock = None
        return False


def _cleanup_process_local_reducers() -> None:
    residual_backend = sys.modules.get("core.residual_field.backend")
    if residual_backend is None:
        return
    for name in (
        "clear_process_local_residual_field_backends",
        "cleanup_process_local_residual_reducer_state",
        "clear_process_local_residual_reducer_state",
    ):
        cleanup = getattr(residual_backend, name, None)
        if callable(cleanup):
            cleanup()
            return


def _final_cleanup() -> None:
    free_gpu_memory()
    _cleanup_process_local_reducers()
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
    _malloc_trim()


def trim_worker_memory() -> None:
    """Drop reclaimable native/GPU pools without clearing live reducer state."""
    free_gpu_memory()
    _malloc_trim()


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


def chunk_mutex(chunk_id: int, *, lock_root: str | os.PathLike | None = None):
    if lock_root is not None:
        root = Path(lock_root)
        return _FileChunkLock(root / ".mosaic_locks" / f"chunk_{int(chunk_id)}.lock")
    return _safe_chunk_lock(f"chunk-{chunk_id}")


def _worker_scratch_base() -> Path:
    try:
        from distributed import get_worker

        worker = get_worker()
        local_directory = getattr(worker, "local_directory", None)
        if local_directory:
            return Path(str(local_directory))
        worker_dir = getattr(worker, "dir", None)
        if worker_dir:
            return Path(str(worker_dir))
    except Exception:
        pass
    return Path(tempfile.gettempdir())


def resolve_worker_scratch_root(
    *,
    preferred: str | None = None,
    stage: str,
) -> str:
    base = (
        preferred
        or os.getenv("MOSAIC_WORKER_SCRATCH_ROOT")
        or _worker_scratch_base()
    )
    base_path = Path(base).expanduser() if isinstance(base, str) else Path(base)
    try:
        from distributed import get_worker

        worker = get_worker()
        worker_token = (
            getattr(worker, "name", None)
            or getattr(worker, "address", None)
            or "worker"
        )
        safe_worker_token = (
            str(worker_token)
            .replace("://", "_")
            .replace(":", "_")
            .replace("/", "_")
        )
        return str((base_path / "mosaic" / stage / safe_worker_token).resolve())
    except Exception:
        return str((base_path / "mosaic" / stage / "local").resolve())


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


def _is_pool_capacity_error(error: Exception) -> bool:
    """CuPy POOL-cap OOM: the task's working set exceeded the per-worker pool
    limit. The device is healthy — the kernel paths self-heal by re-tiling —
    so demoting the worker to CPU-only turns one oversized work unit into a
    permanent ~40x slowdown for the rest of the run (hkl40 streaming: 1 h on
    GPU became a 21 h CPU ETA). Only the pool OOM is exempt; genuine
    runtime/driver faults (illegal memory access, device-side assert, driver
    shutting down) still demote."""
    if type(error).__name__ == "OutOfMemoryError" and "cupy" in type(error).__module__:
        return True
    return "limit set to" in str(error)


def handle_worker_gpu_failure(
    error: Exception,
    *,
    logger: logging.Logger,
) -> bool:
    if not is_gpu_runtime_error(error):
        free_gpu_memory()
        return False
    if _is_pool_capacity_error(error):
        logger.warning(
            "CuPy pool-cap OOM treated as workload sizing, not GPU failure; "
            "worker stays GPU-enabled: %s",
            error,
        )
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
    if not hasattr(client, "register_worker_plugin"):
        return False
    try:
        client.register_worker_plugin(CuPyCleanup(), name="cupy-cleanup")
        return True
    except (AttributeError, ValueError):
        return False


class CuPyCleanup:
    name = "cupy-cleanup"

    def teardown(self, worker):
        _final_cleanup()


class _PerTaskHeapTrim:
    """WorkerPlugin that returns CuPy mempool blocks and glibc arenas to the
    OS/device after each task. Fires strictly after a task's result has been
    produced, so it cannot affect output.

    Why ``MOSAIC_HEAP_TRIM_EVERY`` defaults to ``1`` (every task):
        CuPy's default memory pool retains *all* freed blocks for reuse,
        unbounded. With 16 concurrent in-flight tasks each holding ~500 MB
        of working scratch (rifft target buffers, NUFFT outputs), the pool
        can grow to fill VRAM in seconds. ``free_all_blocks()`` only releases
        UNREFERENCED blocks, so it does NOT touch live cuFINUFFT Plan objects
        in the plan cache; it only returns the per-task working scratch
        that the task itself just released by going out of scope.

    Adaptive trim trigger: if the CuPy default pool reports >= the
    fraction set by ``MOSAIC_GPU_TRIM_AT_PCT`` (default 0.50) of total VRAM
    used, the trim runs unconditionally regardless of throttle counter.

    Pickle-safe: the instance lock is stripped on pickle and recreated on
    unpickle so distributed can ship the plugin to worker processes.
    """

    name = "mosaic-heap-trim"

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()
        try:
            self._every = max(1, int(os.getenv("MOSAIC_HEAP_TRIM_EVERY", "1")))
        except ValueError:
            self._every = 1
        try:
            self._pressure_pct = float(os.getenv("MOSAIC_GPU_TRIM_AT_PCT", "0.50"))
        except ValueError:
            self._pressure_pct = 0.50

    def __getstate__(self):
        # threading.Lock cannot be pickled; rebuild on the worker side.
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        if "_n" not in self.__dict__:
            self._n = 0
        if "_every" not in self.__dict__:
            self._every = 1
        if "_pressure_pct" not in self.__dict__:
            self._pressure_pct = 0.50

    def setup(self, worker):
        return None

    def teardown(self, worker):
        _malloc_trim()

    def _under_gpu_pressure(self) -> bool:
        try:
            import cupy as cp  # type: ignore

            free, total = cp.cuda.runtime.memGetInfo()
            if total <= 0:
                return False
            used_frac = 1.0 - (float(free) / float(total))
            return used_frac >= self._pressure_pct
        except Exception:
            return False

    def _under_host_pressure(self) -> bool:
        """Host RAM can be 2 GB from the OOM killer while the GPU sits
        half idle — GPU pressure alone never trims then. Threshold via
        MOSAIC_HOST_TRIM_AT_AVAIL_FRACTION (default 0.15 of total)."""
        try:
            from core.runtime.cpu_resources import (
                available_memory_bytes,
                total_memory_bytes,
            )

            total = total_memory_bytes()
            available = available_memory_bytes()
            if not total or available is None:
                return False
            raw = os.getenv("MOSAIC_HOST_TRIM_AT_AVAIL_FRACTION", "0.15")
            try:
                fraction = min(0.9, max(0.01, float(raw)))
            except ValueError:
                fraction = 0.15
            return available < fraction * total
        except Exception:
            return False

    def transition(self, key, start, finish, **kwargs):
        if finish != "released":
            return
        with self._lock:
            self._n += 1
            scheduled = (self._n % self._every) == 0
        # Always trim under pressure even if not scheduled.
        if not (
            scheduled or self._under_gpu_pressure() or self._under_host_pressure()
        ):
            return
        try:
            import cupy as cp  # type: ignore

            cp.get_default_memory_pool().free_all_blocks()
            try:
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        except Exception:
            pass
        trim_worker_memory()


atexit.register(_final_cleanup)

__all__ = [
    "CuPyCleanup",
    "_PerTaskHeapTrim",
    "_malloc_trim",
    "chunk_mutex",
    "handle_worker_gpu_failure",
    "is_gpu_runtime_error",
    "register_cleanup_plugin",
    "resolve_worker_scratch_root",
    "trim_worker_memory",
]
