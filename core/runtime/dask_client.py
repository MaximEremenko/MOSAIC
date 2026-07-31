# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:41:08 2025

@author: Maksim Eremenko
"""

"""Project‑level entry‑point for obtaining a Dask client.

Example usage inside MOSAIC code::

    from mosaic.dask_client import get_client
    client = get_client()

If you *need* to customise anything ad‑hoc (e.g. for a notebook), just set
environment variables instead of digging through source files.
"""


import logging
import os
import sys
from pathlib import Path
from typing import Optional
from dask.distributed import Client, get_client as _dd_get_client

from core.runtime.dask_helpers import ensure_dask_client

# Public symbols re‑exported for convenience
__all__ = ["get_client", "default_log_dir", "set_log_dir_for_run", "shutdown_dask"]

logger = logging.getLogger(__name__)

# Singleton cache so repeated calls return the same Client
_CLIENT: Optional[Client] = None


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    import logging as _lg
    _lg.getLogger(__name__).warning(
        "Ignoring invalid boolean %s=%r; using %s",
        name,
        raw,
        default,
    )
    return default


def _build_job_extra(log_dir: Path) -> list[str]:
    """Construct the *job_extra_directives* list for job‑queue clusters."""
    extras: list[str] = [
        "-cwd",
        "-V",
    ]
    for var in ("DASK_GPU", "DASK_PE", "DASK_HOST"):
        if os.getenv(var):
            extras.append(os.environ[var])

    # Log paths – preserve original `$JOB_ID` placeholders
    extras += [
        f"-o {log_dir}/worker.o.$JOB_ID.$TASK_ID",
        f"-e {log_dir}/worker.e.$JOB_ID.$TASK_ID",
    ]
    return extras


def _gpu_resource_slots(backend: str | None) -> int:
    """Declare local GPU task capacity when the runtime asked for GPU work.

    This is deliberately a scheduler-resource declaration only; GPU admission
    still probes each worker before any GPU NUFFT task is accepted.
    """
    raw = os.getenv("MOSAIC_DASK_GPU_RESOURCE")
    if raw is not None and str(raw).strip() != "":
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return 1
        if text in {"0", "false", "no", "off"}:
            return 0
        try:
            return max(0, int(text))
        except ValueError:
            logger.warning(
                "Ignoring invalid MOSAIC_DASK_GPU_RESOURCE=%r; not declaring GPU slots",
                raw,
            )
            return 0

    if str(backend or "").strip().lower() == "cuda-local":
        return 1

    try:
        if int(os.getenv("GPUS_PER_JOB", "0")) > 0:
            return 1
    except ValueError:
        pass

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() and visible.strip().lower() not in {
        "-1",
        "none",
        "nodevfiles",
    }:
        return 1
    return 0


def _mkdir_shared(path: Path, attempts: int = 5) -> None:
    """mkdir -p that tolerates many ranks racing on a network filesystem.

    On NFS/parallel FS, negative dentry caching makes ``exist_ok`` unreliable
    inside the race window (a sibling rank's fresh directory can still raise
    FileExistsError with is_dir() False, or the parent can look absent right
    after another rank created it). Retry with a short backoff.
    """
    import time

    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except (FileNotFoundError, FileExistsError):
            if path.is_dir():
                return
            time.sleep(0.2 * (attempt + 1))
    path.mkdir(parents=True, exist_ok=True)


def default_log_dir(base_dir: str | Path | None = None) -> Path:
    if os.getenv("MOSAIC_LOG_DIR"):
        return Path(os.environ["MOSAIC_LOG_DIR"]).expanduser()
    if base_dir is None:
        return Path("dask_logs")
    return Path(base_dir).expanduser() / "dask_logs"


def set_log_dir_for_run(run_dir: str | Path) -> Path:
    log_dir = default_log_dir(run_dir)
    if "MOSAIC_LOG_DIR" not in os.environ:
        os.environ["MOSAIC_LOG_DIR"] = str(log_dir)
    return log_dir


def _detected_gpu_count() -> int:
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            return len([t for t in visible.split(",") if t.strip()])
        return 0


def _resolve_max_workers(backend: str | None) -> int:
    """Worker count portable across machines: an explicit integer wins;
    ``auto``/unset means one worker per visible GPU for cuda-local (any node
    size), falling back to 4 elsewhere. Hard-coding a count is what quietly
    stranded half the GPUs on a 4-card box and would waste an 8-card node."""
    raw = os.getenv("DASK_MAX_WORKERS", "auto").strip().lower()
    if raw not in {"", "auto"}:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid DASK_MAX_WORKERS=%r; using auto.", raw)
    if str(backend or "").strip().lower() == "cuda-local":
        detected = _detected_gpu_count()
        if detected > 0:
            return detected
    return 4


def get_client() -> Client:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    # Where to write worker *.o / *.e logs – default: <run_dir>/dask_logs.
    # Non-fatal: in SPMD launches (dask-mpi) every rank passes through here
    # concurrently and only the driver strictly needs the directory; a
    # shared-FS race must not kill worker ranks.
    log_dir = default_log_dir().expanduser()
    try:
        _mkdir_shared(log_dir)
    except OSError as exc:
        logger.warning("Could not create dask log dir %s: %s", log_dir, exc)

    extra = {}
    configured_backend = os.getenv("DASK_BACKEND")
    if configured_backend in {"sge", "slurm", "pbs", "lsf", "oar"}:
        extra["job_extra_directives"] = _build_job_extra(log_dir)

    threads_per_worker = int(os.getenv("DASK_THREADS_PER_WORKER", "4"))

    # NUFFT supply per worker: default to one in-flight cuFINUFFT call per
    # worker. Extra threads help Python-side orchestration, but concurrent
    # cuFINUFFT plans on one GPU mostly multiply raw cudaMalloc scratch and
    # exhaust VRAM. Advanced users can still raise this after measuring.
    _raw = os.getenv("MOSAIC_NUFFT_SLOTS_PER_WORKER")
    if _raw is None or _raw.strip() == "":
        nufft_slots_per_worker = 1
    else:
        try:
            nufft_slots_per_worker = max(1, int(_raw))
        except ValueError:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "Ignoring invalid MOSAIC_NUFFT_SLOTS_PER_WORKER=%r; "
                "using one slot per worker",
                _raw,
            )
            nufft_slots_per_worker = 1

    processes = _env_bool("DASK_PROCESSES")

    resources = {"nufft": nufft_slots_per_worker}
    gpu_slots = _gpu_resource_slots(configured_backend)
    if gpu_slots > 0:
        resources["gpu"] = gpu_slots

    _CLIENT = ensure_dask_client(
        backend=configured_backend,
        max_workers=_resolve_max_workers(configured_backend),
        threads_per_worker=threads_per_worker,
        processes=processes,
        gpu=int(os.getenv("GPUS_PER_JOB", "0")),
        worker_dashboard=bool(int(os.getenv("DASK_WORKER_DASHBOARD", "0"))),
        python=os.getenv("DASK_PYTHON", sys.executable),
        scheduler_options={"host": os.getenv("DASK_SCHEDULER_HOST", "0.0.0.0")},
        resources=resources,
        **extra,                         # ← only present for job‑queue back‑ends
    )
    return _CLIENT


# --------------------------------------------------------------------------- #
#  Convenience for interactive sessions                                        #
# --------------------------------------------------------------------------- #

def shutdown_dask() -> None:
    """Close the cached project client (and its cluster) and clear the singleton.

    The ``_CLIENT`` singleton lives in this module, so its lifecycle
    (``get_client`` / ``shutdown_dask``) is owned here. ``dask_helpers`` holds
    only the lower-level cluster builders and never references this singleton,
    keeping the runtime import graph acyclic.
    """
    global _CLIENT
    client = _CLIENT
    try:
        if client is None:
            client = _dd_get_client()
        cluster = getattr(client, "cluster", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        cluster_close = getattr(cluster, "close", None)
        if callable(cluster_close):
            cluster_close()
        logger.info("Dask client closed.")
    except ValueError:
        logger.info("No active Dask client.")
    finally:
        _CLIENT = None
