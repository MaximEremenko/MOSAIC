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

from core.runtime.env import env_bool
from core.runtime.dask_helpers import (
    DEFAULT_DASK_THREADS_PER_WORKER,
    ensure_dask_client,
)

# Public symbols re‑exported for convenience
__all__ = ["get_client", "default_log_dir", "set_log_dir_for_run", "shutdown_dask"]

logger = logging.getLogger(__name__)

# Singleton cache so repeated calls return the same Client
_CLIENT: Optional[Client] = None


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    return env_bool(name, default, logger=logger, level=logging.WARNING)


def _build_job_extra(log_dir: Path, backend: str) -> list[str]:
    """Construct the *job_extra_directives* list for job‑queue clusters.

    dask-jobqueue prefixes each entry with the target scheduler's directive
    marker (``#$``, ``#SBATCH``, ``#PBS``, …), so the flags must be valid for
    that scheduler: an SGE ``-cwd`` rendered as ``#SBATCH -cwd`` is parsed by
    sbatch as ``-c wd`` and the whole job script is rejected.
    """
    extras: list[str] = []
    if backend == "sge":
        extras += [
            "-cwd",
            "-V",
            f"-o {log_dir}/worker.o.$JOB_ID.$TASK_ID",
            f"-e {log_dir}/worker.e.$JOB_ID.$TASK_ID",
        ]
    elif backend == "slurm":
        # SLURM exports the environment and starts in the submit directory by
        # default — no -V/-cwd equivalents needed.
        extras += [
            f"--output={log_dir}/worker.o.%j",
            f"--error={log_dir}/worker.e.%j",
        ]
    elif backend == "pbs":
        extras += [
            "-V",
            f"-o {log_dir}/worker.o",
            f"-e {log_dir}/worker.e",
        ]
    elif backend == "lsf":
        extras += [
            f"-o {log_dir}/worker.o.%J",
            f"-e {log_dir}/worker.e.%J",
        ]
    elif backend == "oar":
        extras += [
            f"--stdout={log_dir}/worker.o.%jobid%",
            f"--stderr={log_dir}/worker.e.%jobid%",
        ]

    # Raw passthrough for site-specific directives (queue/PE/GPU requests).
    for var in ("DASK_GPU", "DASK_PE", "DASK_HOST"):
        if os.getenv(var):
            extras.append(os.environ[var])
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
    """Worker count portable across machines: an explicit integer wins.
    ``auto``/unset resolves per backend — cuda-local: one worker per visible
    GPU; mpi: launch world size minus the scheduler and client ranks;
    job-queue backends: allocation size from the scheduler environment.
    Everything else falls back to 4. Hard-coding a count is what quietly
    stranded half the GPUs on a 4-card box and would waste an 8-card node."""
    raw = os.getenv("DASK_MAX_WORKERS", "auto").strip().lower()
    if raw not in {"", "auto"}:
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid DASK_MAX_WORKERS=%r; using auto.", raw)
    kind = str(backend or "").strip().lower()
    if kind == "cuda-local":
        detected = _detected_gpu_count()
        if detected > 0:
            return detected
    elif kind == "mpi":
        # dask-mpi SPMD layout: rank 0 = scheduler, rank 1 = client, every
        # remaining rank becomes a worker.
        for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
            value = os.getenv(var)
            if value:
                try:
                    world = int(value)
                except ValueError:
                    continue
                workers = max(1, world - 2)
                logger.info(
                    "max_workers=auto resolved to %d from %s=%d", workers, var, world
                )
                return workers
    elif kind in {"sge", "slurm", "pbs", "lsf", "oar"}:
        for var in ("SLURM_GPUS", "SLURM_JOB_NUM_NODES"):
            value = os.getenv(var)
            if value:
                try:
                    count = int(value)
                except ValueError:
                    continue
                if count > 0:
                    logger.info(
                        "max_workers=auto resolved to %d from %s", count, var
                    )
                    return count
    logger.info("max_workers=auto resolved to fallback 4 (backend=%s)", kind or "local")
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
        extra["job_extra_directives"] = _build_job_extra(log_dir, configured_backend)

    threads_per_worker = int(
        os.getenv("DASK_THREADS_PER_WORKER", str(DEFAULT_DASK_THREADS_PER_WORKER))
    )

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
            try:
                close()
            except Exception as exc:
                logger.warning("Dask client close failed (ignored): %s", exc)
        cluster_close = getattr(cluster, "close", None)
        if callable(cluster_close):
            try:
                cluster_close()
            except Exception as exc:
                # A slow/hung teardown (worker exits contending with the next
                # case's spin-up) must never turn a COMPLETED run into rc!=0
                # — the process is exiting and the OS reaps everything anyway.
                logger.warning("Dask cluster close failed (ignored): %s", exc)
        logger.info("Dask client closed.")
    except ValueError:
        logger.info("No active Dask client.")
    finally:
        _CLIENT = None
