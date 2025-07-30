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


import os
import sys
from pathlib import Path
from typing import Optional
from dask.distributed import Client

from utilities.dask_helpres import ensure_dask_client

# Public symbols re‑exported for convenience
__all__ = ["get_client"]

# Singleton cache so repeated calls return the same Client
_CLIENT: Optional[Client] = None


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


def get_client() -> Client:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    # Where to write worker *.o / *.e logs – default: ./dask_logs
    log_dir = Path(os.getenv("MOSAIC_LOG_DIR", "dask_logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    extra = {}
    if os.getenv("DASK_BACKEND", "local") in {"sge", "slurm", "pbs", "lsf", "oar"}:
        extra["job_extra_directives"] = _build_job_extra(log_dir)

    _CLIENT = ensure_dask_client(
        backend=os.getenv("DASK_BACKEND", "local"),
        max_workers=int(os.getenv("DASK_MAX_WORKERS", "4")),
        threads_per_worker=int(os.getenv("DASK_THREADS_PER_WORKER", "4")),
        gpu=int(os.getenv("GPUS_PER_JOB", "0")),
        worker_dashboard=bool(int(os.getenv("DASK_WORKER_DASHBOARD", "0"))),
        python=os.getenv("DASK_PYTHON", sys.executable),
        scheduler_options={"host": os.getenv("DASK_SCHEDULER_HOST", "0.0.0.0")},
        resources={"nufft": 1},
        **extra,                         # ← only present for job‑queue back‑ends
    )
    return _CLIENT