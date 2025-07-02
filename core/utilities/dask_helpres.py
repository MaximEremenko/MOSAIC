# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:53:33 2025

@author: Maksim Eremenko
"""
# utilities/dask_helpers.py
from __future__ import annotations

import os
from typing import Literal, Optional

from dask.distributed import Client, LocalCluster, get_client

_BACKENDS = Literal[
    "local",            # LocalCluster – default
    "cuda-local",       # LocalCUDACluster     (single-node, many GPUs)
    "sge", "slurm", "pbs", "lsf", "oar",  # dask-jobqueue families
    "mpi"               # dask-mpi
]

def ensure_dask_client(
    max_workers: int = 2,
    *,
    threads_per_worker: int = 2,
    processes: bool = True,
    backend: _BACKENDS | str | None = None,
    gpu: int | None = None,           # how many GPUs **per worker job**
    dashboard: bool = True,
    **cluster_kw,                     # forwarded verbatim to the Cluster ctor
) -> Client:
    """
    Return an existing Dask `Client` or create a new one.

    Parameters
    ----------
    backend
        `"local"` (default), `"cuda-local"`, `"sge"`, `"slurm"`, `"pbs"`,
        `"lsf"`, `"oar"`, `"mpi"`.
        If *None* we try to auto-detect (`SLURM_JOB_ID`, `SGE_ROOT`, MPI env…).
    gpu
        Number of GPUs per **worker**.  For SLURM we emit
        ``#SBATCH --gpus={gpu}``; for SGE we stick it in ``resource_spec``;
        for local CUDA we select that many devices automatically.
    cluster_kw
        Any keyword accepted by the selected Cluster class, e.g.
        ``queue="gpu"`` (SLURM/SGE), ``walltime="04:00:00"``,
        ``memory="24GB"`` …  These override the defaults below.
    """
    try:                         # already inside a running client?
        return get_client()
    except ValueError:
        pass                     # => need to create one

    # ------------------------------------------------------------------ #
    # 1. pick a backend (explicit > env-var > quick auto-detect > local)
    # ------------------------------------------------------------------ #
    backend = (
        backend
        or os.getenv("DASK_BACKEND")     # user can export DASK_BACKEND=sge …
        or _auto_detect_backend()
        or "local"
    ).lower()

    # ------------------------------------------------------------------ #
    # 2. create the appropriate Cluster object
    # ------------------------------------------------------------------ #
    if backend == "local":
        cluster = LocalCluster(
            n_workers=max_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            dashboard_address=":8787" if dashboard else None,
            **cluster_kw,
        )

    elif backend == "cuda-local":
        from dask_cuda import LocalCUDACluster  # pip/conda install dask-cuda
        cluster = LocalCUDACluster(
            n_workers=max_workers,
            threads_per_worker=threads_per_worker,
            dashboard_address=":8787" if dashboard else None,
            CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"),
            **cluster_kw,
        )

    elif backend in {"sge", "slurm", "pbs", "lsf", "oar"}:
        from dask_jobqueue import (
            SGECluster, SLURMCluster, PBSCluster, LSFCluster, OARCluster
        )
        _JOBQUEUE_MAP = {
            "sge":   SGECluster,
            "slurm": SLURMCluster,
            "pbs":   PBSCluster,
            "lsf":   LSFCluster,
            "oar":   OARCluster,
        }
        C = _JOBQUEUE_MAP[backend]

        # sensible fall-backs that you can always override via **cluster_kw
        defaults = dict(
            processes=1,                       # 1 worker process per *job*
            cores=threads_per_worker,
            memory="4GB",
            walltime="02:00:00",
            interface="ib0"                    # <-- customise per site
        )
        if gpu:                               # add GPU directives
            if backend == "slurm":
                defaults["job_extra_directives"] = [f"--gpus={gpu}"]
            elif backend == "sge":
                defaults["resource_spec"] = f"gpu={gpu}"
            # PBS/LSF/OAR have their own syntax – pass via **cluster_kw

            # make every worker a CUDAWorker so CuPy / RAPIDS works out-of-box
            defaults["worker_extra_args"] = ["--worker-class", "dask_cuda.CUDAWorker"]

        # user-supplied kwargs win over our defaults
        defaults.update(cluster_kw)

        cluster = C(**defaults)
        cluster.scale(jobs=max_workers)

    elif backend == "mpi":
        # Must run *inside* an mpirun/mpiexec allocation
        from dask_mpi import initialize          # pip/conda install dask-mpi
        initialize(
            nthreads=threads_per_worker,
            memory_limit="0",                    # disable nanny memory checks
            local_directory=os.getenv("TMPDIR", "/tmp"),
            worker_class="dask_cuda.CUDAWorker" if gpu else None,
        )
        # at this point Scheduler, workers, and *this* rank are all up
        return Client()                         # nothing else to do

    else:
        raise ValueError(f"Unknown backend '{backend}'")

    return Client(cluster)


# ---------------------------------------------------------------------- #
#                   helpers (internal use only)                          #
# ---------------------------------------------------------------------- #
def _auto_detect_backend() -> Optional[str]:
    """Infer a sensible default backend from environment variables."""
    env = os.environ
    if "SLURM_JOB_ID" in env:
        return "slurm"
    if "SGE_ROOT" in env or env.get("JOB_ID"):
        return "sge"
    if "PBS_JOBID" in env:
        return "pbs"
    if "LSB_JOBID" in env:
        return "lsf"
    if "OMPI_COMM_WORLD_RANK" in env or "PMI_RANK" in env:
        return "mpi"
    return None

def shutdown_dask():
    """
    Close the current Dask Client (and its underlying cluster) **if** one
    exists.  Safe to call multiple times – it becomes a no-op when no client
    is active.
    """
    try:
        client = get_client()      # raises ValueError when there is no client
        # First close the client (disconnects workers & scheduler) …
        client.close()             # <- frees sockets, tasks, futures
        # … then the cluster itself (optional, but frees ports & temp dirs)
        if hasattr(client, "cluster"):
            client.cluster.close()
        print("✅ Dask client closed.")
    except ValueError:
        # Nothing to do – no client was running
        print("ℹ️  No active Dask client.")