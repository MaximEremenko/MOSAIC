# utilities/dask_helpers.py
# ======================================================================
# Create (or reuse) a Dask Client for local, cuda-local, or job-queue
# back-ends.  Works with dask-cuda ≥ 25.06 — no "--resources" flag.
# Robust across clusters where NIC names differ host-to-host.
# ======================================================================

from __future__ import annotations

import os
import socket
from typing import Literal, Optional, Dict, Any, List

from dask.distributed import Client, LocalCluster, get_client

_BACKENDS = Literal["local", "cuda-local", "sge", "slurm", "pbs", "lsf", "oar", "mpi"]


# ---------------------------------------------------------------------- #
def ensure_dask_client(
    max_workers: int = 2,
    *,
    threads_per_worker: int = 2,
    processes: bool = True,
    backend: _BACKENDS | None = None,
    gpu: int | None = None,               # GPUs **per job**
    dashboard: bool = True,
    worker_dashboard: bool | None = None,
    use_sge_gpu_complex: bool = False,    # add "-l gpu=N" only if True
    **cluster_kw: Any,
) -> Client:
    """
    Return an existing Dask Client or spin up one automatically.

    Key behavior change vs. earlier versions:
    -----------------------------------------
    * We no longer force a specific network interface (ib0/eth0/etc.).
      Each worker binds to whatever local address it naturally uses
      when connecting to the scheduler's published IP. This tolerates
      heterogeneous NIC naming across nodes.
    * Scheduler is pinned to a stable, routable IP:PORT (default 8786).
    """

    # 0. Are we already inside a client?
    try:
        return get_client()
    except ValueError:
        pass
    if worker_dashboard is None:
         worker_dashboard = dashboard
    backend = (
        backend or os.getenv("DASK_BACKEND") or _auto_backend() or "local"
    ).lower()

    # ────────── single-node back-ends ──────────
    if backend == "local":
        return Client(
            LocalCluster(
                n_workers=max_workers,
                threads_per_worker=threads_per_worker,
                processes=processes,
                dashboard_address=":8787" if dashboard else None,
                worker_dashboard=worker_dashboard,
                **cluster_kw,
            )
        )

    if backend == "cuda-local":
        from dask_cuda import LocalCUDACluster
        return Client(
            LocalCUDACluster(
                n_workers=max_workers,
                threads_per_worker=threads_per_worker,
                dashboard_address=":8787" if dashboard else None,
                worker_dashboard=worker_dashboard,
                CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"),
                **cluster_kw,
            )
        )

    # ────────── job-queue family (SGE / SLURM / PBS / LSF / OAR) ──────────
    if backend in {"sge", "slurm", "pbs", "lsf", "oar"}:
        from dask_jobqueue import (
            SGECluster, SLURMCluster, PBSCluster, LSFCluster, OARCluster
        )
        _MAP: Dict[str, Any] = {
            "sge": SGECluster,
            "slurm": SLURMCluster,
            "pbs": PBSCluster,
            "lsf": LSFCluster,
            "oar": OARCluster,
        }
        Cluster = _MAP[backend]

        defaults: Dict[str, Any] = dict(
            processes=1,
            cores=threads_per_worker,
            memory="0",
            walltime="02:00:00",
            local_directory=os.getenv("TMPDIR", "/tmp"),
        )

        # GPU tweaks — NO "--resources" flag any more
        if gpu:
            if backend == "slurm":
                _append(defaults, "job_extra_directives", f"--gpus={gpu}")
            elif backend == "sge" and use_sge_gpu_complex:
                _append(defaults, "job_extra_directives", f"-l gpu={gpu}")
            defaults["python"] = "dask-cuda-worker"

        # -------------------------------------------------- #
        # Fixed scheduler endpoint (IP:8786) + dashboard     #
        # -------------------------------------------------- #
        sched_ip = _choose_scheduler_ip(backend)
        sched_opts = defaults.setdefault("scheduler_options", {})
        # DON'T stomp user overrides already in cluster_kw
        sched_opts.setdefault("host", f"{sched_ip}:8786")
        sched_opts.setdefault("port", 8786)
        if dashboard:
            sched_opts.setdefault("dashboard_address", ":8787")

        # Merge user overrides *after* our defaults so user wins
        merged = {**defaults, **cluster_kw}
        merged.pop("resources", None)  # hard-remove if user passed it

        cluster = Cluster(**merged)
        cluster.scale(jobs=max_workers)
        return Client(cluster)

    # ────────── dask-mpi ──────────
    if backend == "mpi":
        from dask_mpi import initialize
        initialize(
            nthreads=threads_per_worker,
            memory_limit="0",
            worker_dashboard=worker_dashboard,
            local_directory=os.getenv("TMPDIR", "/tmp"),
            python="dask-cuda-worker" if gpu else None,
        )
        return Client()

    raise ValueError(f"Unknown backend '{backend}'")


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #
def _auto_backend() -> Optional[str]:
    env = os.environ
    if "SLURM_JOB_ID" in env:                 return "slurm"
    if "SGE_ROOT" in env or env.get("JOB_ID"):return "sge"
    if "PBS_JOBID" in env:                    return "pbs"
    if "LSB_JOBID" in env:                    return "lsf"
    if "OMPI_COMM_WORLD_RANK" in env or "PMI_RANK" in env:
        return "mpi"
    return None


def _choose_scheduler_ip(backend: str) -> str:
    """
    Pick an IP address reachable from all compute nodes.

    Priority:
      1. $DASK_SCHEDULER_IP (user override)
      2. $SGE_O_HOST   (when backend == 'sge')
      3. First non-loopback IPv4 attached to the current host
    """
    env = os.environ
    user = env.get("DASK_SCHEDULER_IP")
    if user:
        return _resolve(user)

    if backend == "sge":
        sge_host = env.get("SGE_O_HOST")
        if sge_host:
            try:
                return _resolve(sge_host)
            except OSError:
                pass

    # fallback: first non-loopback IPv4
    try:
        # getaddrinfo returns (family, socktype, proto, canonname, sockaddr)
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for *_, sockaddr in infos:
            ip = sockaddr[0]
            if not ip.startswith("127."):
                return ip
    except OSError:
        pass

    # absolute worst case: loopback
    return "127.0.0.1"


def _resolve(host: str) -> str:
    return socket.gethostbyname(host)


def _append(d: Dict[str, Any], key: str, *items: str) -> None:
    lst: List[str] = list(d.get(key, []))
    lst += [it for it in items if it not in lst]
    d[key] = lst


# ---------------------------------------------------------------------- #
def shutdown_dask() -> None:
    try:
        client = get_client()
        client.close()
        getattr(client, "cluster", None) and client.cluster.close()
        print("✅  Dask client closed.")
    except ValueError:
        print("ℹ️  No active Dask client.")
