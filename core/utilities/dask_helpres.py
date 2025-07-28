# ============================================================================
#  utilities/dask_helpers.py
#  ---------------------------------------------------------------------------
#  ‣ Every path or cluster parameter comes from either
#      • environment variables   …or…
#      • an optional external JSON/YAML file referenced via $MOSAIC_DASK_CONFIG
#  ‣ Works unchanged across all Dask back‑ends:
#        local, cuda‑local, sge, slurm, pbs, lsf, oar, mpi
#  ---------------------------------------------------------------------------
#  Usage examples
#  --------------
#  $ export MOSAIC_DASK_CONFIG=$HOME/.config/mosaic/dask.yaml
#  $ python run_simulation.py          # picks everything up automatically
#
#  $ export DASK_BACKEND=slurm         # one‑off override beats config file
#  $ export DASK_MAX_WORKERS=32
#  $ export GPUS_PER_JOB=4
#  $ python run_gpu_job.py
# ============================================================================

from __future__ import annotations

import json
import os
import socket
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dask.distributed import Client, LocalCluster, get_client

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # yaml is optional – only needed for .yml configs
    yaml = None  # pragma: no cover

_BACKENDS = Literal[
    "local",
    "cuda-local",
    "sge",
    "slurm",
    "pbs",
    "lsf",
    "oar",
    "mpi",
]

# --------------------------------------------------------------------------- #
#  Configuration helpers                                                      #
# --------------------------------------------------------------------------- #

def _load_external_config() -> Dict[str, Any]:
    """Return dict parsed from file pointed to by $MOSAIC_DASK_CONFIG.

    Supports JSON or YAML (requires PyYAML). If the env‑var is unset or the
    file is missing/unparseable we return an empty dict and continue with env
    vars & function defaults.
    """
    path = os.getenv("MOSAIC_DASK_CONFIG")
    if not path:
        return {}

    p = Path(path).expanduser()
    if not p.is_file():
        warnings.warn(f"⚠️  MOSAIC_DASK_CONFIG={p} does not exist – ignored")
        return {}

    try:
        if p.suffix in {".yml", ".yaml"} and yaml is not None:
            return yaml.safe_load(p.read_text()) or {}
        return json.loads(p.read_text())
    except Exception as exc:  # pragma: no cover – be generous
        warnings.warn(f"⚠️  Failed to parse config file {p}: {exc} – ignored")
        return {}


# --------------------------------------------------------------------------- #
#  Public API                                                                 #
# --------------------------------------------------------------------------- #

def ensure_dask_client(
    max_workers: int = 2,
    *,
    threads_per_worker: int = 2,
    processes: bool = True,
    backend: _BACKENDS | None = None,
    gpu: int | None = None,  # GPUs *per job*
    dashboard: bool = True,
    worker_dashboard: bool | None = None,
    use_sge_gpu_complex: bool = False,  # add "-l gpu=N" only if True
    **cluster_kw: Any,
) -> Client:
    """Get or create a :class:`dask.distributed.Client`.

    All user‑tunable defaults are pulled from

      • $MOSAIC_DASK_CONFIG   (JSON/YAML)   – lowest precedence
      • $ENVIRONMENT_VARIABLES            – middle precedence
      • explicit function arguments       – highest precedence

    so callers *never* need to edit source code to tweak cluster settings.
    """

    # -------------------------------------------------- #
    # 1. Merge configuration tiers (see docstring)       #
    # -------------------------------------------------- #
    cfg_file = _load_external_config()

    def _pick(key: str, *sources, cast=lambda x: x):
        """Select first non‑None from sources then cast to desired type."""
        for src in sources:
            if src is not None:
                return cast(src)
        return None

    # CLI‑level overrides (function arguments) have already taken care of
    # themselves (they sit in the default parameter values).
    max_workers = _pick(
        "max_workers",
        max_workers,
        os.getenv("DASK_MAX_WORKERS"),
        cfg_file.get("max_workers"),
        cast=int,
    )
    threads_per_worker = _pick(
        "threads_per_worker",
        threads_per_worker,
        os.getenv("DASK_THREADS_PER_WORKER"),
        cfg_file.get("threads_per_worker"),
        cast=int,
    )
    gpu = _pick("gpu", gpu, os.getenv("GPUS_PER_JOB"), cfg_file.get("gpu"), cast=int)

    if worker_dashboard is None:
        worker_dashboard = _pick(
            "worker_dashboard",
            None,
            os.getenv("DASK_WORKER_DASHBOARD"),
            cfg_file.get("worker_dashboard"),
            cast=lambda v: bool(int(v)) if isinstance(v, str) else bool(v),
        )

    backend = (
        backend
        or os.getenv("DASK_BACKEND")
        or cfg_file.get("backend")
        or _auto_backend()
        or "local"
    ).lower()

    # Custom scheduler options from config file / env
    sched_opts_cfg: Dict[str, Any] = cfg_file.get("scheduler_options", {})
    sched_opts_env = {
        k.removeprefix("DASK_SCHED_").lower(): v
        for k, v in os.environ.items()
        if k.startswith("DASK_SCHED_")
    }

    # Any explicit *function* cluster_kw wins over both cfg_file & env
    cluster_kw = {**sched_opts_cfg, **sched_opts_env, **cluster_kw}

    # -------------------------------------------------- #
    # 2. Already inside a Client?  →  reuse it            #
    # -------------------------------------------------- #
    try:
        return get_client()
    except ValueError:
        pass

    # Default dash settings
    if worker_dashboard is None:
        worker_dashboard = dashboard

    # ────────── single‑node back‑ends ──────────
    if backend == "local":
        local_directory = os.getenv("DASK_LOCAL_DIR") or cfg_file.get("local_directory")
        cluster_kw.pop("job_extra_directives", None)
        cluster_kw.pop("python", None)
        cluster_kw.pop("scheduler_options", None)
        return Client(
            LocalCluster(
                n_workers=max_workers,
                threads_per_worker=threads_per_worker,
                processes=processes,
                dashboard_address=":8787" if dashboard else None,
                worker_dashboard=worker_dashboard,
                local_directory=local_directory,
                **cluster_kw,
            )
        )

    if backend == "cuda-local":
        from dask_cuda import LocalCUDACluster
        cluster_kw.pop("job_extra_directives", None)
        cluster_kw.pop("python", None)
        cluster_kw.pop("scheduler_options", None)
        local_directory = os.getenv("DASK_LOCAL_DIR") or cfg_file.get("local_directory")
        return Client(
            LocalCUDACluster(
                n_workers=max_workers,
                protocol=os.getenv("DASK_COMM_PROTOCOL", "tcp"),
                threads_per_worker=threads_per_worker,
                dashboard_address=":8787" if dashboard else None,
#worker_dashboard=worker_dashboard,
                local_directory=local_directory,
                CUDA_VISIBLE_DEVICES=os.getenv("CUDA_VISIBLE_DEVICES"),
                **cluster_kw,
            )
        )

    # ────────── job‑queue family ──────────
    if backend in {"sge", "slurm", "pbs", "lsf", "oar"}:
        from dask_jobqueue import (
            LSFCluster,
            OARCluster,
            PBSCluster,
            SGECluster,
            SLURMCluster,
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
            memory=os.getenv("DASK_MEMORY", cfg_file.get("memory", "0")),
            walltime=os.getenv("DASK_WALLTIME", cfg_file.get("walltime", "02:00:00")),
            local_directory=os.getenv("DASK_LOCAL_DIR", cfg_file.get("local_directory", "/tmp")),
        )

        # GPU tweaks — *no* "--resources" flag any more
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
        sched_opts.setdefault("host", f"{sched_ip}:8786")
        sched_opts.setdefault("port", 8786)
        if dashboard:
            sched_opts.setdefault("dashboard_address", ":8787")

        # Merge user overrides *after* defaults so user wins
        merged = {**defaults, **cluster_kw}
        merged.pop("resources", None)  # hard‑remove if user passed it by habit

        cluster = Cluster(**merged)
        cluster.scale(jobs=max_workers)
        return Client(cluster)

    # ────────── dask‑mpi ──────────
    if backend == "mpi":
        from dask_mpi import initialize

        initialize(
            nthreads=threads_per_worker,
            memory_limit="0",
            worker_dashboard=worker_dashboard,
            local_directory=os.getenv("DASK_LOCAL_DIR", cfg_file.get("local_directory", "/tmp")),
            python="dask-cuda-worker" if gpu else None,
        )
        return Client()

    raise ValueError(f"Unknown backend '{backend}'")


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #

def _auto_backend() -> Optional[str]:
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


def _choose_scheduler_ip(backend: str) -> str:
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

    # fallback: first non‑loopback IPv4
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for *_, sockaddr in infos:
            ip = sockaddr[0]
            if not ip.startswith("127."):
                return ip
    except OSError:
        pass

    return "127.0.0.1"  # worst‑case loopback


def _resolve(host: str) -> str:
    return socket.gethostbyname(host)


def _append(d: Dict[str, Any], key: str, *items: str) -> None:
    lst: List[str] = list(d.get(key, []))
    lst += [it for it in items if it not in lst]
    d[key] = lst


# --------------------------------------------------------------------------- #
#  Convenience for interactive sessions                                      #
# --------------------------------------------------------------------------- #

def shutdown_dask() -> None:
    try:
        client = get_client()
        client.close()
        getattr(client, "cluster", None) and client.cluster.close()
        print("✅  Dask client closed.")
    except ValueError:
        print("ℹ️  No active Dask client.")