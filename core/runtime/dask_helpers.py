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
import logging

from dask.distributed import Client, LocalCluster, as_completed, get_client

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
    "single-threaded",
    "sync",
    "synchronous",
]

logger = logging.getLogger(__name__)
DEFAULT_TASK_RETRIES = 4

# --------------------------------------------------------------------------- #
#  Configuration helpers                                                      #
# --------------------------------------------------------------------------- #

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")

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
    processes: bool | None = None,
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
    processes = _pick(
        "processes",
        processes,
        os.getenv("DASK_PROCESSES"),
        cfg_file.get("processes"),
        cast=_as_bool,
    )
    if processes is None:
        processes = True
    gpu = _pick("gpu", gpu, os.getenv("GPUS_PER_JOB"), cfg_file.get("gpu"), cast=int)

    if worker_dashboard is None:
        worker_dashboard = _pick(
            "worker_dashboard",
            None,
            os.getenv("DASK_WORKER_DASHBOARD"),
            cfg_file.get("worker_dashboard"),
            cast=lambda v: bool(int(v)) if isinstance(v, str) else bool(v),
        )

    auto_backend = _auto_backend()
    explicit_backend = backend or os.getenv("DASK_BACKEND") or cfg_file.get("backend")
    backend = (explicit_backend or auto_backend or "local").lower()
    if str(explicit_backend or "").lower() == "local" and auto_backend is not None:
        logger.warning(
            "HPC environment looks like %s, but Dask backend is explicitly local. "
            "This disables job-queue worker scheduling for the run.",
            auto_backend,
        )

    # if backend in {"single-threaded", "sync", "synchronous"}:
    # # Do NOT try to reuse a distributed Client; we want true in-thread execution
    # # so that pdb/cProfile/etc. work normally.
    #     return SyncClient()
    if backend in {"single-threaded", "sync", "synchronous"}:
        return SyncClient()
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

    # Worker-env hygiene: bound glibc arena count so freed blocks actually
    # return to the OS (classic cause of "unmanaged memory" growth in dask
    # workers). Only injected when the user has NOT already set the key,
    # and only for glibc-sensitive knobs that are math-neutral.
    _mem_env = {"MALLOC_ARENA_MAX": "2"}
    _worker_env = {k: v for k, v in _mem_env.items() if k not in os.environ}

    # ────────── single‑node back‑ends ──────────
    # LocalCluster / LocalCUDACluster accept ``resources=`` via the
    # ``**worker_kwargs`` catch-all — forwarded to Worker.__init__. Pass it
    # through at the top level (wrapping in an explicit ``worker_kwargs``
    # dict would make Worker receive ``worker_kwargs=...`` as an unknown
    # kwarg and crash the Nanny).
    if backend == "local":
        local_directory = os.getenv("DASK_LOCAL_DIR") or cfg_file.get("local_directory")
        cluster_kw.pop("job_extra_directives", None)
        cluster_kw.pop("python", None)
        cluster_kw.pop("scheduler_options", None)
        user_env = cluster_kw.get("env", {})
        if bool(processes):
            if isinstance(user_env, dict):
                cluster_kw["env"] = {**_worker_env, **user_env}
        else:
            # In-process/threaded LocalCluster uses distributed.Worker, whose
            # constructor does not accept env=. The current process environment
            # is already the worker environment in this mode.
            cluster_kw.pop("env", None)
            # Multiple threaded Worker objects share one Python process, so the
            # default LocalCluster memory_limit="auto" divides machine RAM by
            # n_workers but every Worker observes the same process RSS. That is
            # the root of false "Worker is at 80% memory usage. Pausing worker"
            # messages for processes=False runs. Disable Dask's per-worker RSS
            # limiter in this mode unless the caller explicitly configured it.
            cluster_kw.setdefault(
                "memory_limit",
                os.getenv(
                    "DASK_MEMORY_LIMIT",
                    cfg_file.get("memory_limit", cfg_file.get("memory", 0)),
                ),
            )
        client = Client(
            LocalCluster(
                n_workers=max_workers,
                threads_per_worker=threads_per_worker,
                processes=bool(processes),
                dashboard_address=":8787" if dashboard else None,
               # worker_dashboard=worker_dashboard,
                local_directory=local_directory,
                **cluster_kw,
            )
        )
        _register_heap_trim_plugin(client)
        return client

    if backend == "cuda-local":
        from dask_cuda import LocalCUDACluster
        cluster_kw.pop("job_extra_directives", None)
        cluster_kw.pop("python", None)
        cluster_kw.pop("scheduler_options", None)
        local_directory = os.getenv("DASK_LOCAL_DIR") or cfg_file.get("local_directory")
        # The residual pipeline stages its large arrays through file-backed
        # memmaps (accumulators, result pairs, target grids). Their resident
        # pages are RECLAIMABLE cache, but they count into process RSS, so
        # dask's default per-worker limit (total RAM / n_workers) reads them
        # as worker memory and the nanny kills healthy workers at 95% --
        # measured on hkl40: every 4-worker run died this way while >20 GB
        # stayed reclaimable. Default the limit OFF for cuda-local and let
        # the kernel arbitrate page cache; DASK_MEMORY_LIMIT overrides for
        # deployments that want a hard ceiling (e.g. cgroup-less shared
        # hosts).
        cluster_kw.setdefault(
            "memory_limit",
            os.getenv(
                "DASK_MEMORY_LIMIT",
                cfg_file.get("memory_limit", cfg_file.get("memory", 0)),
            ),
        )
        user_env = cluster_kw.get("env", {})
        if isinstance(user_env, dict):
            cluster_kw["env"] = {**_worker_env, **user_env}
        client = Client(
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
        _register_heap_trim_plugin(client)
        return client

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

        # Merge user overrides *after* defaults so user wins.
        # dask-jobqueue constructors do not accept Worker(resources=...) at
        # the cluster level; that has to be forwarded to the worker command.
        merged = {**defaults, **cluster_kw}
        resource_args = _resource_worker_args(merged.pop("resources", None))
        if resource_args:
            _append(merged, "worker_extra_args", *resource_args)

        cluster = Cluster(**merged)
        cluster.scale(jobs=max_workers)
        return Client(cluster)

    # ────────── dask‑mpi ──────────
    if backend == "mpi":
        # SPMD shape: every rank runs the same driver; initialize() turns
        # rank 0 into the scheduler and ranks >= 2 into workers (they never
        # return), rank 1 continues as the client. GPU pinning is per-rank
        # CUDA_VISIBLE_DEVICES set by the launch wrapper (one rank per GPU),
        # so plain distributed workers are correct — no dask-cuda needed.
        from dask_mpi import initialize

        worker_options: Dict[str, Any] = {}
        resources = cluster_kw.pop("resources", None)
        if isinstance(resources, dict) and resources:
            worker_options["resources"] = {
                str(name): float(value) for name, value in resources.items()
            }
        initialize(
            nthreads=threads_per_worker,
            # Same rationale as cuda-local: memmap pages count into RSS and
            # the nanny would kill healthy workers; kernel arbitrates.
            memory_limit=os.getenv("DASK_MEMORY_LIMIT", "0"),
            local_directory=os.getenv(
                "DASK_LOCAL_DIR", cfg_file.get("local_directory", "/tmp")
            ),
            worker_options=worker_options or None,
        )
        client = Client()
        _register_heap_trim_plugin(client)
        return client

    raise ValueError(f"Unknown backend '{backend}'")


def is_sync_client(client) -> bool:
    try:
        if client is None:
            return True
        cls = type(client).__name__.lower()
        loop = getattr(client, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)
        return ("syncclient" in cls) or (not has_loop)
    except Exception:
        return True


def yield_futures_with_results(futs, client: Client | None):
    loop = getattr(client, "loop", None)
    for future, result in as_completed(
        futs,
        with_results=True,
        raise_errors=False,
        loop=loop,
    ):
        status = getattr(future, "status", None)
        if status is None:
            ok = result is not None
        else:
            ok = status == "finished" and result is not None
        yield future, ok


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Single-threaded debug "client" (no distributed scheduler)                  #
# --------------------------------------------------------------------------- #
class _ImmediateFuture:
    """Minimal future-like wrapper for sync results."""
    __slots__ = ("_value",)
    def __init__(self, value): self._value = value
    def result(self, timeout=None): return self._value
    def done(self): return True
    def cancel(self): return False
    def cancelled(self): return False


class SyncClient:
    def __init__(self):
        import dask
        self._dask = dask
        self.cluster = None
        self.config = dask.config
        dask.config.set(scheduler="synchronous")

    def wait_for_workers(self, n=1, timeout=None):
        return True

    def register_worker_plugin(self, plugin, name=None):
        import warnings
        warnings.warn(
            f"[SyncClient] Ignoring worker plugin {name or plugin!r} "
            "because backend is single-threaded.",
            RuntimeWarning,
        )
        return None
    def scatter(self, data, broadcast=False, hash=True, direct=None, **kwargs):
            """In sync mode just return the raw data so delayed funcs see arrays, not futures."""
            return data
    def submit(self, func, *args, **kwargs):
        dist_only = {"pure", "resources", "priority", "retries", "fifo_timeout"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in dist_only}
        delayed_task = self._dask.delayed(func)(*args, **safe_kwargs)
        result = delayed_task.compute(scheduler="synchronous")
        return _ImmediateFuture(result)

    def map(self, func, *iterables, **kwargs):
        dist_only = {"pure", "resources", "priority", "retries", "fifo_timeout"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in dist_only}
        if len(iterables) == 1:
            return [_ImmediateFuture(func(x, **safe_kwargs)) for x in iterables[0]]
        else:
            return [_ImmediateFuture(func(*xs, **safe_kwargs)) for xs in zip(*iterables)]

    def gather(self, futures, **kwargs):
        if isinstance(futures, _ImmediateFuture):
            return futures.result()
        if isinstance(futures, dict):
            return {k: self.gather(v) for k, v in futures.items()}
        if isinstance(futures, (list, tuple)):
            return [self.gather(f) for f in futures]
        return futures

    def compute(self, *objs, **kwargs):
        return self._dask.compute(*objs, scheduler="synchronous", **kwargs)

    def persist(self, *objs, **kwargs):
        return self._dask.persist(*objs, scheduler="synchronous", **kwargs)

    def run(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def get_worker(self): return None
    def close(self): return None


    def register_worker_plugin(self, plugin, name=None):
        """Ignore worker plugins in sync mode."""
        import warnings
        warnings.warn(
            f"[SyncClient] Ignoring register_worker_plugin({plugin!r}) "
            "because backend is single-threaded.",
            RuntimeWarning,
        )
        return None

    def run(self, func, *args, **kwargs):
        """Simulate client.run: call the function locally once."""
        return func(*args, **kwargs)

    def get_worker(self):
        """No worker concept in sync mode."""
        return None

    def submit(self, func, *args, **kwargs):
            # kwargs accepted by distributed.Client.submit but NOT your function
            dist_only = {
                "pure",
                "resources",
                "priority",
                "retries",
                "fifo_timeout",
                "key",
                "workers",
                "allow_other_workers",
                "actor",
                "annotations",
                "ttl",
                "batch_size",
            }
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in dist_only}
            delayed_task = self._dask.delayed(func)(*args, **safe_kwargs)
            result = delayed_task.compute(scheduler="synchronous")
            return _ImmediateFuture(result)

    def map(self, func, *iterables, **kwargs):
        dist_only = {
            "pure",
            "resources",
            "priority",
            "retries",
            "fifo_timeout",
            "key",
            "workers",
            "allow_other_workers",
            "actor",
            "annotations",
            "ttl",
            "batch_size",
        }
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in dist_only}
        if len(iterables) == 1:
            return [_ImmediateFuture(func(x, **safe_kwargs)) for x in iterables[0]]
        else:
            return [_ImmediateFuture(func(*xs, **safe_kwargs)) for xs in zip(*iterables)]


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


def _resource_worker_args(resources: Any) -> tuple[str, ...]:
    if not isinstance(resources, dict) or not resources:
        return ()
    parts: list[str] = []
    for name, value in sorted(resources.items()):
        if value is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = value
        parts.append(f"{name}={numeric}")
    if not parts:
        return ()
    return ("--resources", ",".join(parts))


def _register_heap_trim_plugin(client) -> None:
    """Attach the per-task heap-trim WorkerPlugin. Best-effort; silent on
    failure. No-op for SyncClient / clients without register_worker_plugin."""
    try:
        if client is None or not hasattr(client, "register_worker_plugin"):
            return
        if is_sync_client(client):
            return
        from core.runtime.worker_hooks import _PerTaskHeapTrim

        client.register_worker_plugin(_PerTaskHeapTrim(), name="mosaic-heap-trim")
    except Exception:
        return
