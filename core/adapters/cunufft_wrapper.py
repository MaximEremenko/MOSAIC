# -*- coding: utf-8 -*-
"""
Robust, stream-aware cuFINUFFT/FINUFFT wrapper for Mosaic
=========================================================
rev 2025-08-08-mem

*  GPU used automatically if present; else CPU fallback.
*  Set env-var MOSAIC_NUFFT_CPU_ONLY=1  **or**  call `set_cpu_only(True)`
   to force CPU execution.
*  Chunk size is chosen by an explicit VRAM budget model.
"""

from __future__ import annotations
from typing import Callable, Optional
import copy
import hashlib
import logging
import os
import threading
import time
import warnings
import numpy as np

from core.adapters._direct_dft import direct_dft_type3


logger = logging.getLogger(__name__)
_LAST_NUFFT_TELEMETRY = None


###############################################################################
#  Bounded cuFINUFFT Plan cache                                               #
#                                                                             #
#  Cache key is the exact set of Plan() + setpts arguments, so a cache hit    #
#  replays the bit-identical call. Per-plan lock serializes concurrent        #
#  setpts/execute calls on the same plan (cuFINUFFT Plans are not             #
#  thread-safe). On CuPy OOM the cache is flushed once and the builder       #
#  re-runs.                                                                   #
###############################################################################
def _coord_sig(arr) -> bytes:
    if arr is None:
        return b"\x00" * 8
    # Accept CuPy arrays (disallow implicit asarray) and NumPy arrays alike.
    get = getattr(arr, "get", None)
    if callable(get):
        try:
            host = np.ascontiguousarray(get())
        except Exception:
            host = np.ascontiguousarray(np.asarray(arr))
    else:
        host = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(host.view(np.uint8).tobytes()).digest()[:8]


def _coords_sig(cols) -> bytes:
    if cols is None:
        return b"\x00" * 8
    return b"".join(_coord_sig(c) for c in cols)


_PLAN_CACHE_MAX = int(os.getenv("MOSAIC_NUFFT_PLAN_CACHE_MAX", "0"))
_PLAN_CACHE: "dict[tuple, tuple]" = {}
_PLAN_CACHE_ORDER: list = []
_PLAN_CACHE_LOCK = threading.Lock()

# Concurrency-aware GPU admission. Worker threads sharing one process (the
# default Dask ``processes=False`` layout) all target the same card, so the VRAM
# is *partitioned* by the number of GPU worker threads rather than serialized:
#   * 1 thread  -> it is admitted alone and budgeted the whole card (full VRAM).
#   * N threads -> up to N run concurrently, each budgeted ~1/N of the card, so
#     their fine grids sum to <= the card and none races another into OOM.
# The per-transform budget (VRAM/N) feeds the fine-grid tiler, which splits each
# transform to fit its share and keeps everything on the GPU. Separate worker
# *processes* (each its own CUDA context) coordinate through this only within a
# process; across processes the per-share budget still bounds each one.
_GPU_ADMIT_COND = threading.Condition()
_gpu_inflight = 0                 # GPU transforms currently in flight (live share count)
_gpu_reserved = 0                 # VRAM bytes currently reserved by in-flight tiles


def _free_cupy_pool_blocks() -> None:
    """Best-effort: return CuPy pool blocks to the device. Used after a
    plan cache flush so VRAM held by destroyed plans is actually reclaimable
    by subsequent allocations."""
    try:
        import cupy as cp_mod  # type: ignore
        cp_mod.get_default_memory_pool().free_all_blocks()
        try:
            cp_mod.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    except Exception:
        pass


_CUPY_POOL_CAPPED = False
# Default per-worker target = 60% of total VRAM divided by expected worker
# count. Each Dask worker process has its OWN CuPy pool, so a 65% per-worker
# cap on N workers asks for N*0.65 of VRAM total, which OOMs. We split the
# 60% global budget across workers and account for ~30% headroom held by
# cuFINUFFT plan scratch (which bypasses the pool via raw cudaMalloc).
_DEFAULT_CUPY_POOL_GLOBAL_BUDGET_PCT = 0.60
_DEFAULT_NON_POOL_HEADROOM_PCT = 0.30

# Absolute VRAM headroom: bytes that must stay FREE on the card at all times.
# Every budget in this module (reservation pool, CuPy pool cap, lattice plan
# budgets) keys off this ONE number, and the reservation pool is *split* among
# in-flight transforms, so the guaranteed-free floor neither shrinks nor
# multiplies with the number of worker threads sharing the GPU.
_GPU_TOTAL_OCCUPANCY_CEILING = 0.85
_MIN_GPU_HEADROOM_BYTES = 2 << 30


def _headroom_bytes_for_total(total_vram: int) -> int:
    """Absolute VRAM to keep free for a card of ``total_vram`` bytes.

    ``MOSAIC_GPU_HEADROOM_GIB`` (float GiB) overrides; the default is the
    larger of ``(1 - occupancy ceiling) * total`` and 2 GiB, so big cards keep
    the historical 15% margin and small cards keep a usable absolute floor."""
    if total_vram <= 0:
        return 0
    raw = os.getenv("MOSAIC_GPU_HEADROOM_GIB")
    if raw:
        try:
            return max(0, int(float(raw) * (1 << 30)))
        except ValueError:
            logger.warning("Ignoring invalid MOSAIC_GPU_HEADROOM_GIB=%r", raw)
    derived = (1.0 - _GPU_TOTAL_OCCUPANCY_CEILING) * float(total_vram)
    return int(max(derived, _MIN_GPU_HEADROOM_BYTES))


def _expected_worker_count() -> int:
    """Best-effort guess at how many workers share this GPU. Used to split
    the global VRAM budget into per-worker pool caps."""
    raw = os.getenv("MOSAIC_DASK_WORKER_COUNT")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    raw = os.getenv("DASK_MAX_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 4  # matches the project default


def _apply_cupy_pool_cap() -> None:
    """Bound CuPy's default memory pool **per worker process** so the
    cumulative pool footprint across workers stays inside the GPU.

    Resolution order (first match wins):
        1. ``MOSAIC_CUPY_POOL_LIMIT_BYTES``   — absolute per-worker cap
        2. ``MOSAIC_CUPY_POOL_LIMIT_GIB``     — per-worker cap in GiB
        3. ``MOSAIC_CUPY_POOL_LIMIT_PCT``     — per-worker fraction of total
        4. Auto: ``(global_budget - non_pool_headroom) / N_workers``
           where global_budget = 0.60 of total VRAM and non_pool_headroom =
           0.30 of total VRAM (reserved for cuFINUFFT plan internals which
           bypass the CuPy pool).

    On a 32 GiB GPU with 4 workers, the auto cap is
        (0.60 - 0.30) * 32 / 4 = 2.4 GiB per worker pool → ~10 GiB pool
    aggregate + ~10 GiB plan scratch + ~12 GiB free = 60-70% steady-state.

    Idempotent; safe when CuPy is absent.
    """
    global _CUPY_POOL_CAPPED
    if _CUPY_POOL_CAPPED:
        return

    limit_bytes: int | None = None
    raw_bytes = os.getenv("MOSAIC_CUPY_POOL_LIMIT_BYTES")
    raw_gib = os.getenv("MOSAIC_CUPY_POOL_LIMIT_GIB")
    raw_pct = os.getenv("MOSAIC_CUPY_POOL_LIMIT_PCT")

    if raw_bytes:
        try:
            limit_bytes = max(0, int(raw_bytes))
        except ValueError:
            limit_bytes = None

    if limit_bytes is None and raw_gib:
        try:
            limit_bytes = max(0, int(float(raw_gib) * (1 << 30)))
        except ValueError:
            limit_bytes = None

    try:
        import cupy as cp_mod  # type: ignore
        try:
            _free, total_vram = cp_mod.cuda.runtime.memGetInfo()
        except Exception:
            total_vram = 0
    except Exception:
        return  # CuPy unavailable; nothing to cap

    if total_vram <= 0:
        return

    if limit_bytes is None:
        if raw_pct:
            try:
                pct = float(raw_pct)
            except ValueError:
                pct = _DEFAULT_CUPY_POOL_GLOBAL_BUDGET_PCT
            pct = min(max(pct, 0.0), 0.95)
            limit_bytes = int(pct * total_vram)
        else:
            # Per-worker splitting only applies to PROCESS workers, where each
            # process owns its own CuPy pool. Threads-mode Dask (the project
            # default, DASK_PROCESSES=0) runs every worker in THIS process
            # sharing ONE pool -- dividing the global budget by worker count
            # shrank the shared pool N-fold (32 GiB card: 2.4 GiB pool, slab
            # and tile sizing strangled, hkl40 batches ~10x slower).
            n_workers = (
                _expected_worker_count()
                if os.getenv("DASK_PROCESSES", "0") == "1"
                else 1
            )
            usable_pct = max(
                0.05,
                _DEFAULT_CUPY_POOL_GLOBAL_BUDGET_PCT - _DEFAULT_NON_POOL_HEADROOM_PCT,
            )
            limit_bytes = int(usable_pct * total_vram / max(1, n_workers))
            # The pool cap alone must not plan past the absolute headroom
            # (matters on small cards where 30% of total can exceed
            # total - headroom); explicit env overrides above stay authoritative.
            limit_bytes = min(
                limit_bytes,
                max(
                    int(total_vram - _headroom_bytes_for_total(total_vram)),
                    int(0.05 * total_vram),
                ),
            )

    if limit_bytes is None or limit_bytes <= 0:
        return

    try:
        cp_mod.get_default_memory_pool().set_limit(size=int(limit_bytes))
        _CUPY_POOL_CAPPED = True
        logger.info(
            "CuPy pool capped at %.2f GiB (workers=%d, total_vram=%.2f GiB).",
            limit_bytes / (1 << 30),
            _expected_worker_count(),
            total_vram / (1 << 30),
        )
    except Exception as exc:
        logger.debug("Could not set CuPy pool limit: %s", exc)


# Deliberately do not touch CuPy/CUDA at import time.  Dask-CUDA imports the
# workflow in its launcher before assigning each worker's visible device; an
# import-time context would therefore be inherited on GPU 0 by every worker.
# The cap is applied lazily by ``_ensure_gpu_backend`` after worker assignment.
# Fixed, deterministic OOM back-off ladder for ``gpu_maxsubprobsize``.
# Selection is intentionally history-independent: every call starts at the
# same largest subproblem size and, on out-of-memory, backs off through this
# exact sequence.  We deliberately do NOT cache a last-known-good size across
# calls — caching made the subprob path depend on prior-call history (process
# state), so identical inputs could follow different ladders in different
# processes.  With a fixed ladder, identical inputs always follow an identical
# subprob path regardless of process history.  OOM recovery is unchanged: the
# back-off ladder (and the one cache-flush retry per subprob) is preserved.
_DEFAULT_SUBPROBS: tuple = (32, 16, 8, 4, 2, 1)


def _subprob_order(dim: int, n_trans: int) -> tuple:
    """Return the deterministic OOM back-off ladder.

    History-independent by design: always the same fixed sequence, so the
    subprob path a given input follows does not depend on what earlier calls
    in this process happened to succeed at.  ``dim``/``n_trans`` are accepted
    for call-site stability but do not influence the order.
    """
    return _DEFAULT_SUBPROBS


def _destroy_plan_quietly(plan) -> None:
    """Release a cuFINUFFT plan without invoking ``__del__`` directly.

    cuFINUFFT exposes the C destroy callback and handle on the Python Plan.
    Calling ``plan.__del__()`` manually is unsafe because the Python wrapper
    remains live and may be finalized again later. Destroy the handle and
    poison the wrapper state instead, matching the library finalizer's
    idempotency contract.
    """
    if plan is None:
        return
    destroy_plan = getattr(plan, "_destroy_plan", None)
    handle = getattr(plan, "_plan", None)
    if not callable(destroy_plan) or handle is None:
        return
    destroyed = False
    try:
        status = destroy_plan(handle)
        if status:
            logger.debug("cuFINUFFT plan destroy returned status %s", status)
        else:
            destroyed = True
    except Exception as exc:
        logger.debug("cuFINUFFT plan destroy swallowed error: %s", exc)
    if destroyed:
        try:
            plan._plan = None
        except Exception:
            pass
        try:
            plan._references = []
        except Exception:
            pass


def _clear_plan_cache() -> None:
    """Drop all cached plans and return CuPy pool blocks to the device.
    Called on CuPy OOM or external pool flush."""
    with _PLAN_CACHE_LOCK:
        victims = list(_PLAN_CACHE.values())
        _PLAN_CACHE.clear()
        _PLAN_CACHE_ORDER.clear()
    for plan, per_plan_lock in victims:
        with per_plan_lock:
            _destroy_plan_quietly(plan)
    _free_cupy_pool_blocks()


def _evict_one_locked() -> None:
    if not _PLAN_CACHE_ORDER:
        return
    oldest = _PLAN_CACHE_ORDER.pop(0)
    victim = _PLAN_CACHE.pop(oldest, None)
    if victim is not None:
        plan, per_plan_lock = victim
        with per_plan_lock:
            _destroy_plan_quietly(plan)
        # Return CuPy pool blocks held by the destroyed plan so the next
        # allocation can actually use the freed VRAM.
        _free_cupy_pool_blocks()


def _plan_cache_get_or_build(key: tuple, builder):
    """Return ``(plan, per_plan_lock, cached)``.

    When ``_PLAN_CACHE_MAX <= 0`` the cache is disabled — a fresh Plan is
    built and the caller is responsible for destroying it once done. This
    is the default and the safe choice on a single GPU shared by multiple
    Dask worker processes: cuFINUFFT plan scratch is allocated via raw
    ``cudaMalloc`` and **bypasses the CuPy pool**, so leaving plans alive
    pins VRAM that no pool cap can reclaim.
    """
    if _PLAN_CACHE_MAX <= 0:
        plan = builder()
        return (plan, threading.Lock(), False)

    with _PLAN_CACHE_LOCK:
        entry = _PLAN_CACHE.get(key)
        if entry is not None:
            try:
                _PLAN_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            _PLAN_CACHE_ORDER.append(key)
            plan, per_plan_lock = entry
            return (plan, per_plan_lock, True)

    plan = builder()
    per_plan_lock = threading.Lock()

    with _PLAN_CACHE_LOCK:
        existing = _PLAN_CACHE.get(key)
        if existing is not None:
            _destroy_plan_quietly(plan)
            try:
                _PLAN_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            _PLAN_CACHE_ORDER.append(key)
            existing_plan, existing_lock = existing
            return (existing_plan, existing_lock, True)
        while len(_PLAN_CACHE_ORDER) >= max(1, _PLAN_CACHE_MAX):
            _evict_one_locked()
        _PLAN_CACHE[key] = (plan, per_plan_lock)
        _PLAN_CACHE_ORDER.append(key)
        return (plan, per_plan_lock, True)


def _plan_cache_stats() -> dict:
    with _PLAN_CACHE_LOCK:
        return {
            "size": len(_PLAN_CACHE_ORDER),
            "max": _PLAN_CACHE_MAX,
            "keys": list(_PLAN_CACHE_ORDER),
        }

###############################################################################
#  Global CPU-only switch                                                     #
###############################################################################
_CPU_ONLY = os.getenv("MOSAIC_NUFFT_CPU_ONLY", "0") == "1"

def set_cpu_only(flag: bool = True) -> None:
    """
    Force wrapper into CPU-only mode (or re-enable GPU when False).

    Safe to call at ANY time, including from an error handler while sibling
    threads are mid-transform: the module-global ``cp`` reference is left
    intact so in-flight GPU code (device arrays, ``except cp.cuda...``
    clauses) keeps working; only the mode flags flip, and every NEW transform
    checks them. Nulling ``cp`` here is what used to crash concurrent threads
    with ``'NoneType' object has no attribute 'cuda'`` and demote whole
    workers to CPU on a single transient GPU error.
    """
    global _CPU_ONLY, _GPU_AVAILABLE, _GPU_PROBED
    _CPU_ONLY = bool(flag)
    if _CPU_ONLY:
        _GPU_AVAILABLE = False
        _GPU_PROBED = True
        return
    _probe_gpu_backend()


###############################################################################
#  CUDA / CuPy import with graceful degradation                               #
###############################################################################
cp = None                             # type: ignore
_GPU_AVAILABLE = False
_GPU_PROBED = False
_GPU_PROBE_LOCK = threading.Lock()


def _probe_gpu_backend() -> None:
    global cp, _GPU_AVAILABLE, _GPU_PROBED
    if _CPU_ONLY:
        cp = None                     # type: ignore
        _GPU_AVAILABLE = False
        _GPU_PROBED = True
        return
    try:
        import cupy as _cp            # noqa: E402

        cp = _cp                      # type: ignore
        try:
            _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            _GPU_AVAILABLE = False
    except ImportError:
        cp = None                     # type: ignore
        _GPU_AVAILABLE = False
    _GPU_PROBED = True


def _ensure_gpu_backend() -> None:
    """Initialize CUDA only when GPU work begins inside the assigned worker."""
    if _CPU_ONLY:
        return
    # Tests and embedders may inject an already-live backend directly.
    if not _GPU_PROBED and not _GPU_AVAILABLE:
        with _GPU_PROBE_LOCK:
            if not _GPU_PROBED and not _GPU_AVAILABLE:
                _probe_gpu_backend()
            if _GPU_PROBED and _GPU_AVAILABLE:
                _apply_cupy_pool_cap()
        return
    if _GPU_PROBED and _GPU_AVAILABLE and not _CUPY_POOL_CAPPED:
        with _GPU_PROBE_LOCK:
            if not _CUPY_POOL_CAPPED:
                _apply_cupy_pool_cap()

# Lazily imported CPU backend (avoid importing finufft on GPU-only nodes)
_FINUFFT3: dict[int, Callable] | None = None
_DIRECT_CPU_FALLBACK_WARNED = False
_DIRECT_CPU_FALLBACK_MAX_TARGETS = 50_000

###############################################################################
#  Memory helpers                                                             #
###############################################################################
_SCRATCH_ALPHA = 1.1          # cuFINUFFT scratch ≈ 1 complex128 per target
_GRID_LOWER_BOUND_ALPHA = 2.5
_GRID_WORKSPACE_FLOOR_BYTES = 64 << 20


def _free_mem_bytes() -> int:
    """Current free VRAM in bytes (0 if GPU unavailable)."""
    if not _GPU_AVAILABLE:
        return 0
    with cp.cuda.Device(0):
        free, _ = cp.cuda.runtime.memGetInfo()
    return int(free)


def _total_mem_bytes() -> int:
    """Total VRAM in bytes (0 if GPU unavailable)."""
    if not _GPU_AVAILABLE:
        return 0
    with cp.cuda.Device(0):
        _, total = cp.cuda.runtime.memGetInfo()
    return int(total)


def _gpu_headroom_bytes() -> int:
    """Absolute VRAM kept free on this card (0 when no GPU is present or the
    device cannot be queried). See ``_headroom_bytes_for_total`` /
    ``MOSAIC_GPU_HEADROOM_GIB``."""
    try:
        total = _total_mem_bytes()
    except Exception:
        return 0
    return _headroom_bytes_for_total(total)


def _uncommitted_budget_growth_bytes() -> int:
    """VRAM that capped consumers (CuPy pool, type-1 plan cache) are still
    ENTITLED to take but have not allocated yet.

    Live free VRAM cannot see this future growth, so the reservation pool must
    treat it as already spent. Sizing reservations against free alone let the
    pool and cache fill AFTER fine-grid reservations were granted against the
    older, larger free reading -- measured free pinned at ~0.1 GiB on the
    32 GiB card during the hkl32 streaming run despite a 6 GiB headroom."""
    growth = 0
    try:
        pool = cp.get_default_memory_pool()
        limit = int(pool.get_limit() or 0)
        if limit > 0:
            growth += max(0, limit - int(pool.total_bytes()))
    except Exception:
        pass
    try:
        with _TYPE1_PLAN_CACHE_LOCK:
            cache_bytes = _type1_cache_total_bytes_locked()
        growth += max(0, _type1_plan_cache_max_bytes() - cache_bytes)
    except Exception:
        pass
    return growth


def _pool_reservable_bytes() -> int:
    """VRAM the tiler may hand out across all concurrent transforms *right now*.

    Based on live free VRAM plus what our own in-flight tiles already hold, so
    the pool reflects the real card minus anything external (another process, a
    notebook) is using -- we only ever promise VRAM we can actually provide.

    Additionally bounded so an absolute headroom (``_gpu_headroom_bytes``)
    always stays free: sizing purely from a fraction of free VRAM
    asymptotically fills the card (measured 98% on hkl40 -- CuPy pool at
    its cap + fine grids sized from the remaining free), and running at the
    rim turns every allocation into a potential churn/failure. The headroom is
    subtracted from the SHARED pool, not per transform, so the guaranteed-free
    floor is the same whether 1 or N worker threads are in flight. Future
    growth still owed to the CuPy pool cap and the type-1 plan cache is
    subtracted as well (``_uncommitted_budget_growth_bytes``), so the floor
    survives those consumers filling up after a reservation was granted."""
    free = _free_mem_bytes()
    live = free + _gpu_reserved
    total = _total_mem_bytes()
    if live <= 0:
        live = total
    committed = _gpu_headroom_bytes() + _uncommitted_budget_growth_bytes()
    ceiling_room = max(0, int(live - committed))
    return min(int(live * _gpu_vram_headroom_frac()), ceiling_room)


_LOW_FREE_LOG_LOCK = threading.Lock()
_LOW_FREE_LAST_LOG = 0.0


def _warn_if_below_headroom(context: str) -> None:
    """Watchdog: rate-limited WARNING whenever live free VRAM sits below the
    configured headroom, with the full budget breakdown. Names the allocation
    site that materialized a floor breach -- the breakdown separates ledger
    reservations, CuPy pool, and type-1 cache so unaccounted raw cudaMalloc
    shows up as the difference."""
    global _LOW_FREE_LAST_LOG
    if not _GPU_AVAILABLE:
        return
    try:
        free = _free_mem_bytes()
        headroom = _gpu_headroom_bytes()
        if headroom <= 0 or free >= headroom:
            return
        now = time.monotonic()
        with _LOW_FREE_LOG_LOCK:
            if now - _LOW_FREE_LAST_LOG < 5.0:
                return
            _LOW_FREE_LAST_LOG = now
        pool_used = pool_total = -1
        try:
            pool = cp.get_default_memory_pool()
            pool_used = int(pool.used_bytes())
            pool_total = int(pool.total_bytes())
        except Exception:
            pass
        try:
            with _TYPE1_PLAN_CACHE_LOCK:
                cache_bytes = _type1_cache_total_bytes_locked()
        except Exception:
            cache_bytes = -1
        logger.warning(
            "GPU free below headroom after %s | free=%.2f GiB headroom=%.2f GiB "
            "ledger_reserved=%.2f GiB pool_used=%.2f GiB pool_total=%.2f GiB "
            "type1_cache_est=%.2f GiB inflight=%d",
            context,
            free / 2**30,
            headroom / 2**30,
            _gpu_reserved / 2**30,
            pool_used / 2**30,
            pool_total / 2**30,
            cache_bytes / 2**30,
            _gpu_inflight,
        )
    except Exception:
        pass


def _transform_enter() -> None:
    """Register a GPU transform as in-flight (raises the live sharing count)."""
    global _gpu_inflight
    with _GPU_ADMIT_COND:
        _gpu_inflight += 1
        _GPU_ADMIT_COND.notify_all()


def _transform_exit() -> None:
    global _gpu_inflight
    with _GPU_ADMIT_COND:
        _gpu_inflight = max(0, _gpu_inflight - 1)
        _GPU_ADMIT_COND.notify_all()


def _current_tile_budget() -> int:
    """VRAM a single tile may claim *right now*: an equal share of the reservable
    pool among the transforms currently in flight. Re-read per tile, so it tracks
    workers arriving/leaving dynamically -- a lone worker gets the whole pool; the
    moment a sibling starts, both converge to half on their next tile."""
    with _GPU_ADMIT_COND:
        active = max(1, _gpu_inflight)
        pool = _pool_reservable_bytes()
    return max(pool // active, _MIN_TILE_BUDGET_BYTES)


def _reserve_tile(nbytes: int) -> None:
    """Reserve ``nbytes`` of VRAM from the shared pool, blocking until it fits.

    Guarantees the sum of live tile reservations never exceeds the pool, so
    concurrent transforms cannot race each other into an out-of-memory. A tile is
    always admitted when nothing else is reserved (progress guarantee), even if
    it is momentarily larger than the pool."""
    global _gpu_reserved
    with _GPU_ADMIT_COND:
        while _gpu_reserved > 0 and _gpu_reserved + nbytes > _pool_reservable_bytes():
            _GPU_ADMIT_COND.wait()
        _gpu_reserved += nbytes
    _warn_if_below_headroom(f"reserve_tile({nbytes >> 20} MiB)")


def _release_tile(nbytes: int) -> None:
    global _gpu_reserved
    with _GPU_ADMIT_COND:
        _gpu_reserved = max(0, _gpu_reserved - nbytes)
        _GPU_ADMIT_COND.notify_all()


def free_gpu_memory() -> None:
    """Best-effort release of CuPy memory pools when GPU support is active."""
    if not _GPU_AVAILABLE or cp is None:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _max_chunk_size(
    *,
    free_bytes: int,
    baseline_bytes: int,
    per_target_bytes: int,
    mem_frac: float,
    user_cap: Optional[int],
    min_chunk: int,
) -> int:
    """
    Largest chunk that fits in    baseline + chunk*per_target
    ≤  free_bytes * mem_frac
    Returns 0 when nothing fits.
    """
    budget = int(free_bytes * mem_frac) - baseline_bytes
    if budget <= 0:
        return 0
    chunk = budget // per_target_bytes
    if user_cap is not None:
        chunk = min(chunk, user_cap)
    if chunk < min_chunk:
        return 0
    return int(chunk)


###############################################################################
#  Misc helpers                                                               #
###############################################################################
def _as_device(arr: np.ndarray, *, allow_fail: bool = False):
    """Copy numpy array to GPU (cupy)."""
    if not _GPU_AVAILABLE:
        if allow_fail:
            return None
        raise RuntimeError("No CUDA device available")
    try:
        return cp.asarray(arr)
    except (cp.cuda.memory.OutOfMemoryError,
            cp.cuda.driver.CUDADriverError,
            cp.cuda.runtime.CUDARuntimeError,
            MemoryError):
        if allow_fail:
            return None
        raise


def _contig(x):
    """Ensure C-contiguous cupy array."""
    return x if x.flags.c_contiguous else cp.ascontiguousarray(x)


def _estimate_grid_bytes(real: np.ndarray, recip: np.ndarray) -> int:
    """
    Lower-bound proxy for the fine grid size (complex128 only).
    cuFINUFFT allocates additional hidden work arrays, so callers should treat
    this as a base signal and add safety overhead rather than as an exact
    residency model.
    """
    xyz = np.abs(np.vstack((real, recip))).max(axis=0)
    nf = ((2 * np.ceil(xyz) + 2 + 15) // 16) * 16
    return int(nf.prod()) * 16          # 16 B per complex128


# cuFINUFFT type-3 upsampling factor (sigma) used to size the internal fine grid.
_TYPE3_UPSAMPFAC = 2.0

# cuFINUFFT allocates more than just the fine grid: a cuFFT workspace (~1x the
# grid) plus spread/sort scratch. Measured peak residency is ~2.2-2.5x the bare
# fine grid, so tiling budgets the fine grid times this factor to avoid OOM.
_TYPE3_RESIDENCY_FACTOR = 2.5


def fine_grid_bytes_type3(real: np.ndarray, recip: np.ndarray) -> int:
    """Estimate the cuFINUFFT type-3 *fine grid* residency (complex128).

    A type-3 transform builds an intermediate uniform grid whose per-axis size
    scales as the **product of the coordinate spreads** on that axis:
    ``nf_i ~ sigma * spread_real_i * spread_recip_i / pi`` plus spread padding
    (``spread = max - min``). That grid is allocated by cuFINUFFT via raw
    ``cudaMalloc``, bypassing the CuPy pool budget, so it must be sized directly.

    The **spread** (not per-axis ``max``) is the load-bearing quantity: it is
    what shrinks when the source q-points (or targets) are split into spatial
    tiles, which is exactly how :func:`_type3_inverse_gpu_tiled` keeps each GPU
    sub-transform inside VRAM. Concatenating every interval's q-points into one
    transform makes the recip spread the *entire* reciprocal range (~66) and the
    real spread the full supercell (~27) -> nf ~1170/axis -> ~22 GiB; tiling the
    sources restores the small per-tile spreads (and small grids) of the
    per-interval regime while keeping the batched GPU launch.
    """
    real = np.asarray(real)
    recip = np.asarray(recip)
    if real.ndim != 2 or recip.ndim != 2 or real.shape[1] != recip.shape[1] or len(real) == 0 or len(recip) == 0:
        return _estimate_grid_bytes(real, recip)
    x_spread = real.max(axis=0) - real.min(axis=0)
    s_spread = recip.max(axis=0) - recip.min(axis=0)
    nf = np.ceil((_TYPE3_UPSAMPFAC / np.pi) * x_spread * s_spread) + 32.0
    nf = np.ceil(nf / 16.0) * 16.0
    return int(np.prod(nf)) * 16          # 16 B per complex128


def _gpu_vram_headroom_frac() -> float:
    """Fraction of *free* VRAM one type-3 GPU sub-transform's fine grid may
    claim. Sub-transforms larger than this are split (tiled) so they stay on the
    GPU. Keeps margin for cuFINUFFT work arrays / fragmentation. Override with
    ``MOSAIC_NUFFT_GPU_VRAM_HEADROOM`` (0 < frac <= 1)."""
    raw = os.getenv("MOSAIC_NUFFT_GPU_VRAM_HEADROOM")
    if raw is None or str(raw).strip() == "":
        return 0.85
    try:
        frac = float(raw)
    except (TypeError, ValueError):
        return 0.85
    if not (0.0 < frac <= 1.0):
        return 0.85
    return frac


# Hard ceiling on tiling recursion depth (2**depth tiles) -- a runaway guard far
# above any real workload; termination normally comes from the fine-grid budget.
_MAX_TILE_DEPTH = 24

# Floor on the per-tile fine-grid budget. Prevents pathological over-splitting
# into thousands of tiny transforms when free VRAM is momentarily scarce (e.g.
# a second worker holds the card); a ~1 GiB fine grid is already an efficient
# GPU transform. If a tile this size still will not fit, the leaf executor's own
# OOM handling is the final backstop.
_MIN_TILE_BUDGET_BYTES = 1 << 30


def _type3_inverse_gpu_tiled(
    *,
    real_coords: np.ndarray,
    q_coords: np.ndarray,
    weights_arr: np.ndarray,
    eps: float,
    leaf,
    budget_bytes: int,
) -> np.ndarray:
    """Split a type-3 inverse transform along the widest coordinate spread until
    each leaf's fine grid fits ``budget_bytes``, run every leaf on the GPU, and
    recombine exactly into a single preallocated output.

    The type-3 inverse is linear in the sources, so splitting the q-points into
    spatial groups and **summing** the per-group results is exact; splitting the
    real-space targets partitions the output. Splitting along the axis of largest
    spread shrinks the fine grid (``nf_i ~ spread_real_i * spread_recip_i``), so a
    handful of splits turns one VRAM-busting transform into several GPU-sized ones
    -- no CPU fallback.

    Results accumulate **in place** into one ``(n_trans, n_targets)`` buffer, so
    host memory stays at ~one output plus one live tile regardless of the split
    depth (a tree-of-sums would instead hold O(depth) full-size arrays -- fatal
    for the 10^8-target 3D cases). ``leaf(q_sub, w_sub, real_sub) ->
    (n_trans, len(real_sub))`` runs one GPU-sized sub-transform (host result).
    """
    n_trans = int(weights_arr.shape[0])
    out = np.zeros((n_trans, int(len(real_coords))), dtype=np.complex128)
    target_index = np.arange(int(len(real_coords)))

    def _accumulate(tgt_idx, real_sub, q_sub, w_sub, depth):
        budget = int(budget_bytes() if callable(budget_bytes) else budget_bytes)
        residency = int(fine_grid_bytes_type3(real_sub, q_sub) * _TYPE3_RESIDENCY_FACTOR)
        if (residency <= budget or depth >= _MAX_TILE_DEPTH
                or (len(q_sub) <= 1 and len(real_sub) <= 1)):
            out[:, tgt_idx] += leaf(q_sub, w_sub, real_sub)
            return

        s_spread = q_sub.max(axis=0) - q_sub.min(axis=0)
        x_spread = real_sub.max(axis=0) - real_sub.min(axis=0)
        split_source = (
            (float(s_spread.max()) >= float(x_spread.max()) and len(q_sub) > 1)
            or len(real_sub) <= 1
        )

        if split_source and len(q_sub) > 1:
            ax = int(np.argmax(s_spread))
            coord = q_sub[:, ax]
            pivot = float(np.median(coord))
            lo = coord < pivot
            if not lo.any() or lo.all():          # ties defeat the median split
                lo = coord <= pivot
            if lo.any() and not lo.all():
                # Both source halves accumulate into the SAME targets (exact sum).
                _accumulate(tgt_idx, real_sub, q_sub[lo], w_sub[:, lo], depth + 1)
                _accumulate(tgt_idx, real_sub, q_sub[~lo], w_sub[:, ~lo], depth + 1)
                return

        if len(real_sub) > 1:
            ax = int(np.argmax(x_spread))
            coord = real_sub[:, ax]
            pivot = float(np.median(coord))
            lo = coord < pivot
            if not lo.any() or lo.all():
                lo = coord <= pivot
            if lo.any() and not lo.all():
                # Disjoint target slices -> map through the running target index.
                _accumulate(tgt_idx[lo], real_sub[lo], q_sub, w_sub, depth + 1)
                _accumulate(tgt_idx[~lo], real_sub[~lo], q_sub, w_sub, depth + 1)
                return

        # Could not split further (both sides degenerate); run as-is.
        out[:, tgt_idx] += leaf(q_sub, w_sub, real_sub)

    _accumulate(target_index, real_coords, q_coords, weights_arr, 0)
    return out


def _estimate_launch_bytes(real: np.ndarray, recip: np.ndarray) -> int:
    lower_bound = _estimate_grid_bytes(real, recip)
    scaled = int(lower_bound * _GRID_LOWER_BOUND_ALPHA)
    return max(scaled, lower_bound + _GRID_WORKSPACE_FLOOR_BYTES)


def _adaptive_reserve_bytes(*, free_bytes: int, resident_bytes: int) -> int:
    free_gib = float(free_bytes) / float(1 << 30)
    if free_gib <= 1.5:
        reserve = 768 << 20
    elif free_gib <= 3.0:
        reserve = 1 << 30
    elif free_gib <= 6.0:
        reserve = int(1.25 * (1 << 30))
    elif free_gib <= 12.0:
        reserve = int(1.5 * (1 << 30))
    elif free_gib <= 24.0:
        reserve = 2 << 30
    else:
        reserve = 3 << 30

    resident_gib = float(resident_bytes) / float(1 << 30)
    if resident_gib >= 8.0:
        reserve += 1 << 30
    elif resident_gib >= 4.0:
        reserve += 512 << 20
    elif resident_gib >= 2.0:
        reserve += 256 << 20
    # The chunked type-3 paths size purely from free VRAM and never touch the
    # reservation ledger; lifting their reserve to the absolute headroom keeps
    # them out of the guaranteed-free band too. The 0.85*free clamp below
    # stays as the make-progress escape when free is already below headroom.
    reserve = max(reserve, _gpu_headroom_bytes())
    return int(min(max(reserve, 256 << 20), int(max(free_bytes * 0.85, 0))))


def _resolve_budget_policy(
    *,
    mem_frac: Optional[float],
    free_bytes: int,
    resident_bytes: int,
) -> tuple[int, float, str]:
    if mem_frac is not None:
        effective_mem_frac = float(np.clip(float(mem_frac), 0.05, 0.8))
        return 0, effective_mem_frac, "explicit"
    reserve_bytes = _adaptive_reserve_bytes(
        free_bytes=free_bytes,
        resident_bytes=resident_bytes,
    )
    usable_bytes = max(0, free_bytes - reserve_bytes)
    effective_mem_frac = float(np.clip(usable_bytes / float(max(free_bytes, 1)), 0.05, 0.8))
    return reserve_bytes, effective_mem_frac, "adaptive-reserve-default"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer %s=%r", name, raw)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid boolean %s=%r", name, raw)
    return default


def _experimental_overlap_enabled() -> bool:
    return _env_bool("MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP", False)


def _telemetry_enabled() -> bool:
    return _env_bool("MOSAIC_NUFFT_CAPTURE_TELEMETRY", False)


def _begin_telemetry(**fields):
    if not _telemetry_enabled():
        return None
    telemetry = dict(fields)
    telemetry.setdefault("chunks", [])
    telemetry.setdefault("fallback_reason", None)
    telemetry.setdefault("final_d2h_bytes", 0)
    telemetry.setdefault("used_pinned_host_copy", False)
    telemetry.setdefault("total_upload_seconds", 0.0)
    telemetry.setdefault("total_launch_seconds", 0.0)
    telemetry.setdefault("total_download_seconds", 0.0)
    return telemetry


def _record_chunk_telemetry(telemetry, **fields) -> None:
    if telemetry is None:
        return
    telemetry["chunks"].append(dict(fields))
    telemetry["total_upload_seconds"] += float(fields.get("upload_seconds", 0.0))
    telemetry["total_launch_seconds"] += float(fields.get("launch_seconds", 0.0))
    telemetry["total_download_seconds"] += float(fields.get("download_seconds", 0.0))
    telemetry["used_pinned_host_copy"] = bool(telemetry["used_pinned_host_copy"]) or bool(
        fields.get("used_pinned_host_copy", False)
    )


def _finish_telemetry(telemetry, *, fallback_reason: str | None = None, final_d2h_bytes: int | None = None):
    global _LAST_NUFFT_TELEMETRY
    if telemetry is None:
        return
    if fallback_reason is not None:
        telemetry["fallback_reason"] = fallback_reason
    if final_d2h_bytes is not None:
        telemetry["final_d2h_bytes"] = int(final_d2h_bytes)
    telemetry["chunk_count"] = len(telemetry["chunks"])
    telemetry["full_target_fit_in_one_chunk"] = bool(
        telemetry["chunk_count"] == 1
        and telemetry["chunks"][0]["chunk_size"] == telemetry.get("n_targets", -1)
    ) if telemetry["chunks"] else False
    _LAST_NUFFT_TELEMETRY = telemetry


def get_last_nufft_telemetry():
    return copy.deepcopy(_LAST_NUFFT_TELEMETRY)


def _resolve_optional_gpu_stream():
    mode = os.getenv("MOSAIC_NUFFT_GPU_STREAM", "").strip().lower()
    if not mode:
        return None
    if not _GPU_AVAILABLE or cp is None:
        logger.debug("gpu_stream requested but GPU backend is unavailable")
        return None
    if mode != "current":
        logger.warning("Ignoring unsupported MOSAIC_NUFFT_GPU_STREAM=%r", mode)
        return None
    try:
        stream = cp.cuda.get_current_stream()
        ptr = int(getattr(stream, "ptr"))
        logger.debug("Using current CUDA stream for cuFINUFFT launches: %d", ptr)
        return ptr
    except Exception as exc:
        logger.debug("Could not resolve current CUDA stream: %s", exc)
        return None


def _build_gpu_launch_kwargs(*, gpu_maxsubprobsize: int) -> dict:
    kwargs = dict(
        gpu_method=_env_int("MOSAIC_NUFFT_GPU_METHOD", 1),
        gpu_kerevalmeth=_env_int("MOSAIC_NUFFT_GPU_KEREVALMETH", 1),
        gpu_maxsubprobsize=int(gpu_maxsubprobsize),
        gpu_maxbatchsize=_env_int("MOSAIC_NUFFT_GPU_MAXBATCHSIZE", 1),
        gpu_spreadinterponly=int(_env_bool("MOSAIC_NUFFT_GPU_SPREADINTERPONLY", True)),
    )
    gpu_stream = _resolve_optional_gpu_stream()
    if gpu_stream is not None:
        kwargs["gpu_stream"] = gpu_stream
    logger.debug("cuFINUFFT launch kwargs: %s", kwargs)
    return kwargs


def _is_retryable_resource_error(exc: Exception) -> bool:
    if cp is not None and isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "budget",
            "out of memory",
            "memory allocation",
            "cuda_error_out_of_memory",
            "cudaerroroutofmemory",
            "shared memory",
            "insufficient resources",
            "too many resources requested",
            "launch-resource-exhausted",
            "allocate",
        )
    )


def _is_retryable_super_batch_error(exc: Exception) -> bool:
    return _is_retryable_resource_error(exc)


def _copy_device_to_host_with_meta(device_arr):
    if cp is None:
        host = np.asarray(device_arr)
        return host, False, int(host.nbytes)
    use_pinned = _env_bool("MOSAIC_NUFFT_PINNED_HOST", False)
    if not use_pinned:
        host = cp.asnumpy(device_arr)
        return host, False, int(host.nbytes)
    try:
        import cupyx  # type: ignore

        host = cupyx.empty_pinned(device_arr.shape, dtype=device_arr.dtype)
        cp.asnumpy(device_arr, out=host)
        logger.debug("Copied device result to pinned host buffer with shape %s", device_arr.shape)
        return host, True, int(host.nbytes)
    except Exception as exc:
        logger.debug("Pinned host copy unavailable, falling back to standard host copy: %s", exc)
        host = cp.asnumpy(device_arr)
        return host, False, int(host.nbytes)


def _copy_device_to_host(device_arr):
    host, _, _ = _copy_device_to_host_with_meta(device_arr)
    return host


def _log_chunk_timing(
    *,
    label: str,
    upload_s: float,
    launch_s: float,
    download_s: float,
    chunk_size: int,
    used_pinned: bool,
    experimental_overlap: bool,
) -> None:
    logger.debug(
        "%s timings | chunk=%d upload=%.6fs launch=%.6fs download=%.6fs pinned=%s experimental_overlap=%s",
        label,
        chunk_size,
        upload_s,
        launch_s,
        download_s,
        used_pinned,
        experimental_overlap,
    )


def _select_type3_sides(
    real_coords: np.ndarray,
    weights: np.ndarray,
    q_coords: np.ndarray,
    *,
    inverse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if inverse:
        return q_coords, weights, real_coords
    return real_coords, weights, q_coords


def _per_target_bytes(dim: int, n_trans: int = 1) -> int:
    bytes_target_coords = dim * 8
    bytes_output = 16 * int(n_trans)
    bytes_scratch = int(_SCRATCH_ALPHA * 16 * int(n_trans))
    return bytes_target_coords + bytes_output + bytes_scratch


def _resident_bytes(
    resident_coords: np.ndarray,
    resident_weights: np.ndarray,
    *,
    n_trans: int = 1,
) -> int:
    data = int(resident_coords.shape[0] * resident_coords.shape[1] * 8)
    weights = int(np.asarray(resident_weights).size * 16)
    source_scratch = int(resident_coords.shape[0] * _SCRATCH_ALPHA * 16 * int(n_trans))
    return data + weights + source_scratch


def _plan_target_chunk(
    *,
    resident_coords: np.ndarray,
    target_coords: np.ndarray,
    start: int,
    free_bytes: int,
    budget_fraction: float,
    min_chunk: int,
    max_chunk: Optional[int],
    incremental_launch_baseline_bytes: int,
    n_trans: int = 1,
) -> tuple[int, int]:
    remaining = len(target_coords) - start
    if remaining <= 0:
        return 0, 0
    candidate = remaining if max_chunk is None else min(remaining, int(max_chunk))
    while candidate > 0:
        target_slice = target_coords[start : start + candidate]
        # If the whole remaining workload is smaller than the configured
        # minimum chunk, still allow it to run as a single chunk.
        effective_min_chunk = min(min_chunk, candidate)
        grid_bytes = _estimate_launch_bytes(resident_coords, target_slice)
        chunk = _max_chunk_size(
            free_bytes=free_bytes,
            baseline_bytes=incremental_launch_baseline_bytes + grid_bytes,
            per_target_bytes=_per_target_bytes(resident_coords.shape[1], n_trans),
            mem_frac=budget_fraction,
            user_cap=candidate,
            min_chunk=effective_min_chunk,
        )
        if chunk == 0:
            if candidate <= effective_min_chunk:
                return 0, 0
            candidate = max(effective_min_chunk, candidate // 2)
            continue
        if chunk < candidate:
            candidate = chunk
            continue
        return candidate, grid_bytes
    return 0, 0


###############################################################################
#  GPU kernel shortcuts                                                       #
###############################################################################
_KER = {}                              # type: ignore


def _ensure_gpu_kernels() -> None:
    global _KER
    if not _GPU_AVAILABLE or _KER:
        return
    import cufinufft                   # type: ignore

    _KER = {
        1: cufinufft.nufft1d3,
        2: cufinufft.nufft2d3,
        3: cufinufft.nufft3d3,
    }


###############################################################################
#  Public API                                                                 #
###############################################################################
def _resolve_weights(w: Optional[np.ndarray],
                     c: Optional[np.ndarray]) -> np.ndarray:
    if w is not None:
        return w
    if c is not None:
        return c
    raise ValueError("Provide 'weights' or its alias 'c'.")


def execute_cunufft(
    real_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    q_coords: np.ndarray | None = None,
    *,
    c: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 8*64_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Forward (Type-3) NUFFT: real → reciprocal."""
    weights = _resolve_weights(weights, c)
    if q_coords is None:
        raise ValueError("q_coords must be supplied for forward transform")
    return _batched_type3(
        real_coords, weights, q_coords,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
        eps=eps,
        inverse=False,
    )


def execute_inverse_cunufft(
    q_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    real_coords: np.ndarray | None = None,
    *,
    c: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32*256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse (Type-3) NUFFT: reciprocal → real."""
    weights = _resolve_weights(weights, c)
    if real_coords is None:
        raise ValueError("real_coords must be supplied for inverse transform")
    result = execute_inverse_cunufft_batch(
        q_coords=q_coords,
        weights=np.asarray(weights, dtype=np.complex128)[np.newaxis, :],
        real_coords=real_coords,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
        eps=eps,
    )
    return np.asarray(result[0], dtype=np.complex128)


def execute_inverse_cunufft_batch(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32 * 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse (Type-3) NUFFT: reciprocal → real for stacked weight vectors."""
    if real_coords is None:
        raise ValueError("real_coords must be supplied for inverse transform")
    weights_arr = np.asarray(weights, dtype=np.complex128)
    if weights_arr.ndim == 1:
        weights_arr = weights_arr[np.newaxis, :]
    if weights_arr.ndim != 2:
        raise ValueError("weights must be 1-D or 2-D with shape (n_trans, n_sources)")
    if weights_arr.shape[1] != len(q_coords):
        raise ValueError("weights shape must match q_coords on axis 1")
    return _execute_inverse_cunufft_batch(
        q_coords=q_coords,
        weights_arr=weights_arr,
        real_coords=real_coords,
        eps=eps,
        mem_frac=mem_frac,
        min_chunk=min_chunk,
        max_chunk=max_chunk,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
        device_out=False,
    )


def execute_inverse_cunufft_super_batch(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32 * 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
    max_batch_width: Optional[int] = None,
) -> np.ndarray:
    """Inverse type-3 helper that widens same-geometry batches when safe."""
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    if real_coords is None:
        raise ValueError("real_coords must be supplied for inverse transform")
    weights_arr = np.asarray(weights, dtype=np.complex128)
    if weights_arr.ndim == 1:
        weights_arr = weights_arr[np.newaxis, :]
    if weights_arr.ndim != 2:
        raise ValueError("weights must be 1-D or 2-D with shape (n_trans, n_sources)")
    if weights_arr.shape[1] != len(q_coords):
        raise ValueError("weights shape must match q_coords on axis 1")
    total_trans = int(weights_arr.shape[0])
    if total_trans == 0:
        return np.zeros((0, len(real_coords)), dtype=np.complex128)

    if max_batch_width is None:
        width = total_trans
    else:
        width = max(1, min(int(max_batch_width), total_trans))

    def _leaf(q_sub, w_sub, real_sub):
        return np.asarray(
            _execute_inverse_cunufft_batch(
                q_coords=q_sub,
                weights_arr=w_sub,
                real_coords=real_sub,
                eps=eps,
                mem_frac=mem_frac,
                min_chunk=min_chunk,
                max_chunk=max_chunk,
                prefer_cpu=prefer_cpu,
                gpu_only=gpu_only,
                device_out=False,
            )
        )

    def _reserved_leaf(q_sub, w_sub, real_sub):
        # Reserve this tile's VRAM from the shared pool so concurrent transforms
        # pack the card without over-committing; release as soon as it is done.
        tile_bytes = int(fine_grid_bytes_type3(real_sub, q_sub) * _TYPE3_RESIDENCY_FACTOR)
        _reserve_tile(tile_bytes)
        try:
            return _leaf(q_sub, w_sub, real_sub)
        finally:
            _release_tile(tile_bytes)

    # Tile the type-3 fine grid to fit VRAM and keep the work on the GPU (no
    # silent CPU fallback): the transform is linear, so splitting the q-sources
    # (sum) or real targets (partition) into spatial tiles is exact. Only when a
    # GPU is actually the execution target -- a CPU run gains nothing from tiling.
    use_gpu = _GPU_AVAILABLE and not _CPU_ONLY and not prefer_cpu
    real_arr = np.asarray(real_coords, dtype=np.float64)
    q_arr = np.asarray(q_coords, dtype=np.float64)

    outputs: list[np.ndarray] = []
    start = 0
    while start < total_trans:
        end = min(start + width, total_trans)
        batch_weights = weights_arr[start:end]
        try:
            if use_gpu:
                # Transforms run concurrently and share VRAM dynamically: each
                # tile claims an equal live share of the free pool (whole card
                # when alone, 1/N when N are in flight) and reserves it so the
                # concurrent set never over-commits. All work stays on the GPU.
                _transform_enter()
                try:
                    batch_result = _type3_inverse_gpu_tiled(
                        real_coords=real_arr,
                        q_coords=q_arr,
                        weights_arr=batch_weights,
                        eps=eps,
                        leaf=_reserved_leaf,
                        budget_bytes=_current_tile_budget,
                    )
                finally:
                    _transform_exit()
            else:
                batch_result = _leaf(q_coords, batch_weights, real_coords)
            outputs.append(np.asarray(batch_result))
            start = end
        except Exception as exc:
            if width > 1 and _is_retryable_super_batch_error(exc):
                width = max(1, width // 2)
                continue
            raise
    return np.concatenate(outputs, axis=0)


###############################################################################
#  Lattice (scatter + type-2) inverse path                                    #
#                                                                             #
#  MOSAIC's reciprocal-space points sit on a uniform per-axis lattice (h,k,l  #
#  at integer multiples of 1/N_cell); masks select a SUBSET of lattice sites  #
#  but never move points off the lattice. The inverse transform               #
#      F(r) = sum_q v(q) exp(-i r.q)                                          #
#  is therefore a type-2 NUFFT from a dense coefficient grid (masked-out      #
#  sites simply stay zero) instead of a type-3 over scattered points. This    #
#  removes the type-3 fine-grid blow-up entirely: cost is set by the mode     #
#  grid dimensions, not by (real extent x reciprocal extent). Measured on     #
#  hkl32: 27x per transform, ~42 min vs 14-31 h end-to-end.                   #
###############################################################################


def lattice_host_budget_bytes() -> int:
    """Host-RAM budget for a dense lattice coefficient grid.

    Env ``MOSAIC_RESIDUAL_LATTICE_HOST_BUDGET`` wins; the default scales with
    the machine -- 55% of physical RAM (floor 8 GiB, cap 96 GiB) -- so large
    grids (hkl40: ~31 GiB) take the fast lattice path on capable nodes while
    small boxes fall back to type-3 rather than swapping."""
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_HOST_BUDGET")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 0
    if total <= 0:
        return 24 << 30
    return int(min(max(int(total * 0.55), 8 << 30), 96 << 30))


def plan_lattice(q_coords: np.ndarray, *, snap_tol: float = 0.05,
                 host_budget_bytes: int | None = None, n_trans: int = 2,
                 max_snap_dev: float | None = None):
    """Snap scattered q-points onto a uniform per-axis lattice.

    Returns a meta dict ``{origin, dq, dims, snap_dev}`` when every point lies
    on a common per-axis lattice (within ``snap_tol`` of a step) AND the dense
    coefficient grid fits ``host_budget_bytes``; otherwise ``None`` (caller
    falls back to type-3). Degenerate axes (a single plane, e.g. the l=0
    zero-plane role, or a 2D projection) get ``dq=0`` and one mode.

    ``max_snap_dev`` additionally rejects NEAR-lattice data: truly on-lattice
    q (exact float64 products) snaps to <=1e-9 of a step after the LSQ
    refinement, while e.g. a slightly sheared cell produces deviations around
    1e-6 that ``snap_tol`` would silently accept -- evaluating the transform
    at the snapped positions then corrupts the result far beyond NUFFT eps.
    Callers that promise eps-level parity with type-3 must set it."""
    q = np.asarray(q_coords, dtype=np.float64)
    if q.ndim != 2 or len(q) == 0 or q.shape[1] not in (1, 2, 3):
        return None
    if host_budget_bytes is None:
        host_budget_bytes = lattice_host_budget_bytes()
    dim = q.shape[1]
    origin = np.zeros(dim)
    dq = np.zeros(dim)
    dims = np.zeros(dim, dtype=np.int64)
    idx = np.zeros(q.shape, dtype=np.int64)
    snap_dev = 0.0
    for ax in range(dim):
        col = q[:, ax]
        span = float(col.max() - col.min())
        if span <= 1e-12:                      # degenerate axis: one plane
            origin[ax] = float(col[0]); dq[ax] = 0.0; dims[ax] = 1
            continue
        # The min-gap SEED only needs enough points to observe adjacent
        # lattice values; the O(n log n) unique/sort on the full column
        # dominates plan cost for large intervals (Stage-1 calls this per
        # interval). The LSQ refinement below and the snap validation still
        # run over ALL points, so a pathological subsample can only cause a
        # fallback to type-3, never a wrong lattice.
        seed_col = col if len(col) <= 200_000 else col[:: len(col) // 100_000]
        u = np.unique(np.round(seed_col, 9))
        gaps = np.diff(u)
        gaps = gaps[gaps > max(1e-9, span * 1e-6)]   # ignore float-noise micro-gaps
        if gaps.size == 0:
            return None
        step = float(gaps.min())
        n_axis = int(round(span / step)) + 1
        if n_axis > (1 << 20):                 # non-lattice data snaps to absurd dims
            return None
        origin[ax] = float(col.min())
        ratio = (col - origin[ax]) / step
        ax_idx = np.round(ratio).astype(np.int64)
        dev = float(np.abs(ratio - ax_idx).max())
        if dev > snap_tol or ax_idx.min() < 0 or ax_idx.max() >= n_axis:
            return None
        # LSQ step refinement: the min-gap estimate carries the float noise of
        # a single gap, which accumulates linearly with the lattice index and
        # sets the reconstruction floor. Regressing col ~ origin + idx*step
        # over all points cancels it (same as the residual-task builder).
        num = float(np.dot(ax_idx, col - origin[ax]))
        den = float(np.dot(ax_idx, ax_idx))
        if den > 0.0:
            refined = num / den
            if refined > 0.0 and abs(refined - step) < 0.1 * step:
                step = refined
                n_axis = int(round(span / step)) + 1
                ratio = (col - origin[ax]) / step
                ax_idx = np.round(ratio).astype(np.int64)
                dev = float(np.abs(ratio - ax_idx).max())
                if dev > snap_tol or ax_idx.min() < 0 or ax_idx.max() >= n_axis:
                    return None
        dq[ax] = step; dims[ax] = n_axis
        snap_dev = max(snap_dev, dev)
        idx[:, ax] = ax_idx
    if max_snap_dev is not None and snap_dev > float(max_snap_dev):
        return None
    grid_bytes = int(np.prod(dims)) * 16 * max(1, int(n_trans))
    if grid_bytes > host_budget_bytes:
        return None
    return {
        "origin": origin, "dq": dq, "dims": tuple(int(v) for v in dims),
        "snap_dev": snap_dev, "flat_index": np.ravel_multi_index(tuple(idx.T), tuple(int(v) for v in dims)),
    }


def scatter_on_lattice(meta: dict, weights: np.ndarray) -> np.ndarray:
    """Scatter-ADD weight rows onto the dense lattice grid (duplicates sum,
    matching type-3 linearity exactly). Returns ``(n_trans, *dims)``."""
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    dims = meta["dims"]
    flat = meta["flat_index"]
    grids = np.zeros((weights.shape[0], int(np.prod(dims))), dtype=np.complex128)
    for row in range(weights.shape[0]):
        np.add.at(grids[row], flat, weights[row])
    return grids.reshape((weights.shape[0],) + tuple(dims))


def _next_fft_size(n: int) -> int:
    return max(2, int(2 * n))                  # upsampled fine-grid axis estimate


def execute_type2_on_lattice(
    meta: dict,
    grids: np.ndarray,
    real_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
    tile_consumer=None,
) -> np.ndarray | None:
    """Evaluate ``F(r) = sum_m grid[m] exp(-i r.(origin + m*dq))`` at arbitrary
    targets via type-2 NUFFT, mode-slabbed along the FIRST axis. Exact to NUFFT
    eps.

    Memory discipline (the failure modes this design closes):
      * Slabs are taken along axis 0 of each transform row, so every slab is a
        contiguous VIEW of the cached grid -- zero host memcpy per work unit.
      * Uploads (slab + coords + result) live in the CuPy POOL, which the
        pipeline caps at ~2.4 GiB/worker; slab length is sized against the
        pool's actual free capacity, not free VRAM (sizing against VRAM is what
        OOM-crashed the first hkl32 integration run).
      * TARGETS are tiled too: every target-axis buffer (coords, phase, result
        rows) is sized by the tile, never by the full call. Streaming
        subchunks carry the chunk's FULL point range, so n_tgt is
        points x rifft-grid (hkl40: 3072 x 41^3 = 2.1e8 targets = ~5 GiB of
        coords alone) -- unconditionally uploading that is what OOM-crashed
        the first hkl40 streaming run and demoted its workers to CPU.
      * The cuFINUFFT fine grid is raw cudaMalloc OUTSIDE the pool; it is sized
        against the dynamically shared VRAM tile budget.
      * Rows run as separate n_trans=1 calls (halves the fine grid; the phase
        vector is shared per slab), and any residual OOM halves the slab length
        and retries instead of failing the work unit.

    Throughput discipline (GPU):
      * One explicit cuFINUFFT plan per slab shape, reused across slabs and
        rows -- ``setpts`` (which sorts all targets) runs once per plan, not
        once per (slab x row) as the simple interface would.
      * Phase multiply and accumulation happen on the device; the result
        crosses PCIe once at the end instead of once per (slab x row). If the
    fixed device buffers do not fit the pool, the call falls back to
    per-slab host accumulation transparently.

    ``tile_consumer(t0, t1, tile_out)``: when given, the full ``(n_trans,
    n_tgt)`` result is NEVER materialized. Each completed target tile is
    handed to the callback as a host ``(n_trans, t1 - t0)`` array (owned by
    the callee only for the duration of the call) and the function returns
    ``None``. This is the streaming path for huge target sets: the caller
    folds tiles straight into its (possibly memmap-backed) accumulator, so
    host memory is O(tile), not O(targets). The wrapped type-2 coordinates
    are likewise computed per tile, so ``real_coords`` may be a read-only
    memmap and no full-size host temporary of it is ever made."""
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    tgt = np.asarray(real_coords, dtype=np.float64)
    grids = np.asarray(grids, dtype=np.complex128)
    dims = meta["dims"]
    dq = np.asarray(meta["dq"], dtype=np.float64)
    origin = np.asarray(meta["origin"], dtype=np.float64)
    dim = len(dims)
    n_trans = int(grids.shape[0])
    n_tgt = int(len(tgt))
    out = (
        None
        if tile_consumer is not None
        else np.zeros((n_trans, n_tgt), dtype=np.complex128)
    )
    if n_tgt == 0 or int(np.prod(dims)) == 0:
        return out

    use_gpu = (
        not (_CPU_ONLY or prefer_cpu) and _GPU_AVAILABLE and cp is not None
    )
    if use_gpu:
        try:
            _ensure_gpu_kernels()
        except ImportError:
            use_gpu = False
    if not use_gpu and gpu_only:
        raise RuntimeError("GPU execution forced but unavailable for lattice type-2.")

    if use_gpu:
        # Per-target pool bytes while one slab is in flight (device-accum
        # worst case): d_x (8*dim) + d_tgt (8*dim) + d_out (16*n_trans) +
        # d_phase (16) + uncommitted row_results (16*n_trans). setpts sort
        # scratch is raw cudaMalloc outside the pool and is covered by the
        # 0.5 pool-fraction headroom.
        bytes_per_tgt = 16 * dim + 32 * n_trans + 16
        try:
            pool = cp.get_default_memory_pool()
            limit = int(pool.get_limit() or 0)
            pool_free = (
                max(0, limit - int(pool.used_bytes())) if limit > 0 else _free_mem_bytes()
            )
        except Exception:
            pool_free = _free_mem_bytes()
        # Every (tile x slab) pair costs one cuFINUFFT execute, and each
        # execute re-runs the slab's FINE-GRID FFT (the API offers no
        # FFT-once-interpolate-many). Minimizing executes means maximizing
        # the tile_len x slab_len PRODUCT under the shared pool budget --
        # an equal split, with the slab side clamped at the whole grid.
        # (hkl40 measured: 10M-target tiles starved slabs to the 64 MiB
        # floor -> 22x40 executes, 805 s/batch; a 75/25 slab-first split
        # still paid 85x3, 642 s/batch. The equal split with an adequate
        # pool gives ~1 slab x ~a dozen tiles.) Small grids leave the tile
        # budget ~whole-pool, so small configs never tile.
        budget = int(pool_free * 0.5)
        grid_row_bytes = 16 * int(np.prod(dims[1:])) if dim > 1 else 16
        desired_slab_bytes = min(int(dims[0]) * grid_row_bytes, budget // 2)
        tile_len = max(
            262_144, (budget - desired_slab_bytes) // bytes_per_tgt
        )
        tile_len = min(tile_len, n_tgt)
    else:
        # CPU path: with a tile consumer the caller is streaming tiles into an
        # accumulator precisely to bound host memory, so honor that here too
        # instead of materializing every target-axis temporary at full size.
        tile_len = (
            min(n_tgt, _env_int("MOSAIC_NUFFT_CPU_TILE_TARGETS", 8_000_000))
            if tile_consumer is not None
            else n_tgt
        )
    tile_bounds = [
        (t0, min(t0 + tile_len, n_tgt)) for t0 in range(0, n_tgt, tile_len)
    ]
    if len(tile_bounds) > 1:
        logger.info(
            "lattice type-2: %d targets exceed pool tile budget; running %d "
            "target tiles of <=%d (plans + fine grids reused across tiles)",
            n_tgt,
            len(tile_bounds),
            tile_len,
        )

    # Wrapped type-2 coordinates are computed PER TILE inside the loop below
    # (degenerate axes with dq=0 map to x=0, a single mode). Materializing the
    # full (n_tgt, dim) wrap here doubled the target-side host footprint --
    # ~5 GiB per hkl40 chunk -- before a single tile ran.
    centers = [d // 2 for d in dims]
    row_bytes_per_len = 16 * int(np.prod(dims[1:])) if dim > 1 else 16
    fine_other = 1
    for a in range(1, dim):
        fine_other *= _next_fft_size(dims[a])

    def _pool_free_bytes() -> int:
        try:
            pool = cp.get_default_memory_pool()
            limit = int(pool.get_limit() or 0)
            if limit > 0:
                return max(0, limit - int(pool.used_bytes()))
        except Exception:
            pass
        return _free_mem_bytes()

    def _slab_len_now(device_accum: bool, tile_n: int) -> int:
        # pool side: slab upload + coords + per-call result must fit the pool
        # (target-side terms scale with the TILE, not the whole call)
        fixed = 24 * tile_n + 16 * tile_n * 2
        if device_accum:
            # d_tgt + d_phase + d_out + per-slab row temporaries
            fixed += 24 * tile_n + 16 * tile_n + 16 * tile_n * n_trans * 2
        pool_room = max(64 << 20, int(_pool_free_bytes() * 0.5) - fixed)
        len_pool = pool_room // max(1, row_bytes_per_len)
        # fine-grid side (raw cudaMalloc): sized from the shared VRAM budget
        fine_budget = _current_tile_budget() if use_gpu else int(
            os.getenv("MOSAIC_RESIDUAL_LATTICE_CPU_FINE_BUDGET", str(8 << 30))
        )
        len_fine = int(fine_budget // max(1.0, 16 * 2 * fine_other * 1.35))
        len_fine = max(1, min(int(dims[0]), len_fine))
        if use_gpu:
            # Shrink until the ACTUAL reservation footprint (full+tail slab
            # shapes, next-fft-doubled axes, cuFFT workspace, sort scratch,
            # x1.25) fits the budget. The divisor above models ~1.35x per unit
            # length but the real footprint is ~4-5x: on the first hkl32 batch
            # of a fresh process the slab it produced reserved 30.8 GiB against
            # a ~13 GiB pool, the empty-ledger progress guarantee admitted it,
            # and materializing the plans drove free VRAM to 0 (watchdog-
            # confirmed). Sizing and reserving from the SAME footprint model
            # keeps the ledger honest regardless of the model's absolute error.
            while len_fine > 1 and _plan_footprint_bytes(len_fine, tile_n) > fine_budget:
                len_fine //= 2
        return max(1, min(int(dims[0]), int(len_pool), len_fine))

    def _is_cufinufft_alloc_failure(exc: Exception) -> bool:
        # cuFINUFFT raises bare RuntimeErrors when its internal raw cudaMalloc
        # fails: 'Error creating plan.' (fine grid), 'Error setting
        # non-uniform points.' (bin-sort scratch, sized by n_tgt), 'Error
        # executing plan.'. All are memory-shaped under VRAM pressure; treat
        # them like a pool OOM so the slab backoff can recover instead of
        # failing the work unit.
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return "plan" in message or "non-uniform points" in message

    def _plan_footprint_bytes(s_len: int, tile_n: int) -> int:
        # cuFINUFFT raw-cudaMalloc footprint of this call's live plans: the
        # upsampled fine grid (fw, complex128) plus kernel/sort work arrays
        # (idxnupts + sortidx + bins ~ 16 B/point), for BOTH slab shapes the
        # partition produces (full + tail). Reserved via _reserve_tile so
        # concurrent transforms QUEUE for VRAM instead of racing cuFINUFFT's
        # makeplan into a cudaErrorMemoryAllocation -- whose C++ cleanup
        # throws in a destructor and SIGABRTs the whole process (measured on
        # the base CaTiO3 run with 4 in-flight transforms).
        shapes = {min(max(1, int(s_len)), int(dims[0]))}
        tail = int(dims[0]) % max(1, int(s_len))
        if tail:
            shapes.add(tail)
        fine_total = sum(
            _next_fft_size(m0) * fine_other * 16 * 2 for m0 in shapes
        )
        sort_bytes = 16 * int(tile_n) * len(shapes)
        return int((fine_total + sort_bytes) * 1.25)

    if use_gpu:
        _transform_enter()
    plans: dict[tuple[int, ...], object] = {}
    plans_tile: dict[tuple[int, ...], int] = {}   # shape -> tile of last setpts
    reserved_bytes = 0

    def _ensure_plan_reservation(s_len: int) -> None:
        # Single-point reservation per call: drop the old claim (and the
        # plans it covered) BEFORE waiting on the new one, so two concurrent
        # transforms can never hold-and-wait on each other.
        nonlocal reserved_bytes
        needed = _plan_footprint_bytes(s_len, tile_len)
        if needed == reserved_bytes:
            return
        if reserved_bytes:
            plans.clear()
            plans_tile.clear()
            _release_tile(reserved_bytes)
            reserved_bytes = 0
        _reserve_tile(needed)
        reserved_bytes = needed

    try:
        if use_gpu:
            plan_cls = _lazy_cufinufft("Plan")
        else:
            import finufft
            fn = {1: finufft.nufft1d2, 2: finufft.nufft2d2, 3: finufft.nufft3d2}[dim]
        oom_errors = (
            (cp.cuda.memory.OutOfMemoryError, RuntimeError) if use_gpu else tuple()
        )
        for tile_idx, (t0, t1) in enumerate(tile_bounds):
            n_t = t1 - t0
            tgt_t = np.asarray(tgt[t0:t1], dtype=np.float64)
            x_t = ((tgt_t * dq[None, :] + np.pi) % (2.0 * np.pi)) - np.pi
            if use_gpu:
                try:
                    d_x = [
                        cp.asarray(np.ascontiguousarray(x_t[:, a]))
                        for a in range(dim)
                    ]
                except cp.cuda.memory.OutOfMemoryError:
                    # pool shrank since tile sizing (concurrent transforms):
                    # drop cached plans and retry once before giving up
                    plans.clear()
                    plans_tile.clear()
                    free_gpu_memory()
                    d_x = [
                        cp.asarray(np.ascontiguousarray(x_t[:, a]))
                        for a in range(dim)
                    ]

                def _plan_for(n_modes: tuple[int, ...]):
                    # Plans (and their raw-cudaMalloc fine grids) are keyed by
                    # slab SHAPE and live across tiles -- only setpts (the
                    # target sort) reruns per tile. Rebuilding plans per tile
                    # is what made hkl40 transforms ~10x slower than the
                    # arithmetic said they should be.
                    plan = plans.get(n_modes)
                    if plan is None:
                        # The CuPy pool caches freed blocks up to its limit;
                        # that VRAM looks free to us but raw cudaMalloc (which
                        # cuFINUFFT uses) cannot touch it. Hand blocks back to
                        # the driver before a large plan build when raw VRAM
                        # is short.
                        fine_bytes = 16 * 2 * _next_fft_size(int(n_modes[0])) * fine_other
                        if _free_mem_bytes() < fine_bytes:
                            _free_cupy_pool_blocks()
                        plan = plan_cls(
                            2, n_modes, n_trans=1, eps=eps, isign=-1, dtype="complex128"
                        )
                        plans[n_modes] = plan
                        plans_tile[n_modes] = -1
                        _warn_if_below_headroom(f"type2-plan{n_modes}")
                    if plans_tile.get(n_modes) != tile_idx:
                        plan.setpts(*d_x)
                        plans_tile[n_modes] = tile_idx
                        _warn_if_below_headroom(f"type2-setpts{n_modes}")
                    return plan
            else:
                d_x = [np.ascontiguousarray(x_t[:, a]) for a in range(dim)]
            d_tgt = d_out = None
            device_accum = False
            if use_gpu:
                try:
                    d_tgt = cp.asarray(tgt_t)
                    d_out = cp.zeros((n_trans, n_t), dtype=cp.complex128)
                    device_accum = True
                except cp.cuda.memory.OutOfMemoryError:
                    d_tgt = d_out = None
                    free_gpu_memory()
            # Host-side accumulation target for this tile: the caller's full
            # array when no consumer, a tile-sized scratch row otherwise.
            tile_buf = None
            if tile_consumer is not None and not device_accum:
                tile_buf = np.zeros((n_trans, n_t), dtype=np.complex128)
            slab_len = _slab_len_now(device_accum, n_t)
            if use_gpu:
                _ensure_plan_reservation(slab_len)
            a0 = 0
            while a0 < int(dims[0]):
                a1 = min(a0 + slab_len, int(dims[0]))
                c = list(centers)
                c[0] = a0 + (a1 - a0) // 2
                q_c = origin + np.asarray(c, dtype=np.float64) * dq
                plan = d_sub = d_o = d_phase = row_results = None
                try:
                    if use_gpu:
                        plan = _plan_for(tuple(int(v) for v in (a1 - a0,) + tuple(dims[1:])))
                        row_results = []
                        for row in range(n_trans):
                            sub = grids[row, a0:a1]    # contiguous view, no copy
                            d_sub = cp.asarray(sub)
                            row_results.append(plan.execute(d_sub).reshape(n_t))
                            d_sub = None
                        # commit only after the whole slab succeeded, so an OOM
                        # retry (smaller slab) never double-counts a partial slab
                        if device_accum:
                            d_phase = cp.exp(-1j * (d_tgt @ cp.asarray(q_c)))
                            # in-place ops only from here on -- nothing below
                            # allocates, so an OOM cannot land BETWEEN row
                            # commits (which would double-count this slab's
                            # already-committed rows on the retry)
                            for d_o in row_results:
                                d_o *= d_phase
                            for row, d_o in enumerate(row_results):
                                d_out[row] += d_o
                            d_phase = None
                        else:
                            phase = np.exp(-1j * (tgt_t @ q_c))
                            host_rows = [cp.asnumpy(d_o) for d_o in row_results]
                            for row, o in enumerate(host_rows):
                                if tile_buf is not None:
                                    tile_buf[row] += o * phase
                                else:
                                    out[row, t0:t1] += o * phase
                        row_results = None
                    else:
                        phase = np.exp(-1j * (tgt_t @ q_c))
                        row_results = []
                        for row in range(n_trans):
                            sub = grids[row, a0:a1]
                            o = fn(*d_x, sub, isign=-1, eps=eps)
                            row_results.append(
                                np.asarray(o, dtype=np.complex128).reshape(n_t)
                            )
                        for row, o in enumerate(row_results):
                            if tile_buf is not None:
                                tile_buf[row] += o * phase
                            else:
                                out[row, t0:t1] += o * phase
                except oom_errors as exc:
                    if use_gpu and isinstance(exc, RuntimeError) and not (
                        isinstance(exc, cp.cuda.memory.OutOfMemoryError)
                        or _is_cufinufft_alloc_failure(exc)
                    ):
                        raise
                    if slab_len <= 1:
                        raise
                    slab_len = max(1, slab_len // 2)
                    # drop every reference to the failed attempt BEFORE freeing:
                    # these locals would otherwise pin the old fine grid and row
                    # buffers through the retry's (smaller) allocations
                    plan = d_sub = d_o = d_phase = row_results = None
                    plans.clear()
                    plans_tile.clear()
                    free_gpu_memory()
                    if use_gpu:
                        _ensure_plan_reservation(slab_len)
                    logger.debug(
                        "lattice type-2 slab OOM; halving slab_len to %d (%s)",
                        slab_len,
                        exc,
                    )
                    continue
                a0 = a1
            if use_gpu and device_accum:
                tile_host = cp.asnumpy(d_out)
                if tile_consumer is not None:
                    tile_consumer(t0, t1, tile_host)
                else:
                    out[:, t0:t1] = tile_host
                tile_host = None
            elif tile_buf is not None:
                tile_consumer(t0, t1, tile_buf)
            tile_buf = None
            d_out = d_tgt = None
            d_x = None
        if use_gpu:
            plans.clear()
            plans_tile.clear()
            _free_cupy_pool_blocks()
    finally:
        plans.clear()
        plans_tile.clear()
        if reserved_bytes:
            _release_tile(reserved_bytes)
            reserved_bytes = 0
        if use_gpu:
            _transform_exit()
    return out


def _lazy_cufinufft(name):
    import cufinufft
    return getattr(cufinufft, name)


def cufinufft_nufft1d2(*args, **kwargs):
    return _lazy_cufinufft("nufft1d2")(*args, **kwargs)


def cufinufft_nufft2d2(*args, **kwargs):
    return _lazy_cufinufft("nufft2d2")(*args, **kwargs)


def cufinufft_nufft3d2(*args, **kwargs):
    return _lazy_cufinufft("nufft3d2")(*args, **kwargs)


def execute_lattice_type2_batch(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
):
    """Convenience: plan + scatter + type-2. Returns ``None`` when the q-points
    are not lattice-eligible (caller falls back to the type-3 path)."""
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    meta = plan_lattice(q_coords, n_trans=int(weights.shape[0]))
    if meta is None:
        return None
    grids = scatter_on_lattice(meta, weights)
    return execute_type2_on_lattice(
        meta, grids, real_coords, eps=eps, prefer_cpu=prefer_cpu, gpu_only=gpu_only
    )


###############################################################################
#  Lattice type-1 (forward): nonuniform sources -> uniform lattice box        #
#                                                                             #
#  The Stage-1 scattering transform evaluates A(q) = sum_k w_k exp(+i q.r_k)  #
#  at the q-points of one interval. Those q-points sit on the run's global    #
#  uniform lattice, so the transform is a type-1 NUFFT onto the interval's    #
#  mode box. Crucially the wrapped source coordinates x_k = wrap(dq * r_k)    #
#  depend only on the LATTICE PITCH, not on the interval: one plan + setpts   #
#  per (source set, box dims, dq) serves every interval of the run, and the   #
#  interval's box origin enters as a per-call weight phase. The cache below   #
#  holds those plans; it is bounded, per-process, and must be cleared at the  #
#  end of the scattering stage (plan scratch is raw cudaMalloc outside the    #
#  CuPy pool).                                                                #
###############################################################################
_TYPE1_PLAN_CACHE: "dict[tuple, _Type1PlanEntry]" = {}
_TYPE1_PLAN_CACHE_ORDER: list = []
_TYPE1_PLAN_CACHE_LOCK = threading.Lock()


class _Type1PlanEntry:
    """Leased cache entry.

    Concurrent worker threads can fetch an entry while another thread evicts
    or clears it (LRU overflow, GPU-OOM cache flush). Destroying the plan in
    that window would hand a freed cuFINUFFT handle to ``execute`` -- a
    native use-after-free. Leases make destruction safe: eviction only marks
    the entry doomed while leases are outstanding, and the LAST release
    destroys the plan.
    """

    __slots__ = ("plan", "d_x", "d_r", "lock", "leases", "doomed", "nbytes")

    def __init__(self, plan, d_x, d_r, nbytes=0):
        self.plan = plan
        self.d_x = d_x
        self.d_r = d_r
        self.lock = threading.Lock()
        self.leases = 0
        self.doomed = False
        self.nbytes = int(nbytes)


def _type1_entry_destroy(entry: "_Type1PlanEntry") -> None:
    with entry.lock:
        _destroy_plan_quietly(entry.plan)
        entry.plan = None


def _type1_entry_release(entry: "_Type1PlanEntry") -> None:
    destroy = False
    with _TYPE1_PLAN_CACHE_LOCK:
        entry.leases -= 1
        destroy = entry.doomed and entry.leases <= 0
    if destroy:
        _type1_entry_destroy(entry)


def _type1_plan_cache_max() -> int:
    # A run's working set is (source sets) x (distinct interval box shapes):
    # interior/edge/zero-plane boxes easily produce 40-80 keys, and a cap
    # below the working set thrashes plan creation per transform. The byte
    # budget below is the real bound; the count is a backstop.
    raw = os.getenv("MOSAIC_SCATTERING_TYPE1_PLAN_CACHE_MAX", "256")
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


def _type1_plan_cache_max_bytes() -> int:
    raw = os.getenv("MOSAIC_SCATTERING_TYPE1_CACHE_MAX_BYTES")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    total = _total_mem_bytes()
    if total > 0:
        # 1/8 of the card, capped at the historical 4 GiB: on big cards this
        # fits the ~30%-of-VRAM plan-scratch model unchanged, on small cards
        # it stops the cache from squatting on the absolute headroom.
        return int(min(4 << 30, max(total // 8, 512 << 20)))
    return 4 << 30


def _type1_cache_total_bytes_locked() -> int:
    return sum(entry.nbytes for entry in _TYPE1_PLAN_CACHE.values())


def clear_lattice_type1_plan_cache() -> None:
    doomed_now: list[_Type1PlanEntry] = []
    with _TYPE1_PLAN_CACHE_LOCK:
        for entry in _TYPE1_PLAN_CACHE.values():
            if entry.leases > 0:
                entry.doomed = True
            else:
                doomed_now.append(entry)
        _TYPE1_PLAN_CACHE.clear()
        _TYPE1_PLAN_CACHE_ORDER.clear()
    for entry in doomed_now:
        _type1_entry_destroy(entry)
    _free_cupy_pool_blocks()


def _type1_cache_acquire_or_build(key: tuple, builder):
    """Return ``(entry, cached)`` with one lease taken on ``entry``.

    The caller MUST pair this with ``_type1_entry_release`` (cached entries)
    or destroy the plan itself (``cached=False``, cache disabled). Leases are
    taken under the cache lock, so an entry handed out here can never be
    destroyed underneath its user by eviction or a concurrent clear.
    """
    max_entries = _type1_plan_cache_max()
    if max_entries <= 0:
        plan, d_x, d_r, nbytes = builder()
        return _Type1PlanEntry(plan, d_x, d_r, nbytes), False
    with _TYPE1_PLAN_CACHE_LOCK:
        entry = _TYPE1_PLAN_CACHE.get(key)
        if entry is not None:
            try:
                _TYPE1_PLAN_CACHE_ORDER.remove(key)
            except ValueError:
                pass
            _TYPE1_PLAN_CACHE_ORDER.append(key)
            entry.leases += 1
            return entry, True
    plan, d_x, d_r, nbytes = builder()
    entry = _Type1PlanEntry(plan, d_x, d_r, nbytes)
    victims: list[_Type1PlanEntry] = []
    max_bytes = _type1_plan_cache_max_bytes()
    with _TYPE1_PLAN_CACHE_LOCK:
        existing = _TYPE1_PLAN_CACHE.get(key)
        if existing is not None:
            victims.append(entry)  # lost the build race; ours is surplus
            existing.leases += 1
            entry = existing
        else:
            while _TYPE1_PLAN_CACHE_ORDER and (
                len(_TYPE1_PLAN_CACHE_ORDER) >= max(1, max_entries)
                or _type1_cache_total_bytes_locked() + entry.nbytes > max_bytes
            ):
                oldest = _TYPE1_PLAN_CACHE_ORDER.pop(0)
                victim = _TYPE1_PLAN_CACHE.pop(oldest, None)
                if victim is None:
                    continue
                if victim.leases > 0:
                    victim.doomed = True
                else:
                    victims.append(victim)
            _TYPE1_PLAN_CACHE[key] = entry
            _TYPE1_PLAN_CACHE_ORDER.append(key)
            entry.leases += 1
    for victim in victims:
        _type1_entry_destroy(victim)
    return entry, True


def _cufinufft_error_is_alloc(exc: Exception) -> bool:
    """cuFINUFFT raises bare RuntimeErrors when its internal raw cudaMalloc
    fails ('Error creating plan.', 'Error setting non-uniform points.',
    'Error executing plan.'); under VRAM pressure they are memory-shaped."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "plan" in message or "non-uniform points" in message


def _type1_gpu_method() -> int:
    raw = os.getenv("MOSAIC_SCATTERING_TYPE1_GPU_METHOD", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _type1_fine_grid_bytes(dims) -> int:
    fine = 16 * 2  # complex128, ~2x residency for FFT work
    for d in dims:
        fine *= _next_fft_size(int(d))
    return int(fine)


def execute_type1_on_lattice(
    meta: dict,
    real_coords: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = 1e-12,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray | None:
    """Evaluate ``A(q) = sum_k w_k exp(+i q.r_k)`` at ``meta``'s lattice
    q-points via type-1 NUFFT. Exact to NUFFT eps and identical in convention
    to ``execute_cunufft`` (forward, isign=+1).

    ``meta`` comes from ``plan_lattice(q_grid)``; the result is gathered at
    ``meta['flat_index']`` so it aligns with the caller's stored q order.
    Returns ``None`` when the transform cannot run here (fine grid beyond the
    memory budget, or GPU OOM persisting after a cache flush) -- callers fall
    back to the type-3 path."""
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    r = np.asarray(real_coords, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)
    dims = tuple(int(v) for v in meta["dims"])
    dq = np.asarray(meta["dq"], dtype=np.float64)
    origin = np.asarray(meta["origin"], dtype=np.float64)
    dim = len(dims)
    flat_index = np.asarray(meta["flat_index"], dtype=np.int64)
    if r.shape[0] != w.shape[0]:
        raise ValueError("Lattice type-1 requires one weight per source point.")
    if flat_index.size == 0:
        return np.zeros(0, dtype=np.complex128)
    if r.shape[0] == 0:
        return np.zeros(flat_index.size, dtype=np.complex128)

    use_gpu = (
        not (_CPU_ONLY or prefer_cpu) and _GPU_AVAILABLE and cp is not None
    )
    if use_gpu:
        try:
            _ensure_gpu_kernels()
        except ImportError:
            use_gpu = False
    if not use_gpu and gpu_only:
        raise RuntimeError("GPU execution forced but unavailable for lattice type-1.")

    # GPU budget comes from the SHARED reservation pool (an equal live share
    # among in-flight transforms), not a private fraction of free VRAM: N
    # threads each sizing themselves from the same free reading is exactly how
    # the total footprint used to grow with thread count.
    fine_budget = (
        _current_tile_budget()
        if use_gpu
        else int(os.getenv("MOSAIC_RESIDUAL_LATTICE_CPU_FINE_BUDGET", str(8 << 30)))
    )
    if _type1_fine_grid_bytes(dims) > fine_budget:
        return None

    q_c = origin + (np.asarray(dims, dtype=np.float64) // 2) * dq
    x = ((r * dq[None, :] + np.pi) % (2.0 * np.pi)) - np.pi

    if not use_gpu:
        import finufft
        fn = {1: finufft.nufft1d1, 2: finufft.nufft2d1, 3: finufft.nufft3d1}[dim]
        c = w * np.exp(1j * (r @ q_c))
        f = fn(
            *[np.ascontiguousarray(x[:, a]) for a in range(dim)],
            c,
            dims,
            isign=+1,
            eps=eps,
        )
        return np.asarray(f, dtype=np.complex128).reshape(-1)[flat_index]

    key = (
        hashlib.sha256(np.ascontiguousarray(r).view(np.uint8).tobytes()).digest()[:16],
        dims,
        dq.tobytes(),
        float(eps),
    )

    def _builder():
        # Reserve the plan's raw-cudaMalloc footprint from the shared pool for
        # the duration of the build, so concurrent type-1 builds on other
        # worker threads queue for VRAM instead of racing cuFINUFFT's
        # makeplan. Once built, the allocation is visible in live free VRAM,
        # so the claim is dropped (holding it would double-count).
        nbytes = _type1_fine_grid_bytes(dims) + r.nbytes * 2
        _reserve_tile(nbytes)
        try:
            d_x = [cp.asarray(np.ascontiguousarray(x[:, a])) for a in range(dim)]
            d_r = cp.asarray(r)
            plan = _lazy_cufinufft("Plan")(
                1, dims, n_trans=1, eps=eps, isign=+1, dtype="complex128",
                # Global-memory spreading: the default shared-memory subproblem
                # method collapses on this shape (moderate point counts, wide
                # eps=1e-12 kernels, small mode boxes) -- measured 70 ms vs 3.9 ms
                # per execute at hkl32 interval scale, identical results to 2e-15.
                gpu_method=_type1_gpu_method(),
            )
            plan.setpts(*d_x)
            _warn_if_below_headroom(f"type1-plan-build{dims}")
            return plan, d_x, d_r, nbytes
        finally:
            _release_tile(nbytes)

    _transform_enter()
    try:
        last_exc = None
        for attempt in (0, 1):
            entry = None
            cached = False
            try:
                entry, cached = _type1_cache_acquire_or_build(key, _builder)
                # only plan.execute needs the entry lock (cuFINUFFT plans are
                # not thread-safe); the weight phase and the gather are on
                # private arrays, and keeping them outside lets other worker
                # threads overlap their host/device prep instead of
                # serializing whole intervals on the shared per-source plan
                d_w = cp.asarray(w) * cp.exp(1j * (entry.d_r @ cp.asarray(q_c)))
                with entry.lock:
                    d_f = entry.plan.execute(d_w)
                result = cp.asnumpy(d_f.reshape(-1)[cp.asarray(flat_index)])
                return result
            except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and not (
                    isinstance(exc, cp.cuda.memory.OutOfMemoryError)
                    or _cufinufft_error_is_alloc(exc)
                ):
                    raise
                last_exc = exc
                clear_lattice_type1_plan_cache()
                free_gpu_memory()
            finally:
                if entry is not None:
                    if cached:
                        _type1_entry_release(entry)
                    else:
                        _type1_entry_destroy(entry)
        logger.warning(
            "lattice type-1 GPU memory failure persists; falling back to "
            "type-3 for this transform (%s)",
            last_exc,
        )
        return None
    finally:
        _transform_exit()


def _local_window_inverse_cpu(q, weights, offsets, centers, eps):
    n_rows, n_q = weights.shape
    n_atoms = int(len(centers))
    n_win = int(len(offsets))
    out = np.empty((n_rows, n_atoms * n_win), dtype=np.complex128)
    per_atom = max(1, n_rows * n_q * 16)
    atom_tile = max(1, min(n_atoms, (256 << 20) // per_atom))
    for a0 in range(0, n_atoms, atom_tile):
        a1 = min(a0 + atom_tile, n_atoms)
        phase = np.exp(-1j * (centers[a0:a1] @ q.T))                 # (na, n_q)
        w_mat = (weights[:, None, :] * phase[None, :, :]).reshape(n_rows * (a1 - a0), n_q)
        field = execute_inverse_cunufft_super_batch(q, w_mat, offsets, eps=eps, prefer_cpu=True)
        out[:, a0 * n_win : a1 * n_win] = np.asarray(field).reshape(
            n_rows, (a1 - a0) * n_win
        )
    return out


def execute_local_window_inverse(
    q_coords: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    centers: np.ndarray,
    *,
    eps: float = 1e-12,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse type-3 in local window coordinates, batched over atoms.

    ``field(center_a + delta_j) = sum_q [w(q) exp(-i center_a.q)] exp(-i delta_j.q)``.

    Because every atom shares the same q-sources and window-target grid, the
    cuFINUFFT plan and ``setpts`` are built **once** and only the per-atom weight
    matrix changes between executes. The centre phase and weight matrix are formed
    on the GPU, so the O(n_atoms x n_q) intermediate never crosses PCIe -- only the
    result (n_rows x n_atoms x n_win) is copied back. Returns ``(n_rows,
    n_atoms*n_win)`` in atom-major order, identical to the global transform to
    NUFFT eps. The fine grid depends on the *window* extent (~1 A), not the
    supercell, which is what makes large-cell 3D tractable on the GPU."""
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    q = np.asarray(q_coords, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.complex128)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    n_rows, n_q = weights.shape
    n_atoms = int(len(centers))
    n_win = int(len(offsets))
    dim = int(q.shape[1])
    out = np.empty((n_rows, n_atoms * n_win), dtype=np.complex128)
    if n_atoms == 0 or n_win == 0:
        return out

    use_gpu = not (_CPU_ONLY or prefer_cpu) and _GPU_AVAILABLE
    if use_gpu:
        try:
            _ensure_gpu_kernels()
        except ImportError:
            use_gpu = False
    if not use_gpu:
        if gpu_only:
            raise RuntimeError("GPU execution forced but unavailable for local-window inverse.")
        return _local_window_inverse_cpu(q, weights, offsets, centers, eps)

    import cufinufft  # type: ignore

    d_q = _as_device(q)
    d_off = _as_device(offsets)
    d_centers = _as_device(centers)
    d_w = _as_device(weights)
    q_cols = [_contig(d_q[:, i]) for i in range(dim)]
    off_cols = [_contig(d_off[:, i]) for i in range(dim)]

    # Atom-tile so the on-device weight matrix (n_rows*tile, n_q) + phase fit VRAM.
    def _build(n_trans):
        plan = cufinufft.Plan(
            3, dim, n_trans=int(n_trans), eps=eps, isign=-1, dtype="complex128",
            **_build_gpu_launch_kwargs(gpu_maxsubprobsize=_subprob_order(dim, int(n_trans))[0]),
        )
        _set_type3_points(plan, dim=dim, source_cols=q_cols, target_cols=off_cols)
        return plan

    # Size the atom tile so the on-device working set -- the (n_atoms x n_q) centre
    # phase, its float matmul temporary, and the (n_rows*na x n_q) weight matrix --
    # fits the CuPy pool (a few GiB, NOT total VRAM; cuFINUFFT plan scratch lives
    # outside the pool). Retry with a smaller tile on OOM.
    workset_per_atom = max(1, n_q * (40 + 16 * n_rows))
    budget = int(os.getenv("MOSAIC_RESIDUAL_LOCAL_WORKSET_BYTES", str(384 << 20)))
    atom_tile = max(1, min(n_atoms, budget // workset_per_atom))
    while True:
        full_plan = _build(n_rows * atom_tile) if atom_tile <= n_atoms else None
        try:
            for a0 in range(0, n_atoms, atom_tile):
                a1 = min(a0 + atom_tile, n_atoms)
                na = a1 - a0
                d_phase = cp.exp(-1j * (d_centers[a0:a1] @ d_q.T))       # (na, n_q)
                d_w_mat = _contig(
                    (d_w[:, None, :] * d_phase[None, :, :]).reshape(n_rows * na, n_q)
                )
                if na == atom_tile and full_plan is not None:
                    d_field = full_plan.execute(d_w_mat)
                else:
                    tail_plan = _build(n_rows * na)
                    try:
                        d_field = tail_plan.execute(d_w_mat)
                    finally:
                        _destroy_plan_quietly(tail_plan)
                field = cp.asnumpy(cp.ascontiguousarray(d_field)).reshape(n_rows, na, n_win)
                out[:, a0 * n_win : a1 * n_win] = field.reshape(n_rows, na * n_win)
                del d_phase, d_w_mat, d_field
            break
        except cp.cuda.memory.OutOfMemoryError:
            if atom_tile <= 1:
                raise
            atom_tile = max(1, atom_tile // 2)
            free_gpu_memory()
        finally:
            if full_plan is not None:
                _destroy_plan_quietly(full_plan)
            _free_cupy_pool_blocks()
    return out


def execute_inverse_cunufft_batch_materialize_once(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32 * 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """
    Inverse type-3 helper for task-local GPU accumulation with a single final
    host materialization when the GPU path succeeds.
    """
    return _execute_inverse_cunufft_batch_device(
        q_coords=q_coords,
        weights=weights,
        real_coords=real_coords,
        eps=eps,
        mem_frac=mem_frac,
        min_chunk=min_chunk,
        max_chunk=max_chunk,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
    )


def _execute_inverse_cunufft_batch_device(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
    mem_frac: Optional[float] = None,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32 * 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
):
    if real_coords is None:
        raise ValueError("real_coords must be supplied for inverse transform")
    weights_arr = np.asarray(weights, dtype=np.complex128)
    if weights_arr.ndim == 1:
        weights_arr = weights_arr[np.newaxis, :]
    if weights_arr.ndim != 2:
        raise ValueError("weights must be 1-D or 2-D with shape (n_trans, n_sources)")
    if weights_arr.shape[1] != len(q_coords):
        raise ValueError("weights shape must match q_coords on axis 1")
    return _execute_inverse_cunufft_batch(
        q_coords=q_coords,
        weights_arr=weights_arr,
        real_coords=real_coords,
        eps=eps,
        mem_frac=mem_frac,
        min_chunk=min_chunk,
        max_chunk=max_chunk,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
        device_out=True,
    )


def _execute_inverse_cunufft_batch(
    *,
    q_coords: np.ndarray,
    weights_arr: np.ndarray,
    real_coords: np.ndarray,
    eps: float,
    mem_frac: Optional[float],
    min_chunk: int,
    max_chunk: Optional[int],
    prefer_cpu: bool,
    gpu_only: bool,
    device_out: bool,
) -> np.ndarray:
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    if weights_arr.shape[1] != len(q_coords):
        raise ValueError("weights shape must match q_coords on axis 1")
    experimental_overlap = _experimental_overlap_enabled()
    if experimental_overlap:
        logger.debug(
            "Experimental overlap requested for inverse batch, but the current path remains serialized with timing diagnostics only."
        )

    if gpu_only and (_CPU_ONLY or prefer_cpu):
        raise RuntimeError("GPU execution forced but CPU execution was requested.")
    if gpu_only and not _GPU_AVAILABLE:
        raise RuntimeError("GPU execution forced but no CUDA/cuFINUFFT backend is available.")

    if _CPU_ONLY or prefer_cpu or not _GPU_AVAILABLE:
        host = np.stack(
            [
                _cpu_fallback(
                    real_coords,
                    weights_arr[index],
                    q_coords,
                    eps,
                    True,
                )
                for index in range(weights_arr.shape[0])
            ],
            axis=0,
        )
        return host
    try:
        _ensure_gpu_kernels()
    except ImportError:
        if gpu_only:
            raise RuntimeError("GPU execution forced but cufinufft is unavailable.")
        host = np.stack(
            [
                _cpu_fallback(
                    real_coords,
                    weights_arr[index],
                    q_coords,
                    eps,
                    True,
                )
                for index in range(weights_arr.shape[0])
            ],
            axis=0,
        )
        return host

    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1-, 2-, and 3-D inputs supported")

    resident_coords = np.asarray(q_coords, dtype=np.float64)
    target_coords = np.asarray(real_coords, dtype=np.float64)
    n_trans = int(weights_arr.shape[0])

    telemetry = _begin_telemetry(
        mode="inverse-batch",
        n_sources=int(len(resident_coords)),
        n_targets=int(len(target_coords)),
        n_trans=n_trans,
        resident_bytes=int(resident_bytes) if 'resident_bytes' in locals() else None,
        effective_mem_frac=None,
        mem_policy_source=None,
        experimental_overlap=experimental_overlap,
    )
    out_device = None
    if device_out:
        out_device = _as_device(
            np.zeros((n_trans, len(target_coords)), dtype=np.complex128),
            allow_fail=True,
        )
        if out_device is None:
            device_out = False
    out_host = None if device_out else np.zeros((n_trans, len(target_coords)), dtype=np.complex128)

    d_resident = _as_device(resident_coords, allow_fail=True)
    if d_resident is None:
        host = np.stack(
            [
                _cpu_fallback(
                    real_coords,
                    weights_arr[index],
                    q_coords,
                    eps,
                    True,
                )
                for index in range(n_trans)
            ],
            axis=0,
        )
        return host
    d_weights = _as_device(weights_arr, allow_fail=True)
    if d_weights is None:
        free_gpu_memory()
        host = np.stack(
            [
                _cpu_fallback(
                    real_coords,
                    weights_arr[index],
                    q_coords,
                    eps,
                    True,
                )
                for index in range(n_trans)
            ],
            axis=0,
        )
        return host

    resident_cols = [_contig(d_resident[:, i]) for i in range(dim)]
    d_weights = _contig(d_weights)
    resident_bytes = _resident_bytes(
        resident_coords,
        weights_arr,
        n_trans=n_trans,
    )
    initial_free = _free_mem_bytes()
    reserve_bytes, mem_frac, mem_policy_source = _resolve_budget_policy(
        mem_frac=mem_frac,
        free_bytes=initial_free,
        resident_bytes=resident_bytes,
    )
    incremental_launch_baseline_bytes = 0
    logger.debug(
        "type3 inverse-batch memory policy | source=%s free_vram=%d resident_bytes=%d reserve_bytes=%d incremental_launch_baseline_bytes=%d effective_mem_frac=%.3f",
        mem_policy_source,
        initial_free,
        resident_bytes,
        reserve_bytes,
        incremental_launch_baseline_bytes,
        mem_frac,
    )
    if telemetry is not None:
        telemetry["resident_bytes"] = int(resident_bytes)
        telemetry["effective_mem_frac"] = float(mem_frac)
        telemetry["mem_policy_source"] = mem_policy_source
        telemetry["reserve_bytes"] = int(reserve_bytes)
        telemetry["incremental_launch_baseline_bytes"] = int(
            incremental_launch_baseline_bytes
        )
    chunk_cap = max_chunk
    retry_count = 0

    start = 0
    while start < len(target_coords):
        free_now = _free_mem_bytes()
        chunk, grid_bytes = _plan_target_chunk(
            resident_coords=resident_coords,
            target_coords=target_coords,
            start=start,
            free_bytes=free_now,
            budget_fraction=mem_frac,
            min_chunk=min_chunk,
            max_chunk=chunk_cap,
            incremental_launch_baseline_bytes=incremental_launch_baseline_bytes,
            n_trans=n_trans,
        )
        logger.debug(
            "type3 inverse-batch chunk planning | n_sources=%d n_targets=%d n_trans=%d start=%d free_vram=%d resident_bytes=%d reserve_bytes=%d incremental_launch_baseline_bytes=%d grid_bytes=%d per_target_bytes=%d chunk=%d retries=%d",
            len(resident_coords),
            len(target_coords),
            n_trans,
            start,
            free_now,
            resident_bytes,
            reserve_bytes,
            incremental_launch_baseline_bytes,
            grid_bytes,
            _per_target_bytes(dim, n_trans),
            chunk,
            retry_count,
        )
        if chunk == 0:
            if gpu_only:
                raise RuntimeError("GPU execution forced but the memory budget cannot fit the requested minimum chunk.")
            free_gpu_memory()
            _finish_telemetry(telemetry, fallback_reason="budget-exhausted")
            return np.stack(
                [
                    _cpu_fallback(
                        real_coords,
                        weights_arr[index],
                        q_coords,
                        eps,
                        True,
                    )
                    for index in range(n_trans)
                ],
                axis=0,
            )

        end = min(start + chunk, len(target_coords))
        target_slice = target_coords[start:end]

        d_target = None
        target_cols = None
        try:
            t0 = time.perf_counter()
            d_target = _as_device(target_slice)
            target_cols = [_contig(d_target[:, i]) for i in range(dim)]
            upload_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            d_chunk = _execute_inverse_batch_gpu(
                resident_cols=resident_cols,
                d_weights=d_weights,
                target_cols=target_cols,
                dim=dim,
                n_trans=n_trans,
                eps=eps,
            )
            launch_s = time.perf_counter() - t1
            t2 = time.perf_counter()
            if device_out and out_device is not None:
                out_device[:, start:end] = d_chunk
                used_pinned = False
                d2h_bytes = 0
            else:
                host_chunk, used_pinned, d2h_bytes = _copy_device_to_host_with_meta(d_chunk)
                out_host[:, start:end] = host_chunk
            download_s = time.perf_counter() - t2
            _log_chunk_timing(
                label="type3 inverse-batch",
                upload_s=upload_s,
                launch_s=launch_s,
                download_s=download_s,
                chunk_size=int(chunk),
                used_pinned=used_pinned,
                experimental_overlap=experimental_overlap,
            )
            _record_chunk_telemetry(
                telemetry,
                chunk_index=len(telemetry["chunks"]) if telemetry is not None else 0,
                chunk_size=int(chunk),
                n_sources=int(len(resident_coords)),
                n_targets=int(len(target_coords)),
                n_trans=int(n_trans),
                free_vram_bytes=int(free_now),
                resident_bytes=int(resident_bytes),
                reserve_bytes=int(reserve_bytes),
                incremental_launch_baseline_bytes=int(
                    incremental_launch_baseline_bytes
                ),
                grid_bytes=int(grid_bytes),
                per_target_bytes=int(_per_target_bytes(dim, n_trans)),
                upload_seconds=float(upload_s),
                launch_seconds=float(launch_s),
                download_seconds=float(download_s),
                d2h_bytes=int(d2h_bytes),
                used_pinned_host_copy=bool(used_pinned),
                retry_count=int(retry_count),
            )
        except (
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
            cp.cuda.driver.CUDADriverError,
            MemoryError,
            RuntimeError,
            OSError,
        ) as err:
            if not _is_retryable_resource_error(err):
                raise
            if chunk > min_chunk:
                chunk_cap = max(min_chunk, chunk // 2)
                retry_count += 1
                free_gpu_memory()
                logger.debug(
                    "type3 inverse-batch retrying on GPU | reason=%s new_chunk_cap=%d retries=%d",
                    type(err).__name__,
                    chunk_cap,
                    retry_count,
                )
                continue
            if gpu_only:
                raise RuntimeError("GPU execution forced but failed") from err
            free_gpu_memory()
            logger.debug(
                "type3 inverse-batch fallback to CPU | reason=%s retries=%d",
                type(err).__name__,
                retry_count,
            )
            _finish_telemetry(telemetry, fallback_reason=type(err).__name__)
            host = np.stack(
                [
                    _cpu_fallback(
                        real_coords,
                        weights_arr[index],
                        q_coords,
                        eps,
                        True,
                )
                for index in range(n_trans)
            ],
            axis=0,
        )
            return host
        finally:
            d_target = None
            target_cols = None
            d_chunk = None

        chunk_cap = max_chunk
        retry_count = 0
        start = end

    d_resident = None
    d_weights = None
    resident_cols = None
    if device_out and out_device is not None:
        t_end = time.perf_counter()
        result, used_pinned, final_d2h_bytes = _copy_device_to_host_with_meta(out_device)
        _log_chunk_timing(
            label="type3 inverse-batch-finalize",
            upload_s=0.0,
            launch_s=0.0,
            download_s=time.perf_counter() - t_end,
            chunk_size=int(len(target_coords)),
            used_pinned=used_pinned,
            experimental_overlap=experimental_overlap,
        )
        _finish_telemetry(
            telemetry,
            final_d2h_bytes=int(final_d2h_bytes),
        )
    else:
        result = out_host
        _finish_telemetry(telemetry, final_d2h_bytes=int(np.asarray(result).nbytes))
    out_device = None
    free_gpu_memory()
    return result


###############################################################################
#  Core driver                                                                #
###############################################################################
def _batched_type3(
    real_coords: np.ndarray,
    weights: np.ndarray,
    q_coords: np.ndarray,
    *,
    eps: float,
    inverse: bool,
    mem_frac: Optional[float],
    min_chunk: int,
    max_chunk: Optional[int],
    prefer_cpu: bool,
    gpu_only: bool,
) -> np.ndarray:
    if not (_CPU_ONLY or prefer_cpu):
        _ensure_gpu_backend()
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1-, 2-, and 3-D inputs supported")

    if gpu_only and (_CPU_ONLY or prefer_cpu):
        raise RuntimeError("GPU execution forced but CPU execution was requested.")
    if gpu_only and not _GPU_AVAILABLE:
        raise RuntimeError("GPU execution forced but no CUDA/cuFINUFFT backend is available.")

    if _CPU_ONLY or prefer_cpu or not _GPU_AVAILABLE:
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)
    experimental_overlap = _experimental_overlap_enabled()
    if experimental_overlap:
        logger.debug(
            "Experimental overlap requested for %s type3, but the stable serialized path remains active; collecting timing diagnostics only.",
            "inverse" if inverse else "forward",
        )
    try:
        _ensure_gpu_kernels()
    except ImportError:
        if gpu_only:
            raise RuntimeError("GPU execution forced but cufinufft is unavailable.")
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

    resident_coords, resident_weights, target_coords = _select_type3_sides(
        real_coords,
        weights,
        q_coords,
        inverse=inverse,
    )
    resident_coords = np.asarray(resident_coords, dtype=np.float64)
    resident_weights = np.asarray(resident_weights, dtype=np.complex128)
    target_coords = np.asarray(target_coords, dtype=np.float64)
    out = np.zeros(len(target_coords), dtype=np.complex128)
    telemetry = _begin_telemetry(
        mode="inverse" if inverse else "forward",
        n_sources=int(len(resident_coords)),
        n_targets=int(len(target_coords)),
        n_trans=1,
        resident_bytes=None,
        effective_mem_frac=None,
        mem_policy_source=None,
        experimental_overlap=experimental_overlap,
    )

    d_resident = _as_device(resident_coords, allow_fail=True)
    if d_resident is None:
        _finish_telemetry(telemetry, fallback_reason="resident-upload-failed")
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)
    d_weights = _as_device(resident_weights, allow_fail=True)
    if d_weights is None:
        _finish_telemetry(telemetry, fallback_reason="weight-upload-failed")
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

    resident_cols = [_contig(d_resident[:, i]) for i in range(dim)]
    d_weights = _contig(d_weights)
    resident_bytes = _resident_bytes(resident_coords, resident_weights)
    initial_free = _free_mem_bytes()
    reserve_bytes, mem_frac, mem_policy_source = _resolve_budget_policy(
        mem_frac=mem_frac,
        free_bytes=initial_free,
        resident_bytes=resident_bytes,
    )
    incremental_launch_baseline_bytes = 0
    logger.debug(
        "type3 %s memory policy | source=%s free_vram=%d resident_bytes=%d reserve_bytes=%d incremental_launch_baseline_bytes=%d effective_mem_frac=%.3f",
        "inverse" if inverse else "forward",
        mem_policy_source,
        initial_free,
        resident_bytes,
        reserve_bytes,
        incremental_launch_baseline_bytes,
        mem_frac,
    )
    if telemetry is not None:
        telemetry["resident_bytes"] = int(resident_bytes)
        telemetry["effective_mem_frac"] = float(mem_frac)
        telemetry["mem_policy_source"] = mem_policy_source
        telemetry["reserve_bytes"] = int(reserve_bytes)
        telemetry["incremental_launch_baseline_bytes"] = int(
            incremental_launch_baseline_bytes
        )
    chunk_cap = max_chunk

    start = 0
    retry_count = 0
    while start < len(target_coords):
        free_now = _free_mem_bytes()
        chunk, _grid_bytes = _plan_target_chunk(
            resident_coords=resident_coords,
            target_coords=target_coords,
            start=start,
            free_bytes=free_now,
            budget_fraction=mem_frac,
            min_chunk=min_chunk,
            max_chunk=chunk_cap,
            incremental_launch_baseline_bytes=incremental_launch_baseline_bytes,
        )
        logger.debug(
            "type3 %s chunk planning | n_sources=%d n_targets=%d start=%d free_vram=%d resident_bytes=%d reserve_bytes=%d incremental_launch_baseline_bytes=%d grid_bytes=%d per_target_bytes=%d chunk=%d retries=%d",
            "inverse" if inverse else "forward",
            len(resident_coords),
            len(target_coords),
            start,
            free_now,
            resident_bytes,
            reserve_bytes,
            incremental_launch_baseline_bytes,
            _grid_bytes,
            _per_target_bytes(dim),
            chunk,
            retry_count,
        )
        if chunk == 0:
            if gpu_only:
                raise RuntimeError("GPU execution forced but the memory budget cannot fit the requested minimum chunk.")
            free_gpu_memory()
            logger.debug(
                "type3 %s fallback to CPU | reason=budget-exhausted n_sources=%d n_targets=%d",
                "inverse" if inverse else "forward",
                len(resident_coords),
                len(target_coords),
            )
            _finish_telemetry(telemetry, fallback_reason="budget-exhausted")
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

        end = min(start + chunk, len(target_coords))
        target_slice = target_coords[start:end]

        d_target = None
        target_cols = None
        d_res = None
        try:
            t0 = time.perf_counter()
            d_target = _as_device(target_slice)
            target_cols = [_contig(d_target[:, i]) for i in range(dim)]
            upload_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            d_res = _adaptive_gpu_launch(
                dim,
                resident_cols,
                d_weights,
                target_cols,
                eps,
                inverse,
            )
            launch_s = time.perf_counter() - t1
            t2 = time.perf_counter()
            host_chunk, used_pinned, d2h_bytes = _copy_device_to_host_with_meta(d_res)
            out[start:end] = host_chunk
            download_s = time.perf_counter() - t2
            _log_chunk_timing(
                label=f"type3 {'inverse' if inverse else 'forward'}",
                upload_s=upload_s,
                launch_s=launch_s,
                download_s=download_s,
                chunk_size=int(chunk),
                used_pinned=used_pinned,
                experimental_overlap=experimental_overlap,
            )
            _record_chunk_telemetry(
                telemetry,
                chunk_index=len(telemetry["chunks"]) if telemetry is not None else 0,
                chunk_size=int(chunk),
                n_sources=int(len(resident_coords)),
                n_targets=int(len(target_coords)),
                n_trans=1,
                free_vram_bytes=int(free_now),
                resident_bytes=int(resident_bytes),
                reserve_bytes=int(reserve_bytes),
                incremental_launch_baseline_bytes=int(
                    incremental_launch_baseline_bytes
                ),
                grid_bytes=int(_grid_bytes),
                per_target_bytes=int(_per_target_bytes(dim)),
                upload_seconds=float(upload_s),
                launch_seconds=float(launch_s),
                download_seconds=float(download_s),
                d2h_bytes=int(d2h_bytes),
                used_pinned_host_copy=bool(used_pinned),
                retry_count=int(retry_count),
            )
        except (
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
            cp.cuda.driver.CUDADriverError,
            MemoryError,
            RuntimeError,
            OSError,
        ) as err:
            if not _is_retryable_resource_error(err):
                raise
            if chunk > min_chunk:
                chunk_cap = max(min_chunk, chunk // 2)
                retry_count += 1
                free_gpu_memory()
                logger.debug(
                    "type3 %s retrying on GPU | reason=%s new_chunk_cap=%d retries=%d",
                    "inverse" if inverse else "forward",
                    type(err).__name__,
                    chunk_cap,
                    retry_count,
                )
                continue
            if gpu_only:
                raise RuntimeError("GPU execution forced but failed") from err
            free_gpu_memory()
            logger.debug(
                "type3 %s fallback to CPU | reason=%s retries=%d",
                "inverse" if inverse else "forward",
                type(err).__name__,
                retry_count,
            )
            _finish_telemetry(telemetry, fallback_reason=type(err).__name__)
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)
        finally:
            d_target = None
            target_cols = None
            d_res = None

        chunk_cap = max_chunk
        retry_count = 0
        start = end

    d_resident = None
    d_weights = None
    resident_cols = None
    free_gpu_memory()
    _finish_telemetry(telemetry, final_d2h_bytes=int(np.asarray(out).nbytes))
    return out


###############################################################################
#  GPU kernel chooser                                                         #
###############################################################################
def _adaptive_gpu_launch(dim, resident_cols, d_w, target_cols, eps, inverse):
    isign = -1 if inverse else 1
    # Single-source the deterministic, history-independent back-off ladder.
    subprobs = _subprob_order(dim, 1)

    for s in subprobs:
        kw = _build_gpu_launch_kwargs(gpu_maxsubprobsize=s)
        try:
            return _launch_once(
                dim,
                resident_cols,
                d_w,
                target_cols,
                eps,
                isign,
                kw,
            )
        except (
            cp.cuda.memory.OutOfMemoryError,
            RuntimeError,
            OSError,
            cp.cuda.runtime.CUDARuntimeError,
            cp.cuda.driver.CUDADriverError,
        ) as e:
            if _is_retryable_resource_error(e):
                continue
            raise
    raise RuntimeError("launch-resource-exhausted: all no-shmem kernel variants failed")


def _launch_once(dim, resident_cols, d_w, target_cols, eps, isign, kw):
    try:
        if dim == 1:
            return _KER[1](
                resident_cols[0],
                d_w,
                target_cols[0],
                eps=eps,
                isign=isign,
                **kw,
            )
        if dim == 2:
            return _KER[2](
                resident_cols[0],
                resident_cols[1],
                d_w,
                target_cols[0],
                target_cols[1],
                eps=eps,
                isign=isign,
                **kw,
            )
        return _KER[3](*resident_cols, d_w, *target_cols, eps=eps, isign=isign, **kw)
    finally:
        _warn_if_below_headroom("type3-simple-launch")


def _set_type3_points(plan, *, dim: int, source_cols, target_cols) -> None:
    if dim == 1:
        plan.setpts(source_cols[0], None, None, target_cols[0])
        return
    if dim == 2:
        plan.setpts(
            source_cols[0],
            source_cols[1],
            None,
            target_cols[0],
            target_cols[1],
        )
        return
    plan.setpts(
        source_cols[0],
        source_cols[1],
        source_cols[2],
        target_cols[0],
        target_cols[1],
        target_cols[2],
    )


def _execute_inverse_batch_gpu(
    *,
    resident_cols,
    d_weights,
    target_cols,
    dim: int,
    n_trans: int,
    eps: float,
):
    """Inverse type-3 NUFFT with bounded-cache plan reuse.

    Plan cache is keyed only on the Plan() constructor arguments
    ``(dim, type=3, isign=-1, n_trans, eps, dtype, gpu_maxsubprobsize)``.
    Different point sets reuse the same cached Plan via ``setpts()`` —
    that is the operation cuFINUFFT is built around amortizing. The point
    coords are NOT part of the cache key, so the cache stays small (≈ 6
    distinct subprobs × small set of n_trans values) regardless of how
    many distinct geometries the workload exposes. Both ``setpts`` and
    ``execute`` happen under the per-plan lock because cuFINUFFT Plans
    are not thread-safe.
    """
    import cufinufft                   # type: ignore

    subprobs = _subprob_order(dim, n_trans)
    oom_retry_done = False

    for s in subprobs:
        cache_key = (int(dim), 3, -1, int(n_trans), float(eps), int(s))

        def _build(_s=s):
            return cufinufft.Plan(
                3,
                dim,
                n_trans=n_trans,
                eps=eps,
                isign=-1,
                dtype="complex128",
                **_build_gpu_launch_kwargs(gpu_maxsubprobsize=_s),
            )

        def _do_call():
            plan, per_plan_lock, cached = _plan_cache_get_or_build(cache_key, _build)
            try:
                with per_plan_lock:
                    # Re-bind points for this call (cheap relative to Plan ctor).
                    _set_type3_points(
                        plan,
                        dim=dim,
                        source_cols=resident_cols,
                        target_cols=target_cols,
                    )
                    result = plan.execute(d_weights)
                _warn_if_below_headroom("type3-plan-execute")
                return result
            finally:
                # When the cache is disabled, destroy the plan immediately so
                # cuFINUFFT scratch (allocated via raw cudaMalloc, bypassing
                # the CuPy pool) is reclaimed before the next call. This is
                # the only mechanism that bounds plan-internal VRAM under
                # multi-process workers on a single GPU.
                if not cached:
                    _destroy_plan_quietly(plan)

        try:
            result = _do_call()
            # Best-effort: return any pool blocks released by the call back
            # to the device. Live cuFINUFFT Plan internals (kept alive by
            # the cache) are NOT in the free list; only this task's
            # working scratch is. Without this, the pool monotonically
            # grows under concurrent task load.
            _free_cupy_pool_blocks()
            return result
        except (
            cp.cuda.memory.OutOfMemoryError,
            RuntimeError,
            OSError,
            cp.cuda.runtime.CUDARuntimeError,
            cp.cuda.driver.CUDADriverError,
        ) as e:
            if (
                cp is not None
                and isinstance(e, cp.cuda.memory.OutOfMemoryError)
                and not oom_retry_done
            ):
                # Drop cache + free pool blocks, then retry at this subprob.
                _clear_plan_cache()
                oom_retry_done = True
                try:
                    result = _do_call()
                    _free_cupy_pool_blocks()
                    return result
                except Exception as e2:
                    if _is_retryable_resource_error(e2):
                        continue
                    raise
            if _is_retryable_resource_error(e):
                continue
            raise
    raise RuntimeError("launch-resource-exhausted: all inverse-batch Plan variants failed")


###############################################################################
#  CPU fallback                                                               #
###############################################################################
def _cpu_fallback(
    real_coords: np.ndarray,
    weights: np.ndarray,
    q_coords: np.ndarray,
    eps: float,
    inverse: bool,
    *,
    batch: int = 2_000_000,
) -> np.ndarray:
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Unsupported dimensionality")

    global _FINUFFT3
    if _FINUFFT3 is None:
        _FINUFFT3 = {}
    if dim not in _FINUFFT3:
        try:
            import finufft
        except ModuleNotFoundError:
            _warn_direct_cpu_fallback()
            return _direct_cpu_fallback(
                real_coords,
                weights,
                q_coords,
                inverse=inverse,
                batch=batch,
            )
        _FINUFFT3[dim] = {1: finufft.nufft1d3,
                          2: finufft.nufft2d3,
                          3: finufft.nufft3d3}[dim]
    nufft = _FINUFFT3[dim]

    isign = -1 if inverse else 1
    resident_coords, resident_weights, target_coords = _select_type3_sides(
        real_coords,
        weights,
        q_coords,
        inverse=inverse,
    )
    resident_weights = np.asarray(resident_weights, dtype=np.complex128)
    resident_split = [
        resident_coords[:, i].astype(np.float64) for i in range(dim)
    ]
    out = np.zeros(len(target_coords), dtype=np.complex128)

    start = 0
    while start < len(target_coords):
        end = min(start + batch, len(target_coords))
        target_chunk = [target_coords[start:end, i].astype(np.float64) for i in range(dim)]
        args = (*resident_split, resident_weights, *target_chunk)
        res = nufft(*args, eps=eps, isign=isign)
        out[start:end] = res
        start = end

    return out


def _warn_direct_cpu_fallback() -> None:
    global _DIRECT_CPU_FALLBACK_WARNED
    if _DIRECT_CPU_FALLBACK_WARNED:
        return
    warnings.warn(
        "finufft is not available; using a slow direct CPU fallback. "
        "This path is intended for smoke-scale validation only.",
        RuntimeWarning,
        stacklevel=3,
    )
    _DIRECT_CPU_FALLBACK_WARNED = True


def _direct_cpu_fallback(
    real_coords: np.ndarray,
    weights: np.ndarray,
    q_coords: np.ndarray,
    *,
    inverse: bool,
    batch: int,
) -> np.ndarray:
    sources = np.asarray(q_coords if inverse else real_coords, dtype=np.float64)
    targets = np.asarray(real_coords if inverse else q_coords, dtype=np.float64)
    coeffs = np.asarray(weights, dtype=np.complex128)
    if sources.ndim != 2 or targets.ndim != 2:
        raise ValueError("NUFFT coordinates must be 2-D arrays.")
    if sources.shape[1] != targets.shape[1]:
        raise ValueError("Source and target coordinates must have matching dimensionality.")

    n_sources = int(sources.shape[0])
    n_targets = int(targets.shape[0])
    if coeffs.shape[0] != n_sources:
        raise ValueError("Weights length must match the source coordinate count.")

    target_batch = min(int(batch), _DIRECT_CPU_FALLBACK_MAX_TARGETS)
    target_batch = max(1, target_batch)
    isign = -1 if inverse else 1
    sources_t = np.ascontiguousarray(sources.T)

    return direct_dft_type3(
        targets,
        sources_t,
        coeffs,
        isign=isign,
        batch=target_batch,
    )


###############################################################################
#  Mute harmless destructor warnings                                          #
###############################################################################
warnings.filterwarnings("ignore", message=r"Error destroying plan.")
