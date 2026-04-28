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
    return hashlib.sha1(host.view(np.uint8).tobytes()).digest()[:8]


def _coords_sig(cols) -> bytes:
    if cols is None:
        return b"\x00" * 8
    return b"".join(_coord_sig(c) for c in cols)


_PLAN_CACHE_MAX = int(os.getenv("MOSAIC_NUFFT_PLAN_CACHE_MAX", "0"))
_PLAN_CACHE: "dict[tuple, tuple]" = {}
_PLAN_CACHE_ORDER: list = []
_PLAN_CACHE_LOCK = threading.Lock()


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
            n_workers = _expected_worker_count()
            usable_pct = max(
                0.05,
                _DEFAULT_CUPY_POOL_GLOBAL_BUDGET_PCT - _DEFAULT_NON_POOL_HEADROOM_PCT,
            )
            limit_bytes = int(usable_pct * total_vram / max(1, n_workers))

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


# Apply at import; safe even when CuPy is absent.
_apply_cupy_pool_cap()
_SUCCESSFUL_SUBPROB: "dict[tuple[int, int], int]" = {}
_SUCCESSFUL_SUBPROB_LOCK = threading.Lock()
_DEFAULT_SUBPROBS: tuple = (32, 16, 8, 4, 2, 1)


def _subprob_order(dim: int, n_trans: int) -> tuple:
    """Return the retry sequence with the last-known-good size first."""
    with _SUCCESSFUL_SUBPROB_LOCK:
        preferred = _SUCCESSFUL_SUBPROB.get((int(dim), int(n_trans)))
    if preferred is None or preferred not in _DEFAULT_SUBPROBS:
        return _DEFAULT_SUBPROBS
    rest = tuple(s for s in _DEFAULT_SUBPROBS if s != preferred)
    return (preferred,) + rest


def _record_successful_subprob(dim: int, n_trans: int, subprob: int) -> None:
    with _SUCCESSFUL_SUBPROB_LOCK:
        _SUCCESSFUL_SUBPROB[(int(dim), int(n_trans))] = int(subprob)


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
    Call once, before the first execute_* function.
    """
    global _CPU_ONLY, _GPU_AVAILABLE, cp
    _CPU_ONLY = bool(flag)
    if _CPU_ONLY:
        _GPU_AVAILABLE = False
        cp = None                     # type: ignore
        return
    _probe_gpu_backend()


###############################################################################
#  CUDA / CuPy import with graceful degradation                               #
###############################################################################
cp = None                             # type: ignore
_GPU_AVAILABLE = False


def _probe_gpu_backend() -> None:
    global cp, _GPU_AVAILABLE
    if _CPU_ONLY:
        cp = None                     # type: ignore
        _GPU_AVAILABLE = False
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


_probe_gpu_backend()

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
    outputs: list[np.ndarray] = []
    start = 0
    while start < total_trans:
        end = min(start + width, total_trans)
        batch_weights = weights_arr[start:end]
        try:
            batch_result = _execute_inverse_cunufft_batch(
                q_coords=q_coords,
                weights_arr=batch_weights,
                real_coords=real_coords,
                eps=eps,
                mem_frac=mem_frac,
                min_chunk=min_chunk,
                max_chunk=max_chunk,
                prefer_cpu=prefer_cpu,
                gpu_only=gpu_only,
                device_out=False,
            )
            outputs.append(np.asarray(batch_result))
            start = end
        except Exception as exc:
            if width > 1 and _is_retryable_super_batch_error(exc):
                width = max(1, width // 2)
                continue
            raise
    return np.concatenate(outputs, axis=0)


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
    if weights_arr.shape[1] != len(q_coords):
        raise ValueError("weights shape must match q_coords on axis 1")
    experimental_overlap = _experimental_overlap_enabled()
    if experimental_overlap:
        logger.debug(
            "Experimental overlap requested for inverse batch, but the current path remains serialized with timing diagnostics only."
        )

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
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1-, 2-, and 3-D inputs supported")

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
    subprobs = (32, 16, 8, 4, 2, 1)

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
                    return plan.execute(d_weights)
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
            _record_successful_subprob(dim, n_trans, s)
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
                    _record_successful_subprob(dim, n_trans, s)
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
    out = np.zeros(n_targets, dtype=np.complex128)
    sources_t = np.ascontiguousarray(sources.T)

    start = 0
    while start < n_targets:
        end = min(start + target_batch, n_targets)
        target_chunk = targets[start:end]
        phase = target_chunk @ sources_t
        out[start:end] = np.exp(1j * isign * phase) @ coeffs
        start = end

    return out


###############################################################################
#  Mute harmless destructor warnings                                          #
###############################################################################
warnings.filterwarnings("ignore", message=r"Error destroying plan.")
