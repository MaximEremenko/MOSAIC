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
import logging
import os
import time
import warnings
import numpy as np


logger = logging.getLogger(__name__)
_LAST_NUFFT_TELEMETRY = None

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
    import cufinufft                   # type: ignore

    subprobs = (32, 16, 8, 4, 2, 1)
    for s in subprobs:
        try:
            plan = cufinufft.Plan(
                3,
                dim,
                n_trans=n_trans,
                eps=eps,
                isign=-1,
                dtype="complex128",
                **_build_gpu_launch_kwargs(gpu_maxsubprobsize=s),
            )
            _set_type3_points(
                plan,
                dim=dim,
                source_cols=resident_cols,
                target_cols=target_cols,
            )
            return plan.execute(d_weights)
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
