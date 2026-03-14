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
import os
import warnings
import numpy as np

###############################################################################
#  Global CPU-only switch                                                     #
###############################################################################
_CPU_ONLY = os.getenv("MOSAIC_NUFFT_CPU_ONLY", "0") == "1"
#_CPU_ONLY = True

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


###############################################################################
#  CUDA / CuPy import with graceful degradation                               #
###############################################################################
if _CPU_ONLY:
    cp = None                         # type: ignore
    _GPU_AVAILABLE = False
else:
    try:
        import cupy as cp             # noqa: E402
        try:
            _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        except cp.cuda.runtime.CUDARuntimeError:
            _GPU_AVAILABLE = False
    except ImportError:
        cp = None                     # type: ignore
        _GPU_AVAILABLE = False

# Lazily imported CPU backend (avoid importing finufft on GPU-only nodes)
_FINUFFT3: dict[int, Callable] | None = None
_DIRECT_CPU_FALLBACK_WARNED = False
_DIRECT_CPU_FALLBACK_MAX_TARGETS = 50_000

###############################################################################
#  Memory helpers                                                             #
###############################################################################
_SCRATCH_ALPHA = 1.1          # cuFINUFFT scratch ≈ 1 complex128 per target


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
    return max(chunk, min_chunk)


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
            cp.cuda.memory.MemoryError,
            cp.cuda.driver.CUDADriverError,
            cp.cuda.runtime.CUDARuntimeError):
        if allow_fail:
            return None
        raise


def _contig(x):
    """Ensure C-contiguous cupy array."""
    return x if x.flags.c_contiguous else cp.ascontiguousarray(x)


def _estimate_grid_bytes(real: np.ndarray, recip: np.ndarray) -> int:
    """
    Upper bound on the fine grid cuFINUFFT will allocate (complex128).
    """
    xyz = np.abs(np.vstack((real, recip))).max(axis=0)
    nf = ((2 * np.ceil(xyz) + 2 + 15) // 16) * 16
    return int(nf.prod()) * 16          # 16 B per complex128


###############################################################################
#  GPU kernel shortcuts                                                       #
###############################################################################
if _GPU_AVAILABLE:
    import cufinufft                   # type: ignore
    _KER = {1: cufinufft.nufft1d3,
            2: cufinufft.nufft2d3,
            3: cufinufft.nufft3d3}
else:
    _KER = {}                          # type: ignore


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
    mem_frac: float = 0.5,
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
        eps=eps, inverse=False,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
    )


def execute_inverse_cunufft(
    q_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    real_coords: np.ndarray | None = None,
    *,
    c: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    mem_frac: float = 0.50,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 32*256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse (Type-3) NUFFT: reciprocal → real."""
    weights = _resolve_weights(weights, c)
    if real_coords is None:
        raise ValueError("real_coords must be supplied for inverse transform")
    return _batched_type3(
        real_coords, weights, q_coords,
        eps=eps, inverse=True,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
    )


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
    mem_frac: float,
    min_chunk: int,
    max_chunk: Optional[int],
    prefer_cpu: bool,
    gpu_only: bool,
) -> np.ndarray:
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1-, 2-, and 3-D inputs supported")

    # ── early CPU routes ─────────────────────────────────────────────────
    if _CPU_ONLY or prefer_cpu or not _GPU_AVAILABLE:
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

    # ── resident GPU data (sources) ─────────────────────────────────────
    d_real = _as_device(real_coords, allow_fail=True)
    if d_real is None:                           # VRAM not enough for sources
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)
    cols = [_contig(d_real[:, i]) for i in range(dim)]

    # ── memory budget constants ─────────────────────────────────────────
    M = len(real_coords)
    bytes_sources = M * dim * 8
    bytes_weights_full = M * 16 if not inverse else 0
    bytes_output_full = M * 16 if inverse else 0
    grid_bytes = _estimate_grid_bytes(real_coords, q_coords)
    baseline = grid_bytes + bytes_sources + bytes_weights_full + bytes_output_full

    bytes_q = dim * 8
    bytes_weight_tgt = 16 if inverse else 0
    bytes_out_tgt = 16 if not inverse else 0
    bytes_scratch = int(_SCRATCH_ALPHA * 16)
    per_target = bytes_q + bytes_weight_tgt + bytes_out_tgt + bytes_scratch

    # host output
    out = np.zeros(M if inverse else len(q_coords), dtype=np.complex128)

    # clamp mem_frac to sane range
    mem_frac = float(np.clip(mem_frac, 0.05, 0.5))

    start = 0
    while start < len(q_coords):
        free_now = _free_mem_bytes()
        chunk = _max_chunk_size(
            free_bytes=free_now,
            baseline_bytes=baseline,
            per_target_bytes=per_target,
            mem_frac=mem_frac,
            user_cap=max_chunk,
            min_chunk=min_chunk,
        )
        if chunk == 0:
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

        end = min(start + chunk, len(q_coords))
        q_slice = q_coords[start:end]
        if inverse:
            q_slice = q_slice

        # ── GPU execution for this slice ────────────────────────────────
        try:
            d_q = _as_device(q_slice)
            d_w = _as_device(weights[start:end] if inverse else weights)
            q_cols = [_contig(d_q[:, i]) for i in range(dim)]

            d_res = _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse)

            if inverse:
                out += cp.asnumpy(d_res)
            else:
                out[start:end] = cp.asnumpy(d_res)

        except (cp.cuda.memory.OutOfMemoryError,
                cp.cuda.memory.MemoryError,
                cp.cuda.runtime.CUDARuntimeError,
                cp.cuda.driver.CUDADriverError,
                RuntimeError,
                OSError) as err:
            if gpu_only:
                raise RuntimeError("GPU execution forced but failed") from err
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

        finally:
            for name in ("d_q", "q_cols", "d_res", "d_w"):
                if name in locals():
                    del locals()[name]
            if _GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp.cuda.runtime.deviceSynchronize()
                except cp.cuda.runtime.CUDARuntimeError:
                    pass

        start = end

    return out


###############################################################################
#  GPU kernel chooser                                                         #
###############################################################################
def _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse):
    need = _estimate_grid_bytes(
        np.column_stack([c.get() for c in cols]),
        np.column_stack([c.get() for c in q_cols]),
    )
    if need > _free_mem_bytes() * 0.50:
        raise RuntimeError("fine grid exceeds available GPU memory")

    isign = -1 if inverse else 1
    subprobs = (32, 16, 8, 4, 2, 1)

    for s in subprobs:
        kw = dict(gpu_method=1, gpu_kerevalmeth=1,
                  gpu_maxsubprobsize=s,
                  gpu_maxbatchsize=1,
                  gpu_spreadinterponly=1)
        try:
            return _launch_once(dim, cols, d_w, q_cols, eps, isign, kw, inverse)
        except (RuntimeError,
                OSError,
                cp.cuda.runtime.CUDARuntimeError,
                cp.cuda.driver.CUDADriverError) as e:
            if ("shared memory" in str(e).lower() or "allocate" in str(e).lower()):
                continue
            raise
    raise RuntimeError("All no-shmem kernel variants failed – falling back")


def _launch_once(dim, cols, d_w, q_cols, eps, isign, kw, inverse):
    if dim == 1:
        args = (q_cols[0], d_w, cols[0]) if inverse else (cols[0], d_w, q_cols[0])
        return _KER[1](*args, eps=eps, isign=isign, **kw)
    if dim == 2:
        args = (q_cols[0], q_cols[1], d_w, cols[0], cols[1]) if inverse \
               else (cols[0], cols[1], d_w, q_cols[0], q_cols[1])
        return _KER[2](*args, eps=eps, isign=isign, **kw)
    args = (*q_cols, d_w, *cols) if inverse else (*cols, d_w, *q_cols)
    return _KER[3](*args, eps=eps, isign=isign, **kw)


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
    real_split = [real_coords[:, i].astype(np.float64) for i in range(dim)]
    recip_split = [q_coords[:, i].astype(np.float64) for i in range(dim)]

    out = np.zeros(len(real_coords) if inverse else len(q_coords),
                   dtype=np.complex128)

    start = 0
    while start < len(q_coords):
        end = min(start + batch, len(q_coords))
        recip_chunk = [a[start:end] for a in recip_split]

        if inverse:
            #recip_chunk = [-a for a in recip_chunk]
            c_chunk = weights[start:end]
            args = (*recip_chunk, c_chunk, *real_split)
        else:
            c_chunk = weights
            args = (*real_split, c_chunk, *recip_chunk)

        res = nufft(*args, eps=eps, isign=isign)

        if inverse:
            out += res
        else:
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
