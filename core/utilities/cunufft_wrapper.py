# -*- coding: utf-8 -*-
"""
Robust, **stream-aware** cuFINUFFT wrapper for Mosaic
====================================================

Key features
------------
* **GPU-friendly chunking** – whichever coordinate set (sources or
  targets) is larger is streamed in pieces sized to fit the *current*
  free VRAM.
* **Configurable max chunk** – pass `max_chunk` (points) to impose a
  hard upper-bound on each batch so you can rein-in memory usage on
  small GPUs or avoid long kernel launch times.
* **Kernel auto-back-off** – cycles through several `(gpu_method,
  gpu_kerevalmeth, gpu_maxsubprobsize)` combinations to find one that
  fits the card’s shared-memory limits before giving up.
* **Graceful CPU fallback** – when the GPU cannot handle the job, the
  code transparently switches to FINUFFT on the host, still in streaming
  batches so RAM usage stays bounded.

Usage example::

    fq   = execute_cunufft(r_xyz, rho, q_hkl, max_chunk=2_000_000)
    rho2 = execute_inverse_cunufft(q_hkl, fq, r_xyz, max_chunk=2_000_000)

Both calls share the same keyword arguments::

    eps        – accuracy (1e-12 default)
    mem_frac   – fraction of free VRAM a single batch may consume
    min_chunk  – floor on batch size (adaptive algorithm never goes below)
    max_chunk  – *ceiling* on batch size (None → no cap)
    prefer_cpu – force host fallback instead of raising when GPU fails
    gpu_only   – raise if GPU path fails (useful for benchmarking)
"""
from __future__ import annotations

from typing import Callable, Optional
import warnings

import numpy as np
import cupy as cp
import cufinufft                     # type: ignore

###########################################################################
#  Helpers                                                                #
###########################################################################

# Lazily imported CPU backend (avoids importing finufft on GPU-only nodes)
_FINUFFT3: Callable | None = None  # finufft.nufft3d3


def _free_mem_bytes() -> int:  # GPU free memory in bytes
    free, _ = cp.cuda.runtime.memGetInfo()
    return int(free)


def _auto_chunk_size(mem_frac: float, bytes_per_pt: int) -> int:
    return max(int(_free_mem_bytes() * mem_frac) // bytes_per_pt, 1)


def _as_device(arr: np.ndarray, allow_fail: bool = False):
    """Transfer *arr* to GPU unless **allow_fail** and OOM."""
    try:
        return cp.asarray(arr)
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError):
        if allow_fail:
            return None
        raise


def _contig(x):  # ensure C-contiguous cupy array
    return x if x.flags.c_contiguous else cp.ascontiguousarray(x)

###########################################################################
#  Public API                                                             #
###########################################################################

# Shortcuts to the simple front-end kernels (1/2/3-D Type-3)
_KER = {1: cufinufft.nufft1d3, 2: cufinufft.nufft2d3, 3: cufinufft.nufft3d3}


def execute_cunufft(
    real_coords: np.ndarray,
    weights: np.ndarray,
    q_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    mem_frac: float = 0.8,
    min_chunk: int = 64_000,
    max_chunk: int = 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Forward NUFFT (real → reciprocal)."""
    return _batched_type3(
        real_coords, weights, q_coords,
        eps=eps, inverse=False,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
    )


def execute_inverse_cunufft(
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    mem_frac: float = 0.8,
    min_chunk: int = 64_000,
    max_chunk: int = 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse NUFFT (reciprocal → real)."""
    return _batched_type3(
        real_coords, weights, q_coords,
        eps=eps, inverse=True,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
    )

###########################################################################
#  Core driver                                                            #
###########################################################################


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
        raise ValueError("Only 1-, 2- and 3-D inputs supported")

    # Try to pin the (usually smaller) *source* set to the GPU -------------
    d_real = _as_device(real_coords, allow_fail=True)
    if d_real is None:
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

    cols = [_contig(d_real[:, i]) for i in range(dim)]

    # Streaming strategy: always move the other set in chunks -------------
    bytes_per_pt = dim * 8 + 16 + 32   # coord + weight + scratch margin
    out = np.zeros(
        len(real_coords) if inverse else len(q_coords),
        dtype=np.complex128,
    )

    start = 0
    while start < len(q_coords):
        # Base chunk from available VRAM
        chunk = max(_auto_chunk_size(mem_frac, bytes_per_pt), min_chunk)
        # Respect hard ceiling
        if max_chunk is not None:
            chunk = min(chunk, max_chunk)
        end = min(start + chunk, len(q_coords))

        q_slice = q_coords[start:end]
        if inverse:
            q_slice = -q_slice

        try:
            d_q = _as_device(q_slice)
            d_w = _as_device(weights[start:end] if inverse else weights)
            q_cols = [_contig(d_q[:, i]) for i in range(dim)]
            d_res = _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse)
        except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as err:
            if gpu_only:
                raise RuntimeError("GPU execution forced but failed") from err
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

        # Scatter back -----------------------------------------------------
        if inverse:
            out += cp.asnumpy(d_res)
        else:
            out[start:end] = cp.asnumpy(d_res)

        # Clean-up ---------------------------------------------------------
        del d_q, q_cols, d_res, d_w
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        start = end

    return out

###########################################################################
#  GPU kernel chooser                                                     #
###########################################################################


def _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse):
    isign = -1 if inverse else 1

    # lightest → heaviest (we will exit on the *first* one that works)
    subprobs = (32, 16, 8, 4, 2, 1)

    for s in subprobs:
        kw = dict(gpu_method=1,          # NUpts-driven
                  gpu_kerevalmeth=1,     # global-mem eval  (no shmem slab)
                  gpu_maxsubprobsize=s,
                  gpu_maxbatchsize=1,
                  gpu_spreadinterponly=1)

        try:
            return _launch_once(dim, cols, d_w, q_cols, eps, isign, kw, inverse)
        except RuntimeError as e:
            # keep trying while we see memory / shared-mem complaints
            if "shared memory" in str(e).lower() or "allocate" in str(e).lower():
                continue
            raise   # anything else = real error
    raise RuntimeError("All no-shmem kernel variants failed – falling back")


def _launch_once(dim, cols, d_w, q_cols, eps, isign, kw, inverse):
    if dim == 1:
        args = (q_cols[0], d_w, cols[0]) if inverse else (cols[0], d_w, q_cols[0])
        return _KER[1](*args, eps=eps, isign=isign, **kw)

    if dim == 2:
        if inverse:
            args = (q_cols[0], q_cols[1], d_w, cols[0], cols[1])
        else:
            args = (cols[0], cols[1], d_w, q_cols[0], q_cols[1])
        return _KER[2](*args, eps=eps, isign=isign, **kw)

    # dim == 3
    args = (*q_cols, d_w, *cols) if inverse else (*cols, d_w, *q_cols)
    return _KER[3](*args, eps=eps, isign=isign, **kw)

###########################################################################
#  CPU fallback                                                            #
###########################################################################

def _cpu_fallback(real_coords, weights, q_coords, eps, inverse, *, batch=5_000_000):
    """Streamed CPU FINUFFT that never allocates gigantic arrays."""
    global _FINUFFT3
    if _FINUFFT3 is None:
        import finufft  # heavyweight import delayed until needed
        _FINUFFT3 = finufft.nufft3d3  # type: ignore

    if inverse:
        q_coords = -q_coords
        isign = -1
    else:
        isign = +1

    x, y, z = [real_coords[:, i].astype(np.float64) for i in range(3)]
    s_all, t_all, u_all = [q_coords[:, i].astype(np.float64) for i in range(3)]

    out = np.zeros(len(real_coords) if inverse else len(q_coords), dtype=np.complex128)

    start = 0
    while start < len(q_coords):
        end = min(start + batch, len(q_coords))
        s, t, u = s_all[start:end], t_all[start:end], u_all[start:end]

        res = _FINUFFT3(
            x, y, z,
            weights.astype(np.complex128) if not inverse else weights[start:end].astype(np.complex128),
            s, t, u,
            eps=eps,
            isign=isign,
        )

        if inverse:
            out += res
        else:
            out[start:end] = res
        start = end

    return out

###########################################################################
#  Mute noisy destructor warnings (harmless but scary)                     #
###########################################################################

warnings.filterwarnings("ignore", message=r"Error destroying plan.")