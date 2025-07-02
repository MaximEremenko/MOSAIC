# -*- coding: utf-8 -*-
"""
GPU‑safe cuFINUFFT wrapper for Mosaic 
keeps the familiar public API while adding **automatic
chunking** and basic **memory‑aware fall‑backs** so that transforms no
longer crash with `cupy.cuda.memory.OutOfMemoryError` when the problem
size exceeds available VRAM.

---------
* Allocate **static data** (real‑space coordinates and weights) once on
the device if it fits.  If not, transparently falls back to a pure‑CPU
  FINUFFT call (optional, see `prefer_cpu` flag).
* Stream the **query points** (`q_coords`) in chunks whose size is
  chosen so that each batch uses at most ~50 % of currently free GPU
  memory.  The chunk size is recomputed adaptively if an allocation
  still fails.
* Works for 1‑, 2‑ and 3‑D type‑3 (non‑uniform → non‑uniform) NUFFTs and
  their inverses.
* Parameters
    • `prefer_cpu` – if `True`, silently switches to finufft (CPU) when
      GPU memory is insufficient; otherwise raises.
    • `mem_frac`   – fraction of the *current* free VRAM a single batch
      is allowed to occupy (default 0.5).
    • `min_chunk`  – minimum batch size before giving up.

Usage::

    out = execute_cunufft(r_xyz, c, q_hkl)            # forward
    rho = execute_inverse_cunufft(q_hkl, fq, r_xyz)   # inverse
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import cupy as cp
import cufinufft  # type: ignore

# Optional CPU fallback – imported lazily
_FINUFFT3: Callable | None = None  # finufft.nufft3d3

###############################################################################
# Helper utilities                                                             #
###############################################################################

def _free_mem_bytes() -> int:
    free, _ = cp.cuda.runtime.memGetInfo()
    return int(free)


def _auto_chunk_size(dim: int, mem_frac: float, per_pt: int) -> int:
    tgt = int(_free_mem_bytes() * mem_frac)
    return max(tgt // per_pt, 1)


def _as_device(arr: np.ndarray, allow_fail: bool = False):
    try:
        return cp.asarray(arr)
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError):
        if allow_fail:
            return None
        raise


def _contig(x) -> cp.ndarray:
    return x if x.flags.c_contiguous else cp.ascontiguousarray(x)

###############################################################################
# Kernel map                                                                   #
###############################################################################
_KER = {1: cufinufft.nufft1d3, 2: cufinufft.nufft2d3, 3: cufinufft.nufft3d3}

###############################################################################
# Public API                                                                   #
###############################################################################

def execute_cunufft(
    real_coords: np.ndarray,
    c: np.ndarray,
    q_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    mem_frac: float = 0.9,
    min_chunk: int = 64_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    return _batched_type3(
        real_coords,
        c,
        q_coords,
        eps=eps,
        inverse=False,
        mem_frac=mem_frac,
        min_chunk=min_chunk,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
    )


def execute_inverse_cunufft(
    q_coords: np.ndarray,
    c: np.ndarray,
    real_coords: np.ndarray,
    *,
    eps: float = 1e-12,
    mem_frac: float = 0.9,
    min_chunk: int = 64_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    return _batched_type3(
        real_coords,
        c,
        q_coords,
        eps=eps,
        inverse=True,
        mem_frac=mem_frac,
        min_chunk=min_chunk,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
    )

###############################################################################
# Core logic                                                                   #
###############################################################################

def _batched_type3(
    real_coords: np.ndarray,
    c: np.ndarray,
    q_coords: np.ndarray,
    *,
    eps: float,
    inverse: bool,
    mem_frac: float,
    min_chunk: int,
    prefer_cpu: bool,
    gpu_only: bool,
) -> np.ndarray:
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1D/2D/3D supported")

    # ── Attempt to keep target grid on GPU ───────────────────────────────────
    d_real = _as_device(real_coords, allow_fail=True)
    if d_real is None:
        return _cpu_fallback(real_coords, c, q_coords, eps, inverse)

    cols = [_contig(d_real[:, i]) for i in range(dim)]

    # Preload full weight vector for forward transform
    d_c_full = None
    if not inverse:
        d_c_full = _as_device(c, allow_fail=True)
        if d_c_full is None:
            return _cpu_fallback(real_coords, c, q_coords, eps, inverse)

    per_q = dim * 8 + 16
    chunk = max(_auto_chunk_size(dim, mem_frac, per_q), min_chunk)

    out = np.zeros(len(real_coords) if inverse else len(q_coords), dtype=np.complex128)

    start = 0
    while start < len(q_coords):
        end = min(start + chunk, len(q_coords))
        q_slice = q_coords[start:end]
        if inverse:
            q_slice = -q_slice

        try:
            d_q = _as_device(q_slice)
        except cp.cuda.memory.OutOfMemoryError:
            if chunk // 2 < min_chunk:
                return _cpu_fallback(real_coords, c, q_coords, eps, inverse)
            chunk //= 2
            continue

        d_c = d_c_full if not inverse else _as_device(c[start:end])
        q_cols = [_contig(d_q[:, i]) for i in range(dim)]

        try:
            d_res = _launch_gpu(dim, cols, d_c, q_cols, eps, inverse)
        except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
            # Plan creation or execution failed – fall back or raise
            if gpu_only:
                raise
            return _cpu_fallback(real_coords, c, q_coords, eps, inverse)

        if inverse:
            out += cp.asnumpy(d_res)
        else:
            out[start:end] = cp.asnumpy(d_res)

        del d_q, q_cols, d_res, d_c
        cp.get_default_memory_pool().free_all_blocks()
        start = end

    return out

###############################################################################
# GPU launcher                                                                 #
###############################################################################

def _launch_gpu(dim, cols, d_c, q_cols, eps, inverse):
    if dim == 1:
        return _KER[1](*(cols + [d_c] + q_cols) if not inverse else (q_cols[0], d_c, cols[0]), eps=eps)
    if dim == 2:
        if not inverse:
            args = (cols[0], cols[1], d_c, q_cols[0], q_cols[1])
        else:
            args = (q_cols[0], q_cols[1], d_c, cols[0], cols[1])
        return _KER[2](*args, eps=eps)
    # dim == 3
    if not inverse:
        args = (*cols, d_c, *q_cols)
    else:
        args = (*q_cols, d_c, *cols)
    return _KER[3](*args, eps=eps)

###############################################################################
# CPU fallback                                                                 #
###############################################################################

def _cpu_fallback(real_coords, c, q_coords, eps, inverse):
    global _FINUFFT3
    if _FINUFFT3 is None:
        try:
            import finufft
            _FINUFFT3 = finufft.nufft3d3  # type: ignore
        except ImportError as exc:
            raise MemoryError("finufft not installed and GPU path failed.") from exc

    if inverse:
        q_coords = -q_coords
        isign = -1
    else:
        isign = 1

    x, y, z = [real_coords[:, i].astype(np.float64) for i in range(3)]
    s, t, u = [q_coords[:, i].astype(np.float64) for i in range(3)]

    return _FINUFFT3(x, y, z, c.astype(np.complex128), s, t, u, isign, eps)