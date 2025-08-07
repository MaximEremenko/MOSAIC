# -*- coding: utf-8 -*-
"""
Robust, **stream-aware** cuFINUFFT wrapper for Mosaic
====================================================

Key features
------------
* **GPU-friendly chunking** – whichever coordinate set (sources or
  targets) is larger is streamed in pieces sized to fit the *current*
  free VRAM.
* **Configurable max chunk** – pass max_chunk (points) to impose a
  hard upper-bound on each batch so you can rein-in memory usage on
  small GPUs or avoid long kernel launch times.
* **Kernel auto-back-off** – cycles through several (gpu_method,
  gpu_kerevalmeth, gpu_maxsubprobsize) combinations to find one that
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

#cp.memory.set_pinned_memory_allocator(None)
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
    
def _estimate_grid_bytes(real: np.ndarray, recip: np.ndarray) -> int:
    """
    Upper‑bound on the fine‑grid buffer cuFINUFFT will cudaMalloc.
    FINUFFT rounds each nf up to a multiple of 16; the grid is complex128.
    """
    xyz = np.abs(np.vstack((real, recip))).max(axis=0)
    nf  = ((2 * np.ceil(xyz) + 2 + 15) // 16) * 16     # round like FINUFFT
    return int(nf.prod()) * 16                         # 16 B per value


###########################################################################
#  Public API                                                             #
###########################################################################

# Shortcuts to the simple front-end kernels (1/2/3-D Type-3)
_KER = {1: cufinufft.nufft1d3, 2: cufinufft.nufft2d3, 3: cufinufft.nufft3d3}

def _resolve_weights(weights: Optional[np.ndarray], c: Optional[np.ndarray]) -> np.ndarray:
    """Return an array to use as weights, giving priority to *weights*.

    Raises
    ------
    ValueError
        If neither *weights* nor *c* are provided.
    """
    if weights is not None:
        return weights
    if c is not None:
        return c
    raise ValueError("You must supply either 'weights' or its alias 'c'.")


# ---------------------------------------------------------------------
# Forward transform (real → reciprocal)
# ---------------------------------------------------------------------

def execute_cunufft(
    real_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    q_coords: np.ndarray | None = None,
    *,
    # legacy alias
    c: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    mem_frac: float = 0.5,
    min_chunk: int = 32_000,
    max_chunk: Optional[int] = 64_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Forward Type‑3 NUFFT.

    Parameters
    ----------
    real_coords, q_coords : (M, 3) and (N, 3) float64 arrays
        Real‑space and reciprocal‑space point clouds.
    weights, c : (M,) complex128
        Amplitudes attached to *real_coords*.  *c* is kept for backward‑
        compatibility and overrides *weights* if both are given.
    """
    weights = _resolve_weights(weights, c)
    if q_coords is None:
        raise ValueError("q_coords must be supplied for the forward transform")
    return _batched_type3(
        real_coords, weights, q_coords,
        eps=eps, inverse=False,
        mem_frac=mem_frac, min_chunk=min_chunk, max_chunk=max_chunk,
        prefer_cpu=prefer_cpu, gpu_only=gpu_only,
    )


# ---------------------------------------------------------------------
# Inverse transform (reciprocal → real)
# ---------------------------------------------------------------------

def execute_inverse_cunufft(
    q_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    real_coords: np.ndarray | None = None,
    *,
    # legacy alias
    c: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    mem_frac: float = 0.25,
    min_chunk: int = 128_000,
    max_chunk: Optional[int] = 256_000,
    prefer_cpu: bool = False,
    gpu_only: bool = False,
) -> np.ndarray:
    """Inverse Type‑3 NUFFT with *c* alias for legacy code."""
    weights = _resolve_weights(weights, c)
    if real_coords is None:
        raise ValueError("real_coords must be supplied for the inverse transform")
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
    """
    Stream-aware cuFINUFFT / FINUFFT Type-3 driver.

    If the GPU plan fails (OOM, launch failure, watchdog reset, etc.)
    the current chunk—and all subsequent ones—are computed on CPU
    via streamed FINUFFT so the overall job keeps running.
    """
    # ------------------------------------------------------------------ #
    #  Initial sanity checks / GPU pinning                               #
    # ------------------------------------------------------------------ #
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError("Only 1-, 2- and 3-D inputs supported")

    # Try to keep the usually smaller *source* set resident on GPU
    d_real = _as_device(real_coords, allow_fail=True)
    if d_real is None or prefer_cpu:
        return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

    cols = [_contig(d_real[:, i]) for i in range(dim)]

    # ------------------------------------------------------------------ #
    #  Streaming loop over the *other* (typically larger) point cloud    #
    # ------------------------------------------------------------------ #
    bytes_per_pt = dim * 8 + 16 + 32          # coord + weight + scratch
    out = np.zeros(
        len(real_coords) if inverse else len(q_coords),
        dtype=np.complex128,
    )

    start = 0
    while start < len(q_coords):
        # --- determine chunk size --------------------------------------
        chunk = max(_auto_chunk_size(mem_frac, bytes_per_pt), min_chunk)
        if max_chunk is not None:
            chunk = min(chunk, max_chunk)
        end = min(start + chunk, len(q_coords))

        # --- prepare reciprocal slice ----------------------------------
        q_slice = q_coords[start:end]
        if inverse:
            q_slice = -q_slice

        # ------------------------------------------------------------------
        #  GPU phase (allocation → kernel launch → gather → cleanup)         #
        # ------------------------------------------------------------------
        try:
            d_q = _as_device(q_slice)                               # H2D
            d_w = _as_device(weights[start:end] if inverse else weights)
            q_cols = [_contig(d_q[:, i]) for i in range(dim)]

            d_res = _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse)

            # gather results back to host
            if inverse:
                out += cp.asnumpy(d_res)
            else:
                out[start:end] = cp.asnumpy(d_res)

        # --------- anything here triggers CPU fallback --------------------
        except (cp.cuda.memory.OutOfMemoryError,        # alloc fail
                cp.cuda.runtime.CUDARuntimeError,       # launch/watchdog
                RuntimeError,                           # cuFINUFFT error
                OSError) as err:                        # Windows C++ throw
            if gpu_only:
                raise RuntimeError("GPU execution forced but failed") from err
            return _cpu_fallback(real_coords, weights, q_coords, eps, inverse)

        # --------- always run: free GPU resources, even after crash -------
        finally:
            # locals might not exist if allocation failed
            for name in ("d_q", "q_cols", "d_res", "d_w"):
                if name in locals():
                    del locals()[name]

            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
            except cp.cuda.runtime.CUDARuntimeError:
                # device lost/reset – fine, we will fall back on next loop
                pass

        # next slice
        start = end

    return out


###########################################################################
#  GPU kernel chooser                                                     #
###########################################################################


def _adaptive_gpu_launch(dim, cols, d_w, q_cols, eps, inverse):
    # ---------- plan‑grid sanity check ---------------------------------
    need = _estimate_grid_bytes(
        np.column_stack([c.get() for c in cols]),
        np.column_stack([c.get() for c in q_cols]),
    )
    if need > _free_mem_bytes() * 0.50:        # keep 20 % head room
        raise RuntimeError("fine grid exceeds available GPU memory")
    # -------------------------------------------------------------------

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
        except (RuntimeError,
                 OSError,
                 cp.cuda.runtime.CUDARuntimeError) as e:
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
def _cpu_fallback(
    real_coords: np.ndarray,
    weights:    np.ndarray,
    q_coords:   np.ndarray,
    eps: float,
    inverse: bool,
    *,
    batch: int = 5_000_000,
) -> np.ndarray:
    """
    Streamed FINUFFT fallback (1-, 2-, or 3-D type-3).

    • Keeps memory bounded by processing *q_coords* in chunks of size **batch**.
    • Works for both forward (real → reciprocal) and inverse (reciprocal → real)
      type-3 transforms.
    """
    dim = real_coords.shape[1]
    if dim not in (1, 2, 3):
        raise ValueError(f"Unsupported dimensionality: {dim}")

    # ── lazily import & cache FINUFFT wrappers ────────────────────────────
    global _FINUFFT3
    if not isinstance(_FINUFFT3, dict):
        _FINUFFT3 = {}
    if dim not in _FINUFFT3:
        import finufft
        _FINUFFT3[dim] = {1: finufft.nufft1d3,
                          2: finufft.nufft2d3,
                          3: finufft.nufft3d3}[dim]
    nufft = _FINUFFT3[dim]

    isign = -1 if inverse else +1

    # ── split coordinates per dimension (x, y, z) ────────────────────────
    real_split  = [real_coords[:, i].astype(np.float64) for i in range(dim)]
    recip_split = [q_coords[:,  i].astype(np.float64) for i in range(dim)]

    # ── output buffer ─────────────────────────────────────────────────────
    out = np.zeros(
        len(real_coords) if inverse else len(q_coords),
        dtype=np.complex128,
    )

    # ── streamed execution: loop over reciprocal-space chunks ─────────────
    start = 0
    while start < len(q_coords):
        end = min(start + batch, len(q_coords))

        recip_chunk = [a[start:end] for a in recip_split]
        if inverse:
            # FINUFFT uses opposite sign convention for inverse type-3
            recip_chunk = [-a for a in recip_chunk]
            c_chunk     = weights[start:end]            # sources’ amplitudes

            #   FINUFFT signature: (x, y, z, c,   s, t, u)
            #   Here: sources = reciprocal chunk, targets = real grid
            args = (*recip_chunk, c_chunk, *real_split)
        else:
            c_chunk = weights                           # sources’ amplitudes

            #   Forward transform: sources = real grid, targets = reciprocal chunk
            args = (*real_split, c_chunk, *recip_chunk)

        res = nufft(*args, eps=eps, isign=isign)

        if inverse:
            out += res                    # accumulate over q-chunks
        else:
            out[start:end] = res          # scatter into matching slice

        start = end

    return out


###########################################################################
#  Mute noisy destructor warnings (harmless but scary)                     #
###########################################################################

warnings.filterwarnings("ignore", message=r"Error destroying plan.")