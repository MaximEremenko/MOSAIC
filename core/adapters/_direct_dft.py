# -*- coding: utf-8 -*-
"""
Direct (dense) type-3 DFT helper for the cuFINUFFT/FINUFFT adapter
==================================================================

This module holds the *numerical method* extracted from the adapter's
``_direct_cpu_fallback`` smoke-scale path: a hand-rolled, chunked, dense
type-3 discrete Fourier transform

    out[t] = sum_s exp(1j * isign * <target_t, source_s>) * coeffs[s]

It is deliberately a pure function of already-marshalled NumPy arrays so the
adapter (``cunufft_wrapper.py``) stays a thin marshalling layer. All dtype
coercion, contiguity, validation, the absent-``finufft`` guard, and the
validation-only warning remain on the adapter side.

This file intentionally imports nothing from ``core`` to avoid an import
cycle with ``core.scattering.kernels`` (which imports the adapter at module
top).
"""

from __future__ import annotations

import numpy as np

__all__ = ["direct_dft_type3"]


def direct_dft_type3(
    targets: np.ndarray,
    sources_t: np.ndarray,
    coeffs: np.ndarray,
    *,
    isign: int,
    batch: int,
) -> np.ndarray:
    """Dense, chunked type-3 DFT.

    Computes ``out[t] = sum_s exp(1j * isign * (targets @ sources.T))[t, s] * coeffs[s]``
    in row-chunks of ``targets`` of size ``batch``.

    Parameters
    ----------
    targets:
        ``(n_targets, dim)`` float64 array of target coordinates.
    sources_t:
        ``(dim, n_sources)`` C-contiguous float64 array; the transpose of the
        source coordinates.
    coeffs:
        ``(n_sources,)`` complex128 array of source weights.
    isign:
        Sign convention of the transform (``+1`` forward, ``-1`` inverse).
    batch:
        Number of target rows processed per chunk (must be >= 1).

    Returns
    -------
    np.ndarray
        ``(n_targets,)`` complex128 array of accumulated amplitudes.

    Notes
    -----
    The arithmetic mirrors the previous inline adapter code exactly
    (``np.exp(1j * isign * phase) @ coeffs``) so the output is byte-identical.
    The caller is responsible for all dtype/contiguity marshalling.
    """
    n_targets = int(targets.shape[0])
    out = np.zeros(n_targets, dtype=np.complex128)

    start = 0
    while start < n_targets:
        end = min(start + batch, n_targets)
        target_chunk = targets[start:end]
        phase = target_chunk @ sources_t
        out[start:end] = np.exp(1j * isign * phase) @ coeffs
        start = end

    return out
