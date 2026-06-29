"""Predicted numerical-agreement tolerance for same-work commit reconciliation.

P11 reconciles same-work results (CPU vs GPU, or non-deterministic GPU relaunches)
by NUMERICAL agreement rather than bit equality. The agreement tolerance is NOT a
magic constant: two *correct* results can legitimately differ by up to ~2x the
**forward error** each carries, and that error is predictable from the computation:

    rtol_field = S * ( eps + M * u ) * kappa_field

* ``eps``  -- the NUFFT execution tolerance (finufft/cuFINUFFT deliver ~eps
  relative-L2). Two independent transforms each within ``eps`` of exact differ by
  <= ~2*eps. Dominant term at modest problem size.
* ``M * u`` -- the summation / GPU-atomicAdd term. ``u`` is the float64 unit
  roundoff (~2.22e-16); ``M`` is the summation depth (the number of reciprocal
  points reduced into each output amplitude). cuFINUFFT's atomicAdd spreading adds
  a non-deterministic error on top of the algorithm's ``eps`` guarantee that grows
  with how many terms collide in the reduction -- i.e. with problem size ("bigger
  summation arrays"). Using ``M`` as a conservative collision bound makes the
  tolerance scale with the problem instead of being pinned at one scale.
* ``S``    -- engineering safety factor (two-sided difference + summation-order /
  implementation variance + slack). FLAGGED for the scientist.
* ``kappa_field`` -- cancellation amplification of the field. The ``average``
  channel is well-conditioned (kappa ~ 1). The ``delta`` channel is a SUBTRACTION
  (``q_amp - q_amp_av``); when displacements are small ``||delta|| << ||average||``
  so ``kappa_delta = ||average|| / ||delta||`` can be large and the delta tolerance
  must loosen by exactly that factor. Clamped at ``KAPPA_DELTA_MAX`` so a near-zero
  delta cannot make the gate vacuous. FLAGGED for the scientist.

Sanity check (eps=1e-12): small M (~1e2) -> rtol ~ 8e-12 (matches the observed
GPU-relaunch floor ~1e-12 and CPU-vs-GPU ~6e-14 with margin); large M (~1e6) ->
M*u ~ 2e-10 dominates -> rtol ~ 1.6e-9 (the regime where a fixed 1e-9 became
marginal). The model reproduces reality and scales with the summation depth.

The thresholds here (``AGREEMENT_SAFETY_FACTOR``, ``KAPPA_DELTA_MAX``,
``DEFAULT_NUFFT_EPS``) are documented ENGINEERING values flagged for the scientist
to confirm or replace; this is not a claim of scientific equivalence.
"""

from __future__ import annotations

import numpy as np


#: float64 unit roundoff (machine epsilon), ~2.220446e-16.
FLOAT64_UNIT_ROUNDOFF: float = float(np.finfo(np.float64).eps)

#: Production NUFFT execution tolerance default; used when the realized ``eps`` is
#: not threaded to a commit path. FLAGGED -- should reflect the run's actual eps.
DEFAULT_NUFFT_EPS: float = 1e-12

#: Safety factor on the forward-error bound. FLAGGED for the scientist.
AGREEMENT_SAFETY_FACTOR: float = 8.0

#: Clamp on the delta-channel cancellation amplification so a near-zero delta
#: (zero displacement) cannot make the gate vacuous. FLAGGED for the scientist.
KAPPA_DELTA_MAX: float = 1.0e6


def relative_l2(reference: np.ndarray, other: np.ndarray) -> float:
    """Relative L2 distance ``||ref - other|| / ||ref||`` (inf on shape mismatch)."""
    ref = np.asarray(reference).reshape(-1)
    oth = np.asarray(other).reshape(-1)
    if ref.shape != oth.shape:
        return float("inf")
    denom = float(np.linalg.norm(ref))
    diff = float(np.linalg.norm(ref - oth))
    if denom == 0.0:
        return diff
    return diff / denom


def cancellation_kappa(
    average: np.ndarray,
    delta: np.ndarray,
    *,
    cap: float = KAPPA_DELTA_MAX,
) -> float:
    """Delta-channel cancellation amplification ``||average|| / ||delta||``.

    Clamped to ``[1.0, cap]``: never below 1 (delta is never better-conditioned
    than the well-conditioned average), never above ``cap`` (so a near-zero delta
    cannot drive the tolerance to infinity and make the gate vacuous).
    """
    avg_norm = float(np.linalg.norm(np.asarray(average).reshape(-1)))
    delta_norm = float(np.linalg.norm(np.asarray(delta).reshape(-1)))
    if delta_norm == 0.0:
        return float(cap)
    return float(min(max(1.0, avg_norm / delta_norm), cap))


def predict_agreement_rtol(
    *,
    eps: float,
    summation_terms: int,
    kappa: float = 1.0,
    safety: float = AGREEMENT_SAFETY_FACTOR,
    unit_roundoff: float = FLOAT64_UNIT_ROUNDOFF,
) -> float:
    """Predicted forward-error agreement tolerance ``S * (eps + M*u) * kappa``.

    ``summation_terms`` (``M``) is the summation depth (reciprocal points reduced
    into each output amplitude); it is clamped to ``>= 1``. ``kappa`` is the
    field's cancellation amplification (1 for the average channel).
    """
    m = float(max(1, int(summation_terms)))
    base = float(eps) + m * float(unit_roundoff)
    return float(safety) * base * float(max(1.0, kappa))


__all__ = [
    "AGREEMENT_SAFETY_FACTOR",
    "DEFAULT_NUFFT_EPS",
    "FLOAT64_UNIT_ROUNDOFF",
    "KAPPA_DELTA_MAX",
    "cancellation_kappa",
    "predict_agreement_rtol",
    "relative_l2",
]
