"""P11 predicted numerical-agreement tolerance (core/storage/agreement.py).

The agreement tolerance is the computation's PREDICTED forward-error bound
``rtol = S*(eps + M*u)*kappa`` -- tied to the NUFFT ``eps``, the reciprocal-point
summation depth ``M``, and the per-channel cancellation factor ``kappa`` -- NOT a
magic constant. These tests pin the model's documented behaviour so the scientist
can confirm/replace the engineering parameters from one place.
"""

from __future__ import annotations

import numpy as np

from core.storage.agreement import (
    AGREEMENT_SAFETY_FACTOR,
    FLOAT64_UNIT_ROUNDOFF,
    KAPPA_DELTA_MAX,
    cancellation_kappa,
    predict_agreement_rtol,
    relative_l2,
)


def test_rtol_is_eps_driven_at_modest_summation_depth():
    # Small M: the eps term dominates; rtol ~ S*eps (matches observed GPU-relaunch
    # floor ~1e-12 with margin).
    rtol = predict_agreement_rtol(eps=1e-12, summation_terms=100, kappa=1.0)
    assert np.isclose(rtol, AGREEMENT_SAFETY_FACTOR * (1e-12 + 100 * FLOAT64_UNIT_ROUNDOFF))
    assert 1e-12 < rtol < 1e-10


def test_rtol_grows_with_summation_depth():
    # "Bigger summation arrays": the M*u term takes over for large M, so the tolerance
    # scales UP with the problem instead of being pinned at one scale.
    small = predict_agreement_rtol(eps=1e-12, summation_terms=100, kappa=1.0)
    big = predict_agreement_rtol(eps=1e-12, summation_terms=1_000_000, kappa=1.0)
    assert big > small
    # At M=1e6, M*u ~ 2.2e-10 dominates eps -> rtol lands in the ~1e-9 regime.
    assert 1e-9 < big < 1e-8


def test_rtol_scales_linearly_with_kappa():
    base = predict_agreement_rtol(eps=1e-12, summation_terms=10, kappa=1.0)
    amplified = predict_agreement_rtol(eps=1e-12, summation_terms=10, kappa=50.0)
    assert np.isclose(amplified, 50.0 * base)


def test_cancellation_kappa_amplifies_small_delta():
    # delta is a 1% residual of average -> kappa ~ 100 (looser delta tolerance).
    average = np.array([100.0 + 0.0j])
    delta = np.array([1.0 + 0.0j])
    assert np.isclose(cancellation_kappa(average, delta), 100.0)


def test_cancellation_kappa_never_below_one():
    # A delta larger than the average must NOT tighten below the well-conditioned
    # average channel: kappa floors at 1.
    average = np.array([1.0 + 0.0j])
    delta = np.array([10.0 + 0.0j])
    assert cancellation_kappa(average, delta) == 1.0


def test_cancellation_kappa_clamps_zero_delta_to_cap():
    # Zero displacement (delta -> 0) must not make the gate vacuous: kappa is capped.
    average = np.array([1.0 + 0.0j])
    delta = np.array([0.0 + 0.0j])
    assert cancellation_kappa(average, delta) == KAPPA_DELTA_MAX


def test_relative_l2_shape_mismatch_is_infinite():
    assert relative_l2(np.array([1.0, 2.0]), np.array([1.0])) == float("inf")


def test_relative_l2_basic_ratio():
    ref = np.array([2.0 + 0.0j])
    other = np.array([2.0 + 1e-12j])
    assert np.isclose(relative_l2(ref, other), 1e-12 / 2.0)
