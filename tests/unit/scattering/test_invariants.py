"""Unit tests for the P11 Phase A scattering-invariant validation module.

Hard gates (finiteness, shape/dtype, exact point coverage) each have a passing
case and a failing case. The amplitude-norm ratio, half-space imag/real-ratio,
and conjugate-symmetry checks are NON-GATING advisory (scientist directive): they
always return ``ok=True`` (the norm ratio because "too large but finite" is a
scientific judgment; the half-space checks never call reconstruction math), so
their tests assert exactly that, including that they never reject the frozen P0.0
baseline.

Some failure cases violate the ``ScatteringPartialResult`` construction contract
itself (e.g. a wrong-shape amplitude array, which the contract's
``__post_init__`` rejects). For those, a lightweight ``_FakeResult`` stand-in
exposes the same attributes the invariant functions read, so the invariant
behaviour can be tested in isolation from the construction-time contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.scattering.contracts import ScatteringPartialResult
from core.scattering.half_space import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
)
from core.scattering.invariants import (
    DEFAULT_NORM_MAX_RATIO,
    InvariantFinding,
    is_valid,
    validate_finite,
    validate_half_space_symmetry,
    validate_imag_real_ratio,
    validate_norm_sanity,
    validate_point_coverage,
    validate_scattering_result,
    validate_shape_and_dtype,
)


BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "baselines"
    / "synthetic_half_space_v1"
    / "baseline.json"
)


@dataclass
class _FakeResult:
    """Minimal stand-in exposing the attributes the invariants read.

    Lets us construct results that intentionally violate the
    ``ScatteringPartialResult`` construction contract (e.g. wrong shapes), which
    the real dataclass would reject at ``__post_init__``.
    """

    point_ids: np.ndarray
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray
    grid_shape_nd: np.ndarray = None  # type: ignore[assignment]
    reciprocal_point_count: int = 0


def _make_result(
    *,
    point_ids,
    amplitudes_delta,
    amplitudes_average=None,
    grid_shape_nd=(2, 2),
) -> ScatteringPartialResult:
    point_ids = np.asarray(point_ids)
    amplitudes_delta = np.asarray(amplitudes_delta, dtype=np.complex128)
    if amplitudes_average is None:
        amplitudes_average = np.ones(point_ids.shape, dtype=np.complex128)
    else:
        amplitudes_average = np.asarray(amplitudes_average, dtype=np.complex128)
    return ScatteringPartialResult(
        chunk_id=0,
        contributing_interval_ids=(1,),
        point_ids=point_ids,
        grid_shape_nd=np.asarray(grid_shape_nd),
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
        reciprocal_point_count=int(point_ids.size),
    )


# --------------------------------------------------------------------------- #
# Baseline acceptance (REQUIRED): the non-gating half-space stubs MUST NOT reject
# the frozen P0.0 synthetic_half_space_v1 baseline. The stubs always return
# ok=True; this test pins that they accept the real baseline amplitudes too.
# --------------------------------------------------------------------------- #


def _load_baseline_arrays():
    return np.load(BASELINE_PATH.parent / "baseline_arrays.npz")


def test_half_space_stubs_accept_frozen_baseline():
    arrays = _load_baseline_arrays()
    values = np.asarray(arrays["complex_values"], dtype=np.complex128).ravel()
    result = _make_result(
        point_ids=np.arange(values.size),
        amplitudes_delta=values,
        amplitudes_average=values,
        grid_shape_nd=(values.size,),
    )

    for role in (HALF_SPACE_ROLE_POSITIVE_HALF, HALF_SPACE_ROLE_ZERO_PLANE):
        assert validate_imag_real_ratio(result, half_space_role=role).ok
        assert validate_half_space_symmetry(result, half_space_role=role).ok


# --------------------------------------------------------------------------- #
# 1. validate_finite
# --------------------------------------------------------------------------- #


def test_validate_finite_pass():
    result = _make_result(
        point_ids=[0, 1, 2],
        amplitudes_delta=[1 + 1j, 2 + 0j, 0 - 3j],
    )
    finding = validate_finite(result)
    assert isinstance(finding, InvariantFinding)
    assert finding.ok


def test_validate_finite_fail_on_nan():
    result = _make_result(
        point_ids=[0, 1, 2],
        amplitudes_delta=[1 + 1j, np.nan + 0j, 0 - 3j],
    )
    finding = validate_finite(result)
    assert not finding.ok
    assert "NaN/Inf" in finding.detail


def test_validate_finite_fail_on_inf_in_average():
    result = _make_result(
        point_ids=[0, 1],
        amplitudes_delta=[1 + 1j, 2 + 0j],
        amplitudes_average=[1 + 0j, np.inf + 0j],
    )
    assert not validate_finite(result).ok


# --------------------------------------------------------------------------- #
# 2. validate_shape_and_dtype
# --------------------------------------------------------------------------- #


def test_validate_shape_and_dtype_pass():
    result = _make_result(
        point_ids=[0, 1, 2],
        amplitudes_delta=[1 + 1j, 2 + 0j, 0 - 3j],
    )
    assert validate_shape_and_dtype(result).ok


def test_validate_shape_and_dtype_fail_wrong_shape():
    # Bypass the construction contract to inject a shape mismatch.
    fake = _FakeResult(
        point_ids=np.array([0, 1, 2]),
        amplitudes_delta=np.ones(2, dtype=np.complex128),
        amplitudes_average=np.ones(3, dtype=np.complex128),
    )
    finding = validate_shape_and_dtype(fake)
    assert not finding.ok
    assert "shape" in finding.detail


def test_validate_shape_and_dtype_fail_wrong_dtype():
    fake = _FakeResult(
        point_ids=np.array([0, 1, 2]),
        amplitudes_delta=np.ones(3, dtype=np.complex64),
        amplitudes_average=np.ones(3, dtype=np.complex128),
    )
    finding = validate_shape_and_dtype(fake)
    assert not finding.ok
    assert "dtype" in finding.detail


# --------------------------------------------------------------------------- #
# 3. validate_point_coverage (exact, no tolerance)
# --------------------------------------------------------------------------- #


def test_validate_point_coverage_pass_exact():
    result = _make_result(
        point_ids=[10, 11, 12],
        amplitudes_delta=[1 + 0j, 1 + 0j, 1 + 0j],
    )
    assert validate_point_coverage(result, expected_point_ids=[10, 11, 12]).ok


def test_validate_point_coverage_pass_reordered():
    # Set equality, ordering not required.
    result = _make_result(
        point_ids=[12, 10, 11],
        amplitudes_delta=[1 + 0j, 1 + 0j, 1 + 0j],
    )
    assert validate_point_coverage(result, expected_point_ids=[10, 11, 12]).ok


def test_validate_point_coverage_fail_missing_point():
    result = _make_result(
        point_ids=[10, 11],
        amplitudes_delta=[1 + 0j, 1 + 0j],
    )
    finding = validate_point_coverage(result, expected_point_ids=[10, 11, 12])
    assert not finding.ok
    assert "missing" in finding.detail


def test_validate_point_coverage_fail_extra_point():
    result = _make_result(
        point_ids=[10, 11, 12, 13],
        amplitudes_delta=[1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
    )
    finding = validate_point_coverage(result, expected_point_ids=[10, 11, 12])
    assert not finding.ok
    assert "extra" in finding.detail


def test_validate_point_coverage_fail_duplicate_point():
    fake = _FakeResult(
        point_ids=np.array([10, 11, 11]),
        amplitudes_delta=np.ones(3, dtype=np.complex128),
        amplitudes_average=np.ones(3, dtype=np.complex128),
    )
    finding = validate_point_coverage(fake, expected_point_ids=[10, 11])
    assert not finding.ok
    assert "duplicate" in finding.detail


# --------------------------------------------------------------------------- #
# 4. validate_imag_real_ratio -- NON-GATING ADVISORY STUB
# --------------------------------------------------------------------------- #
# Per scientist directive, this is not authored here: it ALWAYS returns ok=True
# and NEVER calls reconstruction math. Tests pin exactly that.


def test_validate_imag_real_ratio_is_nongating_advisory_stub():
    # Even a wildly non-real field must NOT be rejected (stub is advisory only).
    values = np.array([1 + 9j, 3 - 7j, 0.5 + 4j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1, 2], amplitudes_delta=values)
    for role in (HALF_SPACE_ROLE_POSITIVE_HALF, HALF_SPACE_ROLE_ZERO_PLANE, None):
        finding = validate_imag_real_ratio(result, half_space_role=role)
        assert isinstance(finding, InvariantFinding)
        assert finding.name == "imag_real_ratio"
        assert finding.ok
        assert "ADVISORY" in finding.detail


def test_validate_imag_real_ratio_does_not_call_reconstruction(monkeypatch):
    # Guard: the stub must never invoke any half_space reconstruction. Booby-trap
    # the reconstruction symbol in case a future edit re-introduces a call.
    import core.scattering.invariants as inv

    def _explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("stub must not call reconstruction math")

    monkeypatch.setattr(
        inv, "apply_half_space_conjugate_reconstruction", _explode, raising=False
    )
    monkeypatch.setattr(
        inv, "half_space_conjugate_reconstruction_required", _explode, raising=False
    )
    values = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1], amplitudes_delta=values)
    assert inv.validate_imag_real_ratio(
        result, half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF
    ).ok


# --------------------------------------------------------------------------- #
# 5. validate_half_space_symmetry -- NON-GATING ADVISORY STUB
# --------------------------------------------------------------------------- #


def test_validate_half_space_symmetry_is_nongating_advisory_stub():
    # Crown-jewel directive: symmetry is the scientist's to author. Stub never
    # hard-pass/fail-gates and never calls reconstruction math.
    values = np.array([1 + 9j, 3 - 7j, 0.5 + 4j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1, 2], amplitudes_delta=values)
    for role in (HALF_SPACE_ROLE_POSITIVE_HALF, HALF_SPACE_ROLE_ZERO_PLANE, None):
        finding = validate_half_space_symmetry(result, half_space_role=role)
        assert isinstance(finding, InvariantFinding)
        assert finding.name == "half_space_symmetry"
        assert finding.ok
        assert "ADVISORY" in finding.detail
        assert "scientist" in finding.detail


def test_validate_half_space_symmetry_does_not_call_reconstruction(monkeypatch):
    import core.scattering.invariants as inv

    def _explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("stub must not call reconstruction math")

    monkeypatch.setattr(
        inv, "apply_half_space_conjugate_reconstruction", _explode, raising=False
    )
    monkeypatch.setattr(
        inv, "half_space_conjugate_reconstruction_required", _explode, raising=False
    )
    values = np.array([2 + 1j, 6 + 1j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1], amplitudes_delta=values)
    assert inv.validate_half_space_symmetry(
        result, half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF
    ).ok


# --------------------------------------------------------------------------- #
# 6. validate_norm_sanity
# --------------------------------------------------------------------------- #


def test_validate_norm_sanity_pass():
    result = _make_result(
        point_ids=[0, 1, 2],
        amplitudes_delta=[1 + 0j, 1 + 0j, 1 + 0j],
        amplitudes_average=[1 + 0j, 1 + 0j, 1 + 0j],
    )
    assert validate_norm_sanity(result).ok


def test_validate_norm_sanity_is_advisory_even_for_exploded_norm():
    # norm ratio is non-gating advisory: a huge but FINITE ratio is reported,
    # not rejected (the scientist owns any grounded bound).
    result = _make_result(
        point_ids=[0, 1, 2],
        amplitudes_delta=[1e30 + 0j, 1e30 + 0j, 1e30 + 0j],
        amplitudes_average=[1 + 0j, 1 + 0j, 1 + 0j],
    )
    finding = validate_norm_sanity(result)
    assert finding.ok
    assert "ADVISORY" in finding.detail
    assert "EXCEEDS" in finding.detail


def test_validate_norm_sanity_pass_zero_average_reference():
    result = _make_result(
        point_ids=[0, 1],
        amplitudes_delta=[1 + 0j, 2 + 0j],
        amplitudes_average=[0 + 0j, 0 + 0j],
    )
    finding = validate_norm_sanity(result)
    assert finding.ok
    assert "no reference" in finding.detail


# --------------------------------------------------------------------------- #
# Aggregate gate: validate_scattering_result + is_valid
# --------------------------------------------------------------------------- #


def test_validate_scattering_result_all_pass():
    values = np.array([1 + 2j, 3 - 1j, 0.5 + 0.25j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1, 2], amplitudes_delta=values)
    findings = validate_scattering_result(
        result,
        expected_point_ids=[0, 1, 2],
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )
    assert len(findings) == 6
    assert all(f.ok for f in findings)
    # The two half-space findings are non-gating advisory stubs.
    by_name = {f.name: f for f in findings}
    assert "ADVISORY" in by_name["imag_real_ratio"].detail
    assert "ADVISORY" in by_name["half_space_symmetry"].detail
    assert is_valid(
        result,
        expected_point_ids=[0, 1, 2],
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )


def test_validity_is_decided_only_by_hard_gates():
    # A non-real positive_half field still validates: the half-space stubs are
    # advisory and never gate. Validity is decided by the 4 hard gates only.
    values = np.array([1 + 5j, 3 - 8j, 0.5 + 2j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1, 2], amplitudes_delta=values)
    assert is_valid(
        result,
        expected_point_ids=[0, 1, 2],
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )


def test_validate_scattering_result_role_optional():
    # half_space_role is optional (accepted and ignored by the advisory stubs).
    values = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1], amplitudes_delta=values)
    assert is_valid(result, expected_point_ids=[0, 1])


def test_validate_scattering_result_fail_on_coverage():
    values = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1], amplitudes_delta=values)
    findings = validate_scattering_result(
        result,
        expected_point_ids=[0, 1, 2],
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )
    by_name = {f.name: f for f in findings}
    assert not by_name["point_coverage"].ok
    assert not is_valid(
        result,
        expected_point_ids=[0, 1, 2],
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )


def test_validate_scattering_result_fail_on_nonfinite():
    values = np.array([np.nan + 0j, 1 + 0j], dtype=np.complex128)
    result = _make_result(point_ids=[0, 1], amplitudes_delta=values)
    assert not is_valid(
        result,
        expected_point_ids=[0, 1],
        half_space_role=HALF_SPACE_ROLE_ZERO_PLANE,
    )
