"""Scientific-invariant validation for scattering partial results.

This module is the **Phase A (additive, safe)** deliverable of the P11
device-agnostic durable-identity design
(`implementation_stages/p11_device_agnostic_identity.md`).

The motivation is that exact output-byte identity cannot be used to decide
whether a scattering result is correct: CPU and GPU NUFFT/FFT pipelines agree
only to rel-L2 ~5.7e-14 and GPU spread reductions are not even self-reproducible.
P11 therefore replaces byte/tolerance twin comparison with *scientific-invariant
validation*: a result may be promoted to a durable checkpoint only if it passes a
documented set of invariants. Structural/integer fields are validated by **exact**
equality (no tolerance); only the **float amplitudes** are validated by bounded
invariants.

This module is intentionally pure and side-effect free, and is **not** wired into
any commit path here (that is P11 Phase C). Each check returns a structured
``InvariantFinding`` so callers can log, aggregate, or fail-closed as they see fit.

Scope of authorship (scientist directive)
------------------------------------------
Half-space conjugate-symmetry is the crown jewel of the method; its correctness
is **not** authored here. This module implements ONLY mechanically-objective,
non-controversial invariants as **hard gates**:

* finiteness (no NaN/Inf),
* shape / dtype (``complex128``), and
* EXACT point coverage (``point_ids`` == expected set).

The amplitude-norm ratio, half-space symmetry, and imag/real-ratio checks are
**non-gating advisory** (:func:`validate_norm_sanity`,
:func:`validate_half_space_symmetry`, :func:`validate_imag_real_ratio`) -- the
norm because "too large but finite" is a scientific judgment, the half-space
checks per scientist directive.
They never hard-pass/fail and never call reconstruction math. Half-space
correctness remains enforced by the frozen P0.0 baseline regression
(``tests/unit/scattering/test_synthetic_half_space_imag_regression.py``) and the
``core/scattering/half_space.py`` reconstruction contract; the symmetry invariant
itself is to be defined by the scientist.

Threshold provenance
--------------------
* ``norm_max_ratio`` default ``1e6`` is an *engineering* sanity bound on
  ``‖amplitudes_delta‖ / ‖amplitudes_average‖`` chosen to catch exploded /
  uninitialised amplitudes (NaN-free but absurd magnitude) without rejecting any
  physically plausible delta-vs-average ratio. It is **not** anchored to a frozen
  scientific baseline -- see the flagged note in the module-level
  ``__doc__`` of :func:`validate_norm_sanity`. Callers may tighten it once a
  representative baseline exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.scattering.contracts import ScatteringPartialResult


# --------------------------------------------------------------------------- #
# Documented thresholds and their provenance.
# --------------------------------------------------------------------------- #

#: Engineering sanity bound on ``‖amplitudes_delta‖ / ‖amplitudes_average‖``.
#: Not anchored to a frozen scientific baseline; chosen to flag exploded /
#: uninitialised amplitudes. See :func:`validate_norm_sanity`.
DEFAULT_NORM_MAX_RATIO: float = 1e6

#: Expected amplitude dtype for committed scattering results (P11 records the
#: realized dtype as metadata, but the in-memory contract is complex128).
EXPECTED_AMPLITUDE_DTYPE = np.dtype(np.complex128)


@dataclass(frozen=True)
class InvariantFinding:
    """Outcome of a single invariant check.

    Attributes
    ----------
    name:
        Stable identifier of the invariant (e.g. ``"finite"``).
    ok:
        ``True`` when the invariant holds.
    detail:
        Human-readable explanation. For failures this should be specific enough
        to debug from logs alone (which field, observed vs expected).
    """

    name: str
    ok: bool
    detail: str


def _stacked_amplitudes(result: ScatteringPartialResult) -> np.ndarray:
    """Return delta and average amplitudes stacked for joint inspection."""

    return np.concatenate(
        (
            np.asarray(result.amplitudes_delta).ravel(),
            np.asarray(result.amplitudes_average).ravel(),
        )
    )


def validate_finite(result: ScatteringPartialResult) -> InvariantFinding:
    """No NaN/Inf anywhere in the delta or average amplitudes.

    A non-finite value indicates a broken NUFFT/accumulation and must never be
    promoted. This is a float check but uses an exact predicate (``isfinite``),
    not a tolerance.
    """

    amplitudes = _stacked_amplitudes(result)
    finite_mask = np.isfinite(amplitudes)
    if bool(np.all(finite_mask)):
        return InvariantFinding(
            name="finite",
            ok=True,
            detail=f"all {amplitudes.size} amplitude components are finite",
        )
    non_finite = int(amplitudes.size - int(np.count_nonzero(finite_mask)))
    return InvariantFinding(
        name="finite",
        ok=False,
        detail=f"{non_finite} of {amplitudes.size} amplitude components are NaN/Inf",
    )


def validate_shape_and_dtype(
    result: ScatteringPartialResult,
    *,
    expected_dtype: np.dtype = EXPECTED_AMPLITUDE_DTYPE,
) -> InvariantFinding:
    """Amplitude arrays are 1D, aligned with ``point_ids``, and ``complex128``.

    Shape alignment is an **exact** structural check (the contract requires
    ``amplitudes_*.shape == point_ids.shape``). The dtype defaults to
    ``complex128``, the in-memory amplitude contract; P11 records the realized
    backend dtype separately as metadata.
    """

    point_shape = np.asarray(result.point_ids).shape
    expected = np.dtype(expected_dtype)
    problems: list[str] = []
    for field in ("amplitudes_delta", "amplitudes_average"):
        array = np.asarray(getattr(result, field))
        if array.shape != point_shape:
            problems.append(
                f"{field} shape {array.shape} != point_ids shape {point_shape}"
            )
        if array.dtype != expected:
            problems.append(f"{field} dtype {array.dtype} != expected {expected}")
    if problems:
        return InvariantFinding(
            name="shape_and_dtype", ok=False, detail="; ".join(problems)
        )
    return InvariantFinding(
        name="shape_and_dtype",
        ok=True,
        detail=f"amplitudes are {expected} aligned with point_ids shape {point_shape}",
    )


def validate_point_coverage(
    result: ScatteringPartialResult,
    *,
    expected_point_ids: Sequence[int] | np.ndarray,
) -> InvariantFinding:
    """``point_ids`` EXACTLY equal the expected partition set.

    Coverage is a structural identity check: P11 requires that a committed chunk
    covers exactly the partition's expected point ids -- no missing, no extra,
    no duplicates -- compared by **set equality** with **no tolerance**.
    Ordering is not required to match (amplitudes are aligned positionally to
    ``point_ids`` by the contract), but the *set* must be identical and the
    delivered ``point_ids`` must contain no duplicates.

    Role-awareness: the half-space role changes how many points a partition
    covers (e.g. ``positive_half`` vs ``zero_plane``/``full``). This check is
    role-aware by accepting the **caller-resolved** ``expected_point_ids`` for
    the active role and validating against it exactly. The mapping from
    half-space role to the expected point set is owned upstream (the partition /
    ``half_space.py`` contract), NOT authored here -- this function only asserts
    the exact set match.
    """

    observed = np.asarray(result.point_ids).ravel()
    expected = np.asarray(expected_point_ids).ravel()

    if observed.size != len(set(observed.tolist())):
        return InvariantFinding(
            name="point_coverage",
            ok=False,
            detail="point_ids contain duplicate entries",
        )

    observed_set = set(observed.tolist())
    expected_set = set(expected.tolist())
    if observed_set == expected_set:
        return InvariantFinding(
            name="point_coverage",
            ok=True,
            detail=f"point_ids exactly cover the expected partition ({expected.size} points)",
        )

    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    return InvariantFinding(
        name="point_coverage",
        ok=False,
        detail=(
            f"point coverage mismatch: missing={missing[:8]}"
            f"{'...' if len(missing) > 8 else ''} "
            f"extra={extra[:8]}{'...' if len(extra) > 8 else ''}"
        ),
    )


def validate_imag_real_ratio(
    result: ScatteringPartialResult,
    *,
    half_space_role: object | None = None,
) -> InvariantFinding:
    """ADVISORY STUB -- non-gating. Half-space imag/real ratio is NOT authored here.

    Per scientist directive, half-space reconstruction is the crown jewel of the
    method and its correctness is not derived in this module. Computing the
    imag/real ratio requires invoking the reconstruction contract; that math is
    intentionally **not** called here. This check is therefore a non-gating
    advisory stub: it ALWAYS returns ``ok=True`` and NEVER calls reconstruction
    math.

    Half-space correctness is enforced elsewhere:

    * the frozen P0.0 baseline regression
      (``tests/unit/scattering/test_synthetic_half_space_imag_regression.py``,
      threshold ``positive_half_imag_over_real`` in the
      ``synthetic_half_space_v1`` baseline), and
    * the ``core/scattering/half_space.py`` reconstruction contract.

    The imag/real-ratio invariant is to be defined by the scientist.

    ``half_space_role`` is accepted and ignored for call-signature stability.
    """

    return InvariantFinding(
        name="imag_real_ratio",
        ok=True,
        detail=(
            "ADVISORY -- not gated; half-space imag/real ratio is enforced by the "
            "frozen P0.0 baseline regression and the half_space.py reconstruction "
            "contract. Imag/real invariant to be defined by the scientist."
        ),
    )


def validate_half_space_symmetry(
    result: ScatteringPartialResult,
    *,
    half_space_role: object | None = None,
) -> InvariantFinding:
    """ADVISORY STUB -- non-gating. Half-space symmetry is NOT authored here.

    Per scientist directive, half-space conjugate-symmetry reconstruction is the
    crown jewel of the method; its correctness must be defined by the scientist,
    not derived from a cold start. This check ALWAYS returns ``ok=True`` and
    NEVER calls reconstruction math. Half-space correctness is enforced by the
    frozen P0.0 baseline regression and the ``half_space.py`` reconstruction
    contract.

    ``half_space_role`` is accepted and ignored for call-signature stability.
    """

    return InvariantFinding(
        name="half_space_symmetry",
        ok=True,
        detail=(
            "ADVISORY -- not gated; half-space correctness is enforced by the "
            "frozen P0.0 baseline regression and the half_space.py reconstruction "
            "contract. Symmetry invariant to be defined by the scientist."
        ),
    )


def validate_norm_sanity(
    result: ScatteringPartialResult,
    *,
    norm_max_ratio: float = DEFAULT_NORM_MAX_RATIO,
) -> InvariantFinding:
    """ADVISORY (non-gating). Reports the delta/average amplitude-norm ratio.

    Whether a **finite** amplitude norm is "too large" is a scientific judgment,
    not a mechanically-objective fact, so per scientist directive this is NOT a
    hard gate: it ALWAYS returns ``ok=True`` and only reports the ratio (and
    whether it exceeds the loose engineering reference ``norm_max_ratio``) for
    logging / triage. Non-finite amplitudes are already rejected by the
    :func:`validate_finite` hard gate.

    .. note::
       ``norm_max_ratio`` (default ``1e6``) is an **engineering** reference, not a
       frozen scientific baseline. Flagged for the scientist to define a grounded
       gate if one is wanted.
    """

    delta = np.asarray(result.amplitudes_delta)
    average = np.asarray(result.amplitudes_average)
    delta_norm = float(np.linalg.norm(delta))
    average_norm = float(np.linalg.norm(average))

    if average_norm == 0.0:
        return InvariantFinding(
            name="norm_sanity",
            ok=True,
            detail=(
                f"ADVISORY -- average norm is 0 (no reference); "
                f"delta norm {delta_norm:.3e}"
            ),
        )

    ratio = delta_norm / average_norm
    within = bool(np.isfinite(ratio) and ratio <= norm_max_ratio)
    return InvariantFinding(
        name="norm_sanity",
        ok=True,
        detail=(
            f"ADVISORY -- delta/average norm ratio = {ratio:.3e} "
            f"({'within' if within else 'EXCEEDS'} engineering reference "
            f"{norm_max_ratio:.3e}); not gated"
        ),
    )


def validate_scattering_result(
    result: ScatteringPartialResult,
    *,
    expected_point_ids: Sequence[int] | np.ndarray,
    half_space_role: object | None = None,
    norm_max_ratio: float = DEFAULT_NORM_MAX_RATIO,
    expected_dtype: np.dtype = EXPECTED_AMPLITUDE_DTYPE,
) -> list[InvariantFinding]:
    """Run the P11 invariant set over a scattering partial result.

    Returns one :class:`InvariantFinding` per check, in a stable order, so
    callers can fail-closed on the HARD GATES and log every outcome. This
    function performs **no** side effects and **no** commit wiring (P11 Phase C).

    Hard gates (mechanically objective; ``ok=False`` means reject):

    * :func:`validate_finite`
    * :func:`validate_shape_and_dtype`
    * :func:`validate_point_coverage`  (exact set equality; role-aware via the
      caller-resolved ``expected_point_ids``)

    Advisory, NON-GATING (always ``ok=True``):

    * :func:`validate_norm_sanity`     (delta/average norm ratio; "too large but
      finite" is a scientific judgment, not a mechanical fact)
    * :func:`validate_imag_real_ratio`
    * :func:`validate_half_space_symmetry`  (half-space correctness is the
      scientist's to author, enforced separately by the frozen P0.0 baseline
      regression and the ``half_space.py`` reconstruction contract)

    ``is_valid`` (``all(f.ok)``) is decided entirely by the three hard gates.

    Parameters
    ----------
    expected_point_ids:
        The partition's expected point-id set for the active half-space role
        (resolved by the caller; exact, no tolerance).
    half_space_role:
        Passed through to the advisory stubs and ignored by them. Accepted for
        call-signature stability; the caller already encodes the role's effect on
        coverage via ``expected_point_ids``.
    norm_max_ratio:
        Engineering bound for :func:`validate_norm_sanity` (not baseline-frozen).
    """

    return [
        validate_finite(result),
        validate_shape_and_dtype(result, expected_dtype=expected_dtype),
        validate_point_coverage(result, expected_point_ids=expected_point_ids),
        validate_norm_sanity(result, norm_max_ratio=norm_max_ratio),
        validate_imag_real_ratio(result, half_space_role=half_space_role),
        validate_half_space_symmetry(result, half_space_role=half_space_role),
    ]


def is_valid(
    result: ScatteringPartialResult,
    *,
    expected_point_ids: Sequence[int] | np.ndarray,
    half_space_role: object | None = None,
    norm_max_ratio: float = DEFAULT_NORM_MAX_RATIO,
    expected_dtype: np.dtype = EXPECTED_AMPLITUDE_DTYPE,
) -> bool:
    """Convenience predicate: ``True`` iff every HARD GATE holds.

    The advisory findings (norm ratio, half-space symmetry, imag/real ratio) are
    always ``ok=True`` and never affect this result; validity is decided solely by
    the three hard gates: finiteness, shape/dtype, and exact point coverage.
    """

    findings = validate_scattering_result(
        result,
        expected_point_ids=expected_point_ids,
        half_space_role=half_space_role,
        norm_max_ratio=norm_max_ratio,
        expected_dtype=expected_dtype,
    )
    return all(finding.ok for finding in findings)


__all__ = [
    "DEFAULT_NORM_MAX_RATIO",
    "EXPECTED_AMPLITUDE_DTYPE",
    "InvariantFinding",
    "is_valid",
    "validate_finite",
    "validate_half_space_symmetry",
    "validate_imag_real_ratio",
    "validate_norm_sanity",
    "validate_point_coverage",
    "validate_scattering_result",
    "validate_shape_and_dtype",
]
