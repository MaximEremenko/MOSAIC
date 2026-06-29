"""Cross-environment reduce-order determinism for scattering partials (plan_2 Phase 2, G1).

The durable model promotes a chunk only on an EXACT ``payload_sha256`` match and is
fail-closed: a chunk that hashes differently across worker count / partition layout
silently fails to promote. That premise requires the scattering reduce
(``merge_scattering_partial_results``) to produce a *bit-identical* merged result
regardless of the order in which partials are combined or how they are grouped
(which simulates different worker / partition counts).

This module asserts that property. The reduce is a tree of ``np.add`` on
``complex128`` amplitude arrays (see ``merge_scattering_partial_results`` in
``core/scattering/contracts.py``), and IEEE-754 floating-point addition is **not
associative**, so different reduce orders can produce results that differ in the
last bits. When that happens the canonical payload bytes differ and so does
``payload_sha256``.

Empirically (numpy 2.4.4, this repo) the hashes DO diverge across reduce order and
grouping, so ``test_reduce_order_invariant_payload_sha256`` is marked
``xfail(strict=True)``: if the underlying reduce ever becomes order-stable (e.g. a
fixed canonical reduce order, pairwise/Kahan summation, or fixed-point
accumulation), this test will XPASS and flag that the determinism contract — and
the exact-hash promotion model — can be tightened. The divergence is characterized
in ``tests/unit/scattering/REDUCE_DETERMINISM_FINDINGS.md``.

The two supporting tests document the *mechanism* (so the xfail above is understood
as a real numeric finding, not a flaky harness): one shows the hash is a pure
function of the reduced bytes, and one shows that ``np.add`` order non-associativity
is the root cause.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest

from core.scattering.accumulation import (
    build_scattering_partial_result,
    merge_scattering_partial_results,
)
from core.scattering.contracts import ScatteringPartialResult
from core.storage.fingerprint import payload_sha256


# Schema/expected_set_digest values are fixed across every reduction so that the
# only thing that can change the hash is the reduced amplitude *bytes*. This
# mirrors how core/scattering/commit.py builds the commit-candidate payload.
_PAYLOAD_SCHEMA = "mosaic.scattering.commit_candidate"
_EXPECTED_SET_DIGEST = "reduce-determinism-fixture"
_PAYLOAD_ATTRS = {"chunk_id": 0, "stage": "scattering"}


def _build_partials(
    *,
    n_intervals: int,
    n_points: int,
    seed: int,
) -> list[ScatteringPartialResult]:
    """N additive partials for one chunk with fixed pseudo-random complex data.

    The contributions are deliberately spread across several orders of magnitude
    (the ``10 ** (i - 1)`` scale factor) so that catastrophic cancellation /
    rounding differences between reduce orders are *exposed* rather than hidden by
    similarly-sized operands.
    """
    rng = np.random.default_rng(seed)
    grid_shape_nd = np.array([n_points], dtype=np.int64)
    point_ids = np.arange(n_points, dtype=np.int64)

    partials: list[ScatteringPartialResult] = []
    for interval_id in range(n_intervals):
        scale = 10.0 ** (interval_id - 1)
        delta = (
            rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)
        ) * scale
        average = (
            rng.standard_normal(n_points) + 1j * rng.standard_normal(n_points)
        ) * scale
        partials.append(
            build_scattering_partial_result(
                chunk_id=0,
                interval_id=interval_id,
                amplitudes_delta=delta,
                amplitudes_average=average,
                grid_shape_nd=grid_shape_nd,
                reciprocal_point_count=n_points,
                point_ids=point_ids,
            )
        )
    return partials


def _reduce_sequential(
    partials: list[ScatteringPartialResult],
) -> ScatteringPartialResult:
    """Left-fold the partials in the given order (single-worker reduce)."""
    return functools.reduce(merge_scattering_partial_results, partials)


def _reduce_grouped(
    partials: list[ScatteringPartialResult],
    *,
    group_size: int,
) -> ScatteringPartialResult:
    """Reduce per group, then reduce the per-group results.

    ``group_size`` simulates a different worker / partition count: each group is a
    worker that locally reduces its slice, then a final reduce combines the
    per-worker partial sums.
    """
    groups = [
        partials[start : start + group_size]
        for start in range(0, len(partials), group_size)
    ]
    group_results = [_reduce_sequential(group) for group in groups]
    return _reduce_sequential(group_results)


def _payload_sha256_of(result: ScatteringPartialResult) -> str:
    """Materialize the canonical commit payload and hash it.

    This mirrors ``core/scattering/commit.py``'s ``_payload_datasets`` +
    ``_payload_digest``: amplitudes are cast to ``complex128``, point ids /
    grid shape to ``int64``, then fed to ``payload_sha256``. Keeping this in lockstep
    with the durable model is what makes the assertion meaningful.
    """
    delta = np.asarray(result.amplitudes_delta, dtype=np.complex128).reshape(-1)
    average = np.asarray(result.amplitudes_average, dtype=np.complex128).reshape(-1)
    point_ids = np.asarray(result.point_ids, dtype=np.int64).reshape(-1)
    datasets = {
        "point_ids": point_ids,
        "grid_shape_nd": np.asarray(result.grid_shape_nd, dtype=np.int64),
        "amplitudes_delta": delta,
        "amplitudes_average": average,
    }
    return payload_sha256(
        schema=_PAYLOAD_SCHEMA,
        expected_set_digest=_EXPECTED_SET_DIGEST,
        datasets=datasets,
        attrs=_PAYLOAD_ATTRS,
    )


def _all_reduction_hashes(
    partials: list[ScatteringPartialResult],
) -> dict[str, str]:
    """Hash the chunk reduced in every order and several groupings.

    Returns a label -> payload_sha256 map covering:
      * every permutation of the partials (different reduce orders), and
      * group sizes 1..N (different worker / partition counts).
    """
    n = len(partials)
    hashes: dict[str, str] = {}
    for permutation in itertools.permutations(range(n)):
        ordered = [partials[i] for i in permutation]
        label = "order:" + "".join(str(i) for i in permutation)
        hashes[label] = _payload_sha256_of(_reduce_sequential(ordered))
    for group_size in range(1, n + 1):
        label = f"group:{group_size}"
        hashes[label] = _payload_sha256_of(
            _reduce_grouped(partials, group_size=group_size)
        )
    return hashes


# --- Supporting evidence (must PASS): the hash is a pure function of the bytes ---


def test_identical_reduced_bytes_hash_identically():
    """Two reductions that yield bit-identical arrays must hash identically.

    This pins down that the hashing layer itself is deterministic, so any
    divergence seen in the xfail test below is attributable to the *reduce*, not to
    the fingerprinting.
    """
    partials = _build_partials(n_intervals=4, n_points=129, seed=2026)
    # Same order twice: the reduced arrays are bit-for-bit identical, ...
    first = _reduce_sequential(list(partials))
    second = _reduce_sequential(list(partials))

    assert np.array_equal(first.amplitudes_delta, second.amplitudes_delta)
    assert np.array_equal(first.amplitudes_average, second.amplitudes_average)
    # ... and therefore so are the canonical payload hashes.
    assert _payload_sha256_of(first) == _payload_sha256_of(second)


def test_numpy_add_is_not_associative_for_this_data():
    """Document the root cause: float64 ``np.add`` is order-dependent here.

    If this ever starts passing as 'associative' it means the fixture stopped
    exercising the non-associativity and the xfail test below would no longer be a
    meaningful guard.
    """
    partials = _build_partials(n_intervals=4, n_points=257, seed=99)
    forward = _reduce_sequential([partials[i] for i in (0, 1, 2, 3)])
    reordered = _reduce_sequential([partials[i] for i in (3, 2, 1, 0)])

    # Numerically close (same mathematical sum) ...
    np.testing.assert_allclose(
        forward.amplitudes_delta, reordered.amplitudes_delta, rtol=0, atol=1e-9
    )
    # ... but NOT bit-identical, which is exactly why exact-hash promotion is at risk.
    assert not np.array_equal(
        forward.amplitudes_delta, reordered.amplitudes_delta
    )


# --- The Phase 2 claim under test (currently XFAIL: see findings note) ---


@pytest.mark.xfail(
    strict=True,
    reason=(
        "merge_scattering_partial_results reduces complex128 amplitudes with np.add, "
        "which is not IEEE-754 associative, so arbitrary reduce orders / groupings can "
        "differ in the last bits and yield different payload_sha256 values. SCOPE: the "
        "durable commit path (core/scattering/commit.py create_scattering_commit_candidate) "
        "folds per-interval attempts in canonical sorted order, so this non-associativity "
        "is LATENT there, not an active worker-count defect; the real exact-hash exposure "
        "is attempt-level NUFFT determinism and CPU/GPU parity. See "
        "tests/unit/scattering/REDUCE_DETERMINISM_FINDINGS.md. This test guards the merge "
        "primitive and will XPASS if it is ever made bitwise order-stable."
    ),
)
def test_reduce_order_invariant_payload_sha256():
    """All reduce orders and worker counts must yield ONE payload_sha256.

    Build N=4 partials, reduce them in every order and several groupings, hash the
    canonical payload of each, and assert every hash is identical. A single distinct
    hash means cross-environment determinism holds; more than one means it does not.
    """
    partials = _build_partials(n_intervals=4, n_points=257, seed=20260629)
    hashes = _all_reduction_hashes(partials)

    distinct = sorted(set(hashes.values()))
    assert len(distinct) == 1, (
        f"Expected one payload_sha256 across all reduce orders / worker counts, "
        f"got {len(distinct)} distinct hashes over {len(hashes)} reductions. "
        f"Sample: "
        + ", ".join(f"{label}={digest[:12]}" for label, digest in list(hashes.items())[:6])
    )
