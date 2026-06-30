"""Phase A wiring: build_qspace_plan contracts, the qspace_plan_digest stability
guarantee, and the commit-time reciprocal-point reconciliation assertion.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.qspace.normalization import (
    QNormalizationContract,
    load_q_normalization_contract,
    write_q_normalization_sidecar,
)
from core.scattering.commit import (
    ScatteringNormalizationError,
    write_scattering_attempt,
)
from core.scattering.planning import (
    build_qspace_plan,
    build_run_identity,
    prepare_scattering_run_identity,
    write_qspace_plan,
)
from core.storage.attempt_store import qspace_plan_path
from core.storage.fingerprint import file_sha256


def _parameters():
    return {
        "supercell": np.array([4.0, 4.0, 4.0]),
        "vectors": np.eye(3),
        "reciprocal_space_intervals": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
        ],
        "reciprocal_space_intervals_all": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
        ],
        "mask_parameters": {},
        "mask_strategy": "none",
        "charge": 1.0,
    }


def _identity(parameters):
    return build_run_identity(
        parameters,
        backend="cpu",
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        backend_policy_digest="policy-a",
    )


def test_build_qspace_plan_attaches_contracts_with_correct_counts():
    parameters = _parameters()
    identity = _identity(parameters)
    plan = build_qspace_plan(
        parameters=parameters,
        identity=identity,
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
    )

    by_interval = {c.interval_id: c for c in plan.q_normalization_contracts}
    assert set(by_interval) == {1, 2}

    # Both intervals are single hkl points (all-pass "none" mask): planned == accepted,
    # so mask_rejected == 0 under the default all-pass mask.
    c1 = by_interval[1]
    assert c1.planned_count == 1
    assert c1.accepted_count == 1
    assert c1.mask_rejected == 0
    assert c1.multiplicity == 1  # zero_plane
    assert c1.reciprocal_point_count == 1

    c2 = by_interval[2]
    assert c2.planned_count == 1
    assert c2.accepted_count == 1
    assert c2.mask_rejected == 0
    assert c2.multiplicity == 2  # positive_half
    # multiplicity applied exactly once: accepted(1) x multiplicity(2)
    assert c2.reciprocal_point_count == 2
    # planned_count is multiplicity-FREE and matches the QSpaceIntervalPlan's
    # multiplicity-folded persisted count divided by the multiplicity.
    persisted = {iv.interval_id: iv for iv in plan.intervals}
    assert persisted[2].reciprocal_point_count == c2.planned_count * c2.multiplicity
    # q_digest on the contract matches the plan's per-interval q_grid digest.
    assert c2.q_digest == persisted[2].q_grid_digest


def test_qspace_plan_digest_is_byte_stable_with_and_without_contracts():
    """The sidecar feature must not change qspace_plan.json's bytes (its file_sha256 IS
    qspace_plan_digest, part of work-unit identity). Prove the plan file's hash is
    identical whether or not the in-memory plan carries q_normalization_contracts.
    """
    parameters = _parameters()
    identity = _identity(parameters)
    plan_with = build_qspace_plan(
        parameters=parameters,
        identity=identity,
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
    )
    assert plan_with.q_normalization_contracts  # contracts are present in memory

    # A plan stripped of contracts (simulating the pre-feature shape) must serialize to
    # the same bytes -- to_payload deliberately excludes the contracts.
    plan_without = plan_with.__class__(
        run_digest=plan_with.run_digest,
        scientific_digest=plan_with.scientific_digest,
        execution_digest=plan_with.execution_digest,
        mask_digest=plan_with.mask_digest,
        intervals=plan_with.intervals,
        q_grid_set_digest=plan_with.q_grid_set_digest,
    )
    assert plan_without.q_normalization_contracts == ()
    assert plan_with.to_payload() == plan_without.to_payload()


def test_prepare_run_identity_writes_sidecar_and_keeps_plan_digest_stable(tmp_path):
    parameters = _parameters()
    work_identity = prepare_scattering_run_identity(
        parameters=parameters,
        output_dir=tmp_path,
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
        backend="cpu",
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        scheduler_kind="local",
        interval_artifact_policy="keep",
    )

    plan_path = qspace_plan_path(tmp_path, work_identity.run_digest)
    sidecar_path = plan_path.parent / "q_normalization.json"

    # The sidecar exists next to qspace_plan.json...
    assert sidecar_path.exists()
    # ...and the plan file's hash equals the work identity's qspace_plan_digest
    # (i.e. writing the sidecar did not perturb the plan file's bytes).
    assert file_sha256(plan_path) == work_identity.qspace_plan_digest

    # The sidecar carries the per-interval contract and reconciles round-trip.
    contract = load_q_normalization_contract(tmp_path, work_identity.run_digest, 2)
    assert contract is not None
    assert contract.reciprocal_point_count == 2


_ATTEMPT_IDENTITY = {
    "run_digest": "runA1A4",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _write_sidecar(tmp_path, *, interval_id, reciprocal_point_count):
    # accepted x multiplicity == reciprocal_point_count; pick multiplicity 1 so
    # accepted == reciprocal_point_count and planned >= accepted.
    write_q_normalization_sidecar(
        tmp_path,
        _ATTEMPT_IDENTITY["run_digest"],
        [
            QNormalizationContract(
                planned_count=reciprocal_point_count,
                accepted_count=reciprocal_point_count,
                multiplicity=1,
                half_space_role="full",
                interval_id=interval_id,
                q_digest="d" * 64,
            )
        ],
    )


def test_commit_assertion_passes_when_attempt_agrees_with_contract(tmp_path):
    _write_sidecar(tmp_path, interval_id=2, reciprocal_point_count=4)
    manifest = write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=2,
        chunk_id=7,
        attempt_id="worker1-try1",
        grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        contribution_reciprocal_points=4,  # matches the contract
        **_ATTEMPT_IDENTITY,
    )
    assert manifest.contribution_reciprocal_points == 4


def test_commit_assertion_raises_on_forged_mismatch(tmp_path):
    # Contract says 4; the attempt forges 5 -> fail closed.
    _write_sidecar(tmp_path, interval_id=2, reciprocal_point_count=4)
    with pytest.raises(ScatteringNormalizationError, match="disagrees with the q-normalization"):
        write_scattering_attempt(
            output_dir=tmp_path,
            interval_id=2,
            chunk_id=7,
            attempt_id="worker1-try1",
            grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
            amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),
            amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
            contribution_reciprocal_points=5,  # disagrees with contract (4)
            **_ATTEMPT_IDENTITY,
        )


def test_commit_assertion_skipped_when_sidecar_absent(tmp_path):
    # No sidecar written -> loader returns None -> assertion skipped (back-compat).
    manifest = write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=2,
        chunk_id=7,
        attempt_id="worker1-try1",
        grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        contribution_reciprocal_points=999,  # would mismatch, but no contract present
        **_ATTEMPT_IDENTITY,
    )
    assert manifest.contribution_reciprocal_points == 999
