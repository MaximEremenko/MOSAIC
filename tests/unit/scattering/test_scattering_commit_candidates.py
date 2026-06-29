from __future__ import annotations

import json

import numpy as np
import pytest

from core.scattering.commit import (
    create_scattering_commit_candidate,
    write_scattering_attempt,
)


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _attempt(tmp_path, *, interval_id, attempt_id, delta, identity=None):
    return write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=interval_id,
        chunk_id=3,
        attempt_id=attempt_id,
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([delta], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        **(identity or IDENTITY),
    )


def test_commit_candidate_dedupes_retry_attempts_with_same_payload(tmp_path):
    first = _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    second = _attempt(tmp_path, interval_id=1, attempt_id="try2", delta=2.0 + 0.0j)
    _attempt(tmp_path, interval_id=2, attempt_id="try1", delta=4.0 + 0.0j)

    candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1, 2),
    )

    by_interval = {item["interval_id"]: item for item in candidate.selected_attempts}
    assert first.payload_sha256 == second.payload_sha256
    assert by_interval[1]["attempt_id"] == "try1"
    assert candidate.contributing_interval_ids == (1, 2)
    assert candidate.reciprocal_point_count == 2
    assert candidate.payload_path.startswith(
        ".mosaic/runs/run123/scattering/chunks/chunk_3/commit_attempts/"
    )


def test_commit_candidate_rejects_divergent_retry_attempts(tmp_path):
    # P11: two attempts for the same interval that DISAGREE numerically (a real
    # divergence, not float noise) still fail closed -- bitwise equality was
    # REPLACED by a numerical agreement gate, not removed.
    _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    _attempt(tmp_path, interval_id=1, attempt_id="try2", delta=3.0 + 0.0j)

    with pytest.raises(RuntimeError, match="Divergent scattering results"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )


def test_commit_candidate_accepts_agreeing_retry_attempts_and_picks_winner(tmp_path):
    # A non-deterministic relaunch differs in BYTES but AGREES within tolerance:
    # accepted, with a deterministic winner (sorted attempt_id) -- not byte-equal.
    _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    _attempt(tmp_path, interval_id=1, attempt_id="try2", delta=2.0 + 1e-12j)

    candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1,),
    )

    by_interval = {item["interval_id"]: item for item in candidate.selected_attempts}
    assert by_interval[1]["attempt_id"] == "try1"  # deterministic winner
    assert candidate.contributing_interval_ids == (1,)


def test_commit_candidate_treats_cpu_and_gpu_attempts_as_shared_checkpoint(tmp_path):
    # P11 C-2: a CPU attempt and a GPU attempt for the same (interval, chunk) science
    # differ ONLY in device-bound metadata (execution_digest / backend_policy_digest).
    # They must map to ONE checkpoint identity -- NOT be rejected as "conflicting" --
    # and, agreeing within tolerance (GPU non-determinism vs CPU is ~1e-13 rel-L2),
    # promote to a single deterministically-chosen attempt. This is the cross-device
    # sharing exit criterion of the P11 design.
    cpu_identity = dict(IDENTITY)
    gpu_identity = dict(IDENTITY)
    gpu_identity["execution_digest"] = "f" * 64
    gpu_identity["backend_policy_digest"] = "9" * 64

    _attempt(tmp_path, interval_id=1, attempt_id="cpu", delta=2.0 + 0.0j, identity=cpu_identity)
    _attempt(
        tmp_path, interval_id=1, attempt_id="gpu", delta=2.0 + 1e-12j, identity=gpu_identity
    )

    candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1,),
    )

    by_interval = {item["interval_id"]: item for item in candidate.selected_attempts}
    assert by_interval[1]["attempt_id"] == "cpu"  # deterministic winner (sorted attempt_id)
    assert candidate.contributing_interval_ids == (1,)


def test_commit_candidate_rejects_missing_expected_interval(tmp_path):
    _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)

    with pytest.raises(RuntimeError, match="missing attempts"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1, 2),
        )


def test_commit_candidate_rejects_unexpected_interval_attempt(tmp_path):
    _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    _attempt(tmp_path, interval_id=2, attempt_id="try1", delta=4.0 + 0.0j)

    with pytest.raises(RuntimeError, match="unexpected intervals"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )


def test_commit_candidate_rejects_identity_mismatch(tmp_path):
    _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    mismatched_identity = dict(IDENTITY)
    mismatched_identity["source_structure_digest"] = "c" * 64
    _attempt(
        tmp_path,
        interval_id=1,
        attempt_id="try2",
        delta=2.0 + 0.0j,
        identity=mismatched_identity,
    )

    with pytest.raises(RuntimeError, match="Conflicting scattering attempt identity"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )


def test_commit_candidate_rejects_tampered_work_unit_digest(tmp_path):
    manifest = _attempt(tmp_path, interval_id=1, attempt_id="try1", delta=2.0 + 0.0j)
    attempt_json = (tmp_path / manifest.payload_path).with_name("attempt.json")
    payload = json.loads(attempt_json.read_text(encoding="utf-8"))
    payload["work_unit_digest"] = "0" * 64
    attempt_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="work-unit digest mismatch"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )
