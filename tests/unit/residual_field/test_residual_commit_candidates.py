"""P11 C-4 residual commit identity: numerical agreement gate + device-independence.

Mirrors the scattering C-2/C-3 contract for the residual stage:

* same-work residual retries (non-deterministic GPU relaunches, or a CPU vs GPU
  attempt for the same partition) need not be bit-identical, but a real numerical
  DIVERGENCE fails closed -- bitwise ``payload_sha256`` equality was REPLACED by a
  numerical agreement gate, not removed; and
* a CPU attempt and a GPU attempt for the same partition science differ ONLY in the
  device-bound ``backend_policy_digest`` (now metadata, not identity) and so share
  ONE checkpoint instead of being rejected as "conflicting".
"""

from __future__ import annotations

import numpy as np
import pytest

from core.residual_field.commit import (
    create_residual_commit_candidate,
    write_residual_attempt,
)


IDENTITY = {
    "run_digest": "run123",
    "parameter_digest": "d" * 64,
    "partition_plan_digest": "e" * 64,
    "source_scattering_commit_digest": "a" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "0" * 64,
}


def _attempt(tmp_path, *, attempt_id, delta, identity=None):
    return write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id=attempt_id,
        point_ids=np.array([10], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([delta], dtype=np.complex128),
        amplitudes_average=np.array([0.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=9,
        **(identity or IDENTITY),
    )


def test_residual_candidate_rejects_divergent_retry_attempts(tmp_path):
    # A retry that produces a numerically DIVERGENT payload (real bug, not float
    # noise) fails closed -- the numerical agreement gate replaced the old exact-hash
    # check, it did not remove the safety.
    _attempt(tmp_path, attempt_id="try1", delta=1.0 + 0.0j)
    _attempt(tmp_path, attempt_id="try2", delta=5.0 + 0.0j)

    with pytest.raises(RuntimeError, match="Divergent residual results"):
        create_residual_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=5,
            expected_partitions={0: (10,)},
            expected_reciprocal_point_count=9,
        )


def test_residual_candidate_accepts_agreeing_retry_attempts_and_picks_winner(tmp_path):
    # A non-deterministic relaunch differs in BYTES but AGREES within tolerance:
    # accepted, with a deterministic winner (sorted attempt_id).
    _attempt(tmp_path, attempt_id="try1", delta=1.0 + 0.0j)
    _attempt(tmp_path, attempt_id="try2", delta=1.0 + 1e-12j)

    candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
        expected_partitions={0: (10,)},
        expected_reciprocal_point_count=9,
    )

    assert len(candidate.selected_attempts) == 1
    assert candidate.selected_attempts[0]["attempt_id"] == "try1"


def test_residual_candidate_treats_cpu_and_gpu_attempts_as_shared_checkpoint(tmp_path):
    # P11 C-4: a CPU attempt and a GPU attempt for the same partition science differ
    # ONLY in device-bound metadata (backend_policy_digest). They must map to ONE
    # checkpoint identity -- NOT be rejected as "conflicting" -- and, agreeing within
    # tolerance, promote to a single deterministically-chosen attempt.
    cpu_identity = dict(IDENTITY)
    gpu_identity = dict(IDENTITY)
    gpu_identity["backend_policy_digest"] = "9" * 64

    _attempt(tmp_path, attempt_id="cpu", delta=1.0 + 0.0j, identity=cpu_identity)
    _attempt(tmp_path, attempt_id="gpu", delta=1.0 + 1e-12j, identity=gpu_identity)

    candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
        expected_partitions={0: (10,)},
        expected_reciprocal_point_count=9,
    )

    assert len(candidate.selected_attempts) == 1
    assert candidate.selected_attempts[0]["attempt_id"] == "cpu"  # deterministic winner
