from __future__ import annotations

import numpy as np
import pytest

from core.scattering.commit import create_scattering_commit_candidate, write_scattering_attempt


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _attempt(tmp_path, *, attempt_id, delta):
    return write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=1,
        chunk_id=3,
        attempt_id=attempt_id,
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([delta], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        **IDENTITY,
    )


def test_retry_with_matching_payload_dedupes_to_one_selected_attempt(tmp_path):
    first = _attempt(tmp_path, attempt_id="attempt-a", delta=2.0 + 0.0j)
    second = _attempt(tmp_path, attempt_id="attempt-b", delta=2.0 + 0.0j)

    candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1,),
    )

    assert first.payload_sha256 == second.payload_sha256
    assert len(candidate.selected_attempts) == 1
    assert candidate.selected_attempts[0]["attempt_id"] == "attempt-a"


def test_retry_with_different_payload_fails_closed(tmp_path):
    _attempt(tmp_path, attempt_id="attempt-a", delta=2.0 + 0.0j)
    _attempt(tmp_path, attempt_id="attempt-b", delta=3.0 + 0.0j)

    with pytest.raises(RuntimeError, match="Conflicting scattering attempts"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )
