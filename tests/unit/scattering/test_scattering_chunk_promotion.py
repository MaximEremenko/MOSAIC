from __future__ import annotations

import numpy as np
import pytest

from core.scattering.commit import (
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    write_scattering_attempt,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _candidate(tmp_path, *, chunk_id, interval_id=1, delta=2.0 + 0.0j):
    write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=interval_id,
        chunk_id=chunk_id,
        attempt_id=f"try-{interval_id}",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([delta], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        **IDENTITY,
    )
    return create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=chunk_id,
        expected_interval_ids=(interval_id,),
    )


def test_chunk_promotion_is_manifest_only_and_idempotent(tmp_path):
    candidate = _candidate(tmp_path, chunk_id=3)

    commit = promote_scattering_chunk_commit(output_dir=tmp_path, candidate=candidate)
    repeat = promote_scattering_chunk_commit(output_dir=tmp_path, candidate=candidate)

    assert repeat == commit
    assert commit.selected_candidate_id == candidate.candidate_id
    assert commit.candidate_payload_path == candidate.payload_path
    assert not (tmp_path / "point_data_chunk_3_amplitudes.hdf5").exists()


def test_chunk_promotion_rejects_different_candidate_after_commit(tmp_path):
    first = _candidate(tmp_path, chunk_id=3, delta=2.0 + 0.0j)
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=first)

    write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=2,
        chunk_id=3,
        attempt_id="try-2",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([4.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        **IDENTITY,
    )
    second = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1, 2),
    )

    with pytest.raises(RuntimeError, match="already committed"):
        promote_scattering_chunk_commit(output_dir=tmp_path, candidate=second)


def test_stage_commit_requires_chunk_commits(tmp_path):
    write_scattering_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={3: (1,)},
    )
    candidate = _candidate(tmp_path, chunk_id=3)
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=candidate)

    stage = write_scattering_stage_commit(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_ids=(3,),
    )

    assert stage.chunk_ids == (3,)
    assert stage.stage_plan_digest is not None
    assert len(stage.stage_digest) == 64

    with pytest.raises(RuntimeError, match="do not match stage_plan"):
        write_scattering_stage_commit(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_ids=(3, 4),
        )


def test_stage_plan_is_idempotent_and_conflict_checked(tmp_path):
    first = write_scattering_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={3: (2, 1)},
    )
    repeat = write_scattering_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={3: (1, 2)},
    )

    assert repeat == first

    with pytest.raises(RuntimeError, match="different expected coverage"):
        write_scattering_stage_plan(
            output_dir=tmp_path,
            run_digest="run123",
            expected_by_chunk={3: (1,)},
        )
