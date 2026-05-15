from __future__ import annotations

import numpy as np
import pytest

from core.residual_field.commit import (
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    write_residual_attempt,
    write_residual_stage_commit,
    write_residual_stage_plan,
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


def _attempt(tmp_path, *, partition_id, point_id, reciprocal_count):
    return write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=partition_id,
        point_start=partition_id,
        point_stop=partition_id + 1,
        interval_ids=(1,),
        attempt_id=f"try-{partition_id}",
        point_ids=np.array([point_id], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([partition_id + 1.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=reciprocal_count,
        **IDENTITY,
    )


def test_residual_candidate_keeps_reciprocal_count_per_partition_not_sum(tmp_path):
    _attempt(tmp_path, partition_id=0, point_id=10, reciprocal_count=9)
    _attempt(tmp_path, partition_id=1, point_id=11, reciprocal_count=9)

    candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
        expected_partitions={0: (10,), 1: (11,)},
        expected_reciprocal_point_count=9,
    )
    write_residual_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={5: (0, 1)},
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=candidate)
    stage = write_residual_stage_commit(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_ids=(5,),
    )

    assert candidate.reciprocal_point_count == 9
    assert stage.chunk_ids == (5,)
    assert stage.stage_plan_digest is not None
    assert not (tmp_path / "residual_shards").exists()
    assert not (tmp_path / "residual_chunk_5_amplitudes.hdf5").exists()


def test_residual_candidate_rejects_partition_reciprocal_count_mismatch(tmp_path):
    _attempt(tmp_path, partition_id=0, point_id=10, reciprocal_count=9)
    _attempt(tmp_path, partition_id=1, point_id=11, reciprocal_count=10)

    with pytest.raises(RuntimeError, match="reciprocal point counts"):
        create_residual_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=5,
            expected_partitions={0: (10,), 1: (11,)},
            expected_reciprocal_point_count=9,
        )
