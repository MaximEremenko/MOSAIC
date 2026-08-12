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


def test_residual_candidate_rejects_rewritten_point_ids(tmp_path):
    write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=0,
        point_start=0,
        point_stop=2,
        interval_ids=(1,),
        attempt_id="try1",
        point_ids=np.array([10, 11], dtype=np.int64),
        grid_shape_nd=np.array([[2]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        contribution_reciprocal_points=9,
        **IDENTITY,
    )

    with pytest.raises(RuntimeError, match="point IDs"):
        create_residual_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=5,
            expected_partitions={0: (10, 12)},
            expected_reciprocal_point_count=9,
        )


def test_residual_candidate_rejects_unexpected_partition_attempt(tmp_path):
    write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id="try1",
        point_ids=np.array([10], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=9,
        **IDENTITY,
    )
    write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=1,
        point_start=1,
        point_stop=2,
        interval_ids=(1,),
        attempt_id="try1",
        point_ids=np.array([11], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=9,
        **IDENTITY,
    )

    with pytest.raises(RuntimeError, match="unexpected partitions"):
        create_residual_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=5,
            expected_partitions={0: (10,)},
            expected_reciprocal_point_count=9,
        )
