from __future__ import annotations

import pytest

from core.residual_field.contracts import (
    ResidualFieldWorkUnit,
    validate_residual_field_work_unit,
)


IDENTITY = {
    "run_digest": "run123",
    "partition_plan_digest": "n" * 64,
    "source_scattering_commit_digest": "s" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "o" * 64,
}


def test_residual_work_unit_current_identity_survives_partitioning(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
        interval_ids=(2, 1),
        chunk_id=5,
        parameter_digest="p" * 64,
        output_dir=str(tmp_path),
        **IDENTITY,
    )
    partitioned = work_unit.with_partition(
        partition_id=3,
        point_start=10,
        point_stop=20,
    )

    validate_residual_field_work_unit(partitioned)

    assert partitioned.interval_ids == (1, 2)
    assert partitioned.run_digest == "run123"
    assert partitioned.partition_plan_digest == IDENTITY["partition_plan_digest"]
    assert partitioned.source_scattering_commit_digest == IDENTITY["source_scattering_commit_digest"]
    assert partitioned.backend_policy_digest == IDENTITY["backend_policy_digest"]
    assert partitioned.expected_output_digest == IDENTITY["expected_output_digest"]
    assert partitioned.partition_id == 3
    assert partitioned.point_start == 10
    assert partitioned.point_stop == 20


def test_residual_work_unit_rejects_partial_current_identity(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=5,
        parameter_digest="p" * 64,
        output_dir=str(tmp_path),
        run_digest="run123",
    )

    with pytest.raises(ValueError, match="Current-run residual-field work units"):
        validate_residual_field_work_unit(work_unit)
