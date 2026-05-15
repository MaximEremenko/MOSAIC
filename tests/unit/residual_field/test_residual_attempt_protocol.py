from __future__ import annotations

import numpy as np

from core.residual_field.commit import (
    discover_residual_attempts,
    load_residual_attempt_payload,
    write_residual_attempt,
)
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.kernels import IntervalTask


IDENTITY = {
    "run_digest": "run123",
    "parameter_digest": "d" * 64,
    "partition_plan_digest": "e" * 64,
    "source_scattering_commit_digest": "a" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "0" * 64,
}


def test_residual_attempt_writes_only_run_scoped_attempt_paths(tmp_path):
    manifest = write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=2,
        point_start=0,
        point_stop=2,
        interval_ids=(1, 2),
        attempt_id="worker1-try1",
        point_ids=np.array([10, 11], dtype=np.int64),
        grid_shape_nd=np.array([[2]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        contribution_reciprocal_points=9,
        **IDENTITY,
    )

    assert manifest.payload_path.startswith(
        ".mosaic/runs/run123/residual_field/chunks/chunk_5/attempts/"
    )
    assert "residual_shards" not in manifest.payload_path
    assert not (tmp_path / "residual_shards").exists()
    assert not (tmp_path / "residual_chunk_5_amplitudes.hdf5").exists()

    discovered = discover_residual_attempts(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
    )
    assert discovered == (manifest,)

    datasets, attrs = load_residual_attempt_payload(manifest, output_dir=tmp_path)
    assert attrs["schema"] == "mosaic.residual_field.attempt"
    np.testing.assert_array_equal(datasets["point_ids"], np.array([10, 11]))
    np.testing.assert_allclose(
        datasets["amplitudes_delta"],
        np.array([1.0 + 2.0j, 3.0 + 0.0j]),
    )


def test_identity_complete_residual_task_writes_attempt_not_shard(monkeypatch, tmp_path):
    class FailingReducer:
        def persist_shard_checkpoint(self, *args, **kwargs):
            raise AssertionError("identity-complete current tasks must not write residual_shards")

    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[5.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
    )
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=5,
        output_dir=str(tmp_path),
        **IDENTITY,
    ).with_partition(
        partition_id=0,
        point_start=10,
        point_stop=11,
    )

    manifest = run_residual_field_interval_chunk_task(
        work_unit,
        IntervalTask(
            1,
            "All",
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        None,
        total_reciprocal_points=1,
        output_dir=str(tmp_path),
        reducer_backend=FailingReducer(),
        quiet_logs=True,
        rifft_payload=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[1]], dtype=np.int64),
        ),
    )

    attempts = discover_residual_attempts(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
    )
    assert attempts == (manifest,)
    assert attempts[0].partition_id == 0
    assert attempts[0].point_start == 10
    assert attempts[0].point_stop == 11
    assert not (tmp_path / "residual_shards").exists()
