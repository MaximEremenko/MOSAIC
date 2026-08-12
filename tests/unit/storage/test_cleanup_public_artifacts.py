from __future__ import annotations

import numpy as np

from core.residual_field.commit import (
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    write_residual_attempt,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.scattering.commit import (
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    write_scattering_attempt,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)
from core.storage.attempt_store import run_root
from core.storage.cleanup import cleanup_run_artifacts
from core.storage.publisher import publish_run


SCATTERING_IDENTITY = {
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _complete_private_run(tmp_path, *, run_digest: str):
    write_scattering_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        interval_id=1,
        chunk_id=0,
        attempt_id="try1",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        point_ids=np.array([0], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )
    scattering_candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_interval_ids=(1,),
    )
    write_scattering_stage_plan(
        output_dir=tmp_path,
        run_digest=run_digest,
        expected_by_chunk={0: (1,)},
    )
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=scattering_candidate)
    scattering_stage = write_scattering_stage_commit(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_ids=(0,),
    )

    write_residual_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id="worker1-try1",
        parameter_digest="d" * 64,
        partition_plan_digest="e" * 64,
        source_scattering_commit_digest=scattering_stage.stage_digest,
        source_replacement_digest=None,
        backend_policy_digest="f" * 64,
        expected_output_digest="0" * 64,
        point_ids=np.array([0], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([0.25 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([0.125 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
    )
    residual_candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_partitions={0: (0,)},
        expected_reciprocal_point_count=1,
    )
    write_residual_stage_plan(
        output_dir=tmp_path,
        run_digest=run_digest,
        expected_by_chunk={0: (0,)},
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=residual_candidate)
    write_residual_stage_commit(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_ids=(0,),
    )


def test_cleanup_manifest_gated_removes_only_matching_duplicate_attempts(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    duplicate = write_scattering_attempt(
        output_dir=tmp_path,
        run_digest="run123",
        interval_id=1,
        chunk_id=0,
        attempt_id="try2",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        point_ids=np.array([0], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )
    conflict = write_scattering_attempt(
        output_dir=tmp_path,
        run_digest="run123",
        interval_id=1,
        chunk_id=0,
        attempt_id="try3",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([3.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        point_ids=np.array([0], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )

    report = cleanup_run_artifacts(tmp_path, "run123")

    assert not (tmp_path / duplicate.payload_path).parent.exists()
    assert (tmp_path / conflict.payload_path).parent.exists()
    assert any("try2" in path or duplicate.attempt_id in path for path in report.removed_paths)
    assert any("try3" in path or conflict.attempt_id in path for path in report.retained_paths)


def test_cleanup_keeps_unexplained_failures(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    failure_path = run_root(tmp_path, "run123") / "failure" / "worker.tmp"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text("traceback", encoding="utf-8")

    report = cleanup_run_artifacts(tmp_path, "run123")

    assert failure_path.exists()
    assert any("failure/worker.tmp" in path for path in report.retained_paths)


def test_cleanup_retention_policy_removes_only_superseded_runs(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    _complete_private_run(tmp_path, run_digest="run456")
    publish_run(tmp_path, "run123")
    publish_run(tmp_path, "run456", replace=True)

    cleanup_run_artifacts(tmp_path, "run456", remove_superseded_run=True)
    assert run_root(tmp_path, "run456").exists()

    cleanup_run_artifacts(tmp_path, "run123", remove_superseded_run=True)
    assert not run_root(tmp_path, "run123").exists()
