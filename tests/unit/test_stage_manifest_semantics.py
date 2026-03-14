from __future__ import annotations

import numpy as np

from core.contracts import CompletionStatus
from core.scattering.artifacts import (
    ScatteringArtifactStore,
    assess_scattering_manifest,
    build_scattering_chunk_manifest,
    build_scattering_interval_manifest,
    persist_precomputed_interval_artifact,
    persist_scattering_interval_chunk_result,
)
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import IntervalTask
from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    assess_residual_field_manifest,
    build_residual_field_chunk_manifest,
    persist_residual_field_interval_chunk_result,
)
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.storage.database_manager import DatabaseManager


def _seed_point_and_interval_state(db: DatabaseManager, chunk_id: int = 3) -> int:
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 10,
                "coordinates": [0.1],
                "dist_from_atom_center": [0.2],
                "step_in_frac": [0.05],
                "chunk_id": chunk_id,
                "grid_amplitude_initialized": 1,
            }
        ]
    )
    interval_id = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])[0]
    db.insert_interval_chunk_status_batch([(interval_id, chunk_id, 0)])
    return interval_id


def test_scattering_manifest_assessment_tracks_interval_and_chunk_resume(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ScatteringArtifactStore(str(tmp_path))
    try:
        interval_id = _seed_point_and_interval_state(db)
        interval_work_unit = ScatteringWorkUnit.precompute_interval(
            interval_id=interval_id,
            dimension=1,
            output_dir=str(tmp_path),
        )
        interval_manifest = build_scattering_interval_manifest(
            interval_work_unit,
            completion_status=CompletionStatus.COMMITTED,
        )

        interval_assessment = assess_scattering_manifest(interval_manifest, db_path=db.db_path)
        assert interval_assessment.is_complete is False
        assert interval_assessment.can_resume is True

        persist_precomputed_interval_artifact(
            interval_work_unit,
            IntervalTask(
                interval_id,
                "All",
                np.array([[0.0]]),
                np.array([1 + 0j]),
                np.array([0 + 0j]),
            ),
            db_path=db.db_path,
        )
        interval_assessment = assess_scattering_manifest(interval_manifest, db_path=db.db_path)
        assert interval_assessment.is_complete is True
        assert interval_assessment.can_resume is False

        baseline = np.array([[10 + 0j, 0 + 0j], [11 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )
        chunk_work_unit = ScatteringWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            dimension=1,
            output_dir=str(tmp_path),
        )
        chunk_manifest = build_scattering_chunk_manifest(
            chunk_work_unit,
            output_dir=str(tmp_path),
            completion_status=CompletionStatus.COMMITTED,
        )

        chunk_assessment = assess_scattering_manifest(chunk_manifest, db_path=db.db_path)
        assert chunk_assessment.is_complete is False
        assert chunk_assessment.can_resume is True

        persist_scattering_interval_chunk_result(
            chunk_work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )
        chunk_assessment = assess_scattering_manifest(chunk_manifest, db_path=db.db_path)
        assert chunk_assessment.is_complete is True
        assert chunk_assessment.can_resume is False
    finally:
        db.close()


def test_scattering_chunk_replay_is_idempotent(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ScatteringArtifactStore(str(tmp_path))
    try:
        interval_id = _seed_point_and_interval_state(db)
        persist_precomputed_interval_artifact(
            ScatteringWorkUnit.precompute_interval(
                interval_id=interval_id,
                dimension=1,
                output_dir=str(tmp_path),
            ),
            IntervalTask(
                interval_id,
                "All",
                np.array([[0.0]]),
                np.array([1 + 0j]),
                np.array([0 + 0j]),
            ),
            db_path=db.db_path,
        )
        baseline = np.array([[10 + 0j, 0 + 0j], [11 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )
        work_unit = ScatteringWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            dimension=1,
            output_dir=str(tmp_path),
        )

        kwargs = dict(
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )
        persist_scattering_interval_chunk_result(work_unit, **kwargs)
        persist_scattering_interval_chunk_result(work_unit, **kwargs)

        current, current_av, nrec, _ = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([1 + 0j, 2 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j, 0.75 + 0j]))
        assert nrec == 5
        assert applied == {interval_id}
    finally:
        db.close()


def test_residual_field_manifest_assessment_and_replay_are_explicit(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        interval_id = _seed_point_and_interval_state(db)
        persist_precomputed_interval_artifact(
            ScatteringWorkUnit.precompute_interval(
                interval_id=interval_id,
                dimension=1,
                output_dir=str(tmp_path),
            ),
            IntervalTask(
                interval_id,
                "All",
                np.array([[0.0]]),
                np.array([1 + 0j]),
                np.array([0 + 0j]),
            ),
            db_path=db.db_path,
        )
        baseline = np.array([[10 + 0j, 0 + 0j], [10 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )
        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        manifest = build_residual_field_chunk_manifest(
            work_unit,
            output_dir=str(tmp_path),
            completion_status=CompletionStatus.COMMITTED,
        )

        assessment = assess_residual_field_manifest(manifest, db_path=db.db_path)
        assert assessment.is_complete is False
        assert assessment.can_resume is True

        kwargs = dict(
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 10]),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )
        persist_residual_field_interval_chunk_result(work_unit, **kwargs)
        assessment = assess_residual_field_manifest(manifest, db_path=db.db_path)
        assert assessment.is_complete is True
        assert assessment.can_resume is False

        persist_residual_field_interval_chunk_result(work_unit, **kwargs)
        current, current_av, nrec, _ = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([1 + 0j, 2 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j, 0.75 + 0j]))
        assert nrec == 5
        assert applied == {interval_id}
    finally:
        db.close()
