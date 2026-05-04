from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.residual_field.artifacts import (
    persist_residual_field_shard_checkpoint,
    reduce_residual_field_shards_for_chunk,
    write_stage2_replacement_expected_manifest,
)
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.planning import build_residual_field_parameter_digest
from core.residual_field.stage import ResidualFieldStage
from core.storage.database_manager import DatabaseManager


def _seed_db_for_replacement(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 10,
                "coordinates": [0.0],
                "dist_from_atom_center": [0.0],
                "step_in_frac": [0.0],
                "chunk_id": 3,
                "grid_amplitude_initialized": 1,
            }
        ]
    )
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [{"h_range": (0.0, 1.0)}, {"h_range": (1.0, 2.0)}]
    )
    db.insert_interval_chunk_status_batch(
        [(interval_ids[0], 3, 0), (interval_ids[1], 3, 0)]
    )
    return db, tuple(interval_ids)


def _workflow_parameters():
    return SimpleNamespace(
        runtime_info={
            "scattering_stage2_mode": "replacement",
        }
    )


def test_residual_stage_skips_when_stage2_replacement_manifest_is_valid(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=interval_ids,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[1]], dtype=np.int64),
            total_reciprocal_points=7,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([2 + 0j]),
            amplitudes_average=np.array([3 + 0j]),
            point_ids=np.array([10]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )
        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            expected_interval_ids=interval_ids,
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        called = []

        def fail_if_called(**kwargs):
            called.append(kwargs)
            raise AssertionError("residual fallback should not run")

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fail_if_called,
        )

        result = ResidualFieldStage().execute(
            workflow_parameters=_workflow_parameters(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={
                "residual_parameter_digest": "abc123",
                "stage2_replacement_expected_by_chunk": {3: interval_ids},
            },
        )

        assert called == []
        assert result["residual_parameter_digest"] == "abc123"
    finally:
        db.close()


def test_residual_stage_prefers_fresh_expected_over_db_rows(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        db.update_interval_chunk_status(interval_ids[1], 3, saved=True)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_ids[0],),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[1]], dtype=np.int64),
            total_reciprocal_points=7,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([2 + 0j]),
            amplitudes_average=np.array([3 + 0j]),
            point_ids=np.array([10]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )
        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            expected_interval_ids=(interval_ids[0],),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        called = []

        def fail_if_called(**kwargs):
            called.append(kwargs)
            raise AssertionError("residual fallback should not run")

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fail_if_called,
        )

        ResidualFieldStage().execute(
            workflow_parameters=_workflow_parameters(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={
                "residual_parameter_digest": "abc123",
                "stage2_replacement_expected_by_chunk": {3: (interval_ids[0],)},
            },
        )

        assert called == []
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_residual_stage_treats_empty_fresh_expected_as_complete_noop(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        for interval_id in interval_ids:
            db.update_interval_chunk_status(interval_id, 3, saved=True)

        called = []

        def fail_if_called(**kwargs):
            called.append(kwargs)
            raise AssertionError("residual fallback should not run")

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fail_if_called,
        )

        result = ResidualFieldStage().execute(
            workflow_parameters=_workflow_parameters(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={
                "residual_parameter_digest": "abc123",
                "stage2_replacement_expected_by_chunk": {},
            },
        )

        assert called == []
        assert result["stage2_replacement_expected_by_chunk"] == {}
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_residual_stage_does_not_trust_db_saved_without_replacement_manifest(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        for interval_id in interval_ids:
            db.update_interval_chunk_status(interval_id, 3, saved=True)

        called = []

        def fake_fallback(**kwargs):
            called.append(kwargs)

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fake_fallback,
        )

        ResidualFieldStage().execute(
            workflow_parameters=_workflow_parameters(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={
                "residual_parameter_digest": "abc123",
                "stage2_replacement_expected_by_chunk": {3: interval_ids},
            },
        )

        assert len(called) == 1
        assert set(db.get_unsaved_interval_chunks()) == {
            (interval_ids[0], 3),
            (interval_ids[1], 3),
        }
    finally:
        db.close()


def test_residual_stage_loads_expected_manifest_when_scattering_returns_no_params(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        workflow_parameters = _workflow_parameters()
        parameter_digest = build_residual_field_parameter_digest(workflow_parameters)
        write_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest=parameter_digest,
            expected_by_chunk={3: interval_ids},
        )
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=interval_ids,
            chunk_id=3,
            parameter_digest=parameter_digest,
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[1]], dtype=np.int64),
            total_reciprocal_points=7,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([2 + 0j]),
            amplitudes_average=np.array([3 + 0j]),
            point_ids=np.array([10]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )
        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest=parameter_digest,
            expected_interval_ids=interval_ids,
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        called = []

        def fail_if_called(**kwargs):
            called.append(kwargs)
            raise AssertionError("residual fallback should not run")

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fail_if_called,
        )

        ResidualFieldStage().execute(
            workflow_parameters=workflow_parameters,
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={},
        )

        assert called == []
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_residual_stage_does_not_use_db_rows_as_expected_by_default(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _seed_db_for_replacement(tmp_path)
    try:
        for interval_id in interval_ids:
            db.update_interval_chunk_status(interval_id, 3, saved=True)

        called = []

        def fail_if_called(**kwargs):
            called.append(kwargs)
            raise AssertionError("residual fallback should not run")

        monkeypatch.setattr(
            "core.residual_field.stage.run_residual_field_stage",
            fail_if_called,
        )

        result = ResidualFieldStage().execute(
            workflow_parameters=_workflow_parameters(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path), db_manager=db),
            client=None,
            scattering_parameters={},
        )

        assert result == {}
        assert called == []
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()
