from types import SimpleNamespace

import numpy as np

from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    persist_residual_field_interval_chunk_result,
)
from core.residual_field.loader import load_chunk_residual_field_and_grid
from core.residual_field.planning import (
    build_residual_field_parameter_digest,
    build_residual_field_work_units,
)
from core.residual_field.stage import ResidualFieldStage
from core.storage.database_manager import DatabaseManager


def test_residual_field_planning_builds_interval_chunk_work_units(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(2, 3), (1, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
    )

    assert [(unit.interval_id, unit.chunk_id) for unit in work_units] == [(1, 3), (2, 3)]
    assert work_units[0].source_artifacts[0].kind == "interval-precompute"
    assert work_units[0].retry.idempotency_key.endswith("interval-1")
    assert build_residual_field_parameter_digest(parameters) == work_units[0].parameter_digest


def test_residual_field_planning_batches_intervals_per_chunk(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(1, 3), (2, 3), (3, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
        max_intervals_per_shard=2,
    )

    assert [unit.interval_ids for unit in work_units] == [(1, 2), (3,)]
    assert work_units[0].retry.idempotency_key.endswith("batch-1-2-n2")
    assert work_units[1].retry.idempotency_key.endswith("interval-3")


def test_residual_field_artifacts_preserve_current_saved_and_applied_semantics(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        db.insert_point_data_batch(
            [
                {
                    "central_point_id": 10,
                    "coordinates": [0.1],
                    "dist_from_atom_center": [0.2],
                    "step_in_frac": [0.05],
                    "chunk_id": 3,
                    "grid_amplitude_initialized": 1,
                }
            ]
        )
        interval_id = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])[0]
        db.insert_interval_chunk_status_batch([(interval_id, 3, 0)])

        baseline = np.array([[10 + 0j, 0 + 0j], [10 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )

        from core.residual_field.contracts import ResidualFieldWorkUnit

        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        manifest = persist_residual_field_interval_chunk_result(
            work_unit,
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

        current, current_av, nrec, _ = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        assert manifest.chunk_id == 3
        assert interval_id in applied
        assert (interval_id, 3) not in set(db.get_unsaved_interval_chunks())
        np.testing.assert_allclose(current[:, 1], np.array([1 + 0j, 2 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j, 0.75 + 0j]))
        assert nrec == 5
    finally:
        db.close()


def test_residual_field_loader_reconstructs_grid_and_normalizes_values(tmp_path):
    store = ResidualFieldArtifactStore(str(tmp_path))
    payload = np.array([[10 + 0j, 2 + 0j], [10 + 0j, 4 + 0j]], dtype=np.complex128)
    store.save_chunk_payloads(
        3,
        amplitudes_payload=payload,
        amplitudes_average_payload=payload.copy(),
        reciprocal_point_count=2,
    )
    store.saver.save_data(
        {
            "ntotal_reciprocal_space_points": np.array([2], dtype=np.int64),
            "ntotal_reciprocal_points": np.array([2], dtype=np.int64),
        },
        store.saver.generate_filename(3, suffix="_amplitudes_ntotal_reciprocal_space_points"),
    )

    class FakePointDataProcessor:
        def generate_grid(
            self,
            *,
            chunk_id,
            dimensionality,
            step_in_frac,
            central_point,
            dist,
            central_point_id,
        ):
            return np.array([[0.1], [0.2]]), np.array([2])

    processor = SimpleNamespace(point_data_processor=FakePointDataProcessor())
    point_data_list = [
        {
            "central_point_id": 10,
            "coordinates": np.array([0.0]),
            "dist_from_atom_center": np.array([0.2]),
            "step_in_frac": np.array([0.1]),
        }
    ]
    data, amplitudes, rifft_grid = load_chunk_residual_field_and_grid(
        processor,
        chunk_id=3,
        point_data_list=point_data_list,
        rifft_saver=store.saver,
        logger=None,
    )

    assert "amplitudes" in data
    np.testing.assert_allclose(amplitudes[:, 1], np.array([1 + 0j, 2 + 0j]))
    assert rifft_grid.shape == (2, 2)
    np.testing.assert_allclose(rifft_grid[:, 1], np.array([10, 10]))


def test_residual_field_stage_delegates_to_execution(monkeypatch):
    captured = {}

    def fake_execute(*, workflow_parameters, structure, artifacts, client):
        captured["workflow_parameters"] = workflow_parameters
        captured["structure"] = structure
        captured["artifacts"] = artifacts
        captured["client"] = client

    monkeypatch.setattr(
        "core.residual_field.stage.run_residual_field_stage",
        fake_execute,
    )

    stage = ResidualFieldStage()
    artifacts = SimpleNamespace(db_manager="db", output_dir="/tmp/out")
    params = {"point_data_list": [], "supercell": np.array([4]), "reciprocal_space_intervals_all": []}

    result = stage.execute(
        workflow_parameters=SimpleNamespace(name="workflow"),
        structure=SimpleNamespace(name="structure"),
        artifacts=artifacts,
        client=None,
        scattering_parameters=params,
    )

    assert result is params
    assert captured["workflow_parameters"].name == "workflow"
    assert captured["structure"].name == "structure"
    assert captured["artifacts"] is artifacts
