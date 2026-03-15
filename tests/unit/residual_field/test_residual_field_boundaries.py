from types import SimpleNamespace

import numpy as np

from core.residual_field.backend import resolve_residual_field_reducer_backend
from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    persist_residual_field_interval_chunk_result,
)
from core.residual_field.loader import load_chunk_residual_field_and_grid
from core.residual_field.planning import (
    build_residual_field_parameter_digest,
    build_residual_field_work_units,
)
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.stage import ResidualFieldStage
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.kernels import IntervalTask
from core.storage.database_manager import DatabaseManager


class _CapturingReducerBackend:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def persist_shard_checkpoint(self, work_unit, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _StopAfterIntervalPayloadScatter(RuntimeError):
    pass


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


def test_residual_field_reducer_backend_resolution_is_mode_aware(monkeypatch):
    parameters = SimpleNamespace(runtime_info={})

    local_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=None,
    )
    local_state = local_backend.describe_runtime_state(
        output_dir="/tmp/out",
        scratch_root="/tmp/scratch",
    )
    assert local_backend.layout.kind == "local_restartable"
    assert local_state.durable_root == "/tmp/out"
    assert local_state.local_scratch_root == "/tmp/scratch"
    assert local_state.scattering_interval_transport.startswith(
        "direct in-process interval payload handoff"
    )
    assert local_state.checkpoint_policy.interval_artifacts == "optional_output"
    assert (
        local_state.checkpoint_policy.shard_checkpoints
        == "required_local_restart_state"
    )

    async_client = SimpleNamespace(loop=SimpleNamespace(asyncio_loop=object()))
    durable_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=async_client,
    )
    assert durable_backend.layout.kind == "durable_shared_restartable"
    durable_state = durable_backend.describe_runtime_state(
        output_dir="/tmp/out",
        scratch_root="/tmp/scratch",
    )
    assert durable_state.checkpoint_policy.interval_artifacts == "required_transport"
    assert (
        durable_state.checkpoint_policy.shard_checkpoints
        == "required_durable_checkpoint"
    )
    assert (
        durable_state.checkpoint_policy.worker_local_scratch_role
        == "temporary_staging_only"
    )

    monkeypatch.setenv("DASK_BACKEND", "local")
    local_async_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=async_client,
    )
    assert local_async_backend.layout.kind == "local_restartable"
    monkeypatch.delenv("DASK_BACKEND", raising=False)

    override_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(
            runtime_info={"residual_field_reducer_backend": "local_restartable"}
        ),
        client=async_client,
    )
    assert override_backend.layout.kind == "local_restartable"

    durable_root_override = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(
            runtime_info={"residual_shard_durable_root": "/tmp/shared-shards"}
        ),
        client=async_client,
    )
    assert (
        durable_root_override.describe_runtime_state(
            output_dir="/tmp/out",
            scratch_root="/tmp/scratch",
        ).committed_shard_root
        == "/tmp/shared-shards"
    )


def test_residual_field_async_local_handoff_scatters_interval_payload_mapping(monkeypatch, tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    payload = IntervalTask(
        1,
        "All",
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([0.5 + 0.0j], dtype=np.complex128),
    )
    scatter_calls = []

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, data, broadcast=False, hash=True, **kwargs):
            scatter_calls.append((data, broadcast, hash))
            return {key: f"future-{key}" for key in data}

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterIntervalPayloadScatter()),
    )

    workflow_parameters = SimpleNamespace(runtime_info={})
    structure = SimpleNamespace(supercell=np.array([1]))
    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={1: payload},
    )

    try:
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=_FakeClient(),
        )
    except _StopAfterIntervalPayloadScatter:
        pass
    finally:
        monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert len(scatter_calls) == 1
    data, broadcast, hash_value = scatter_calls[0]
    assert isinstance(data, dict)
    assert data == {1: payload}
    assert broadcast is False
    assert hash_value is False


def test_residual_field_interval_chunk_task_uses_batched_inverse(monkeypatch, tmp_path):
    interval_path = tmp_path / "interval_1.npz"
    np.savez(
        interval_path,
        irecip_id=np.array([1], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: calls.__setitem__("count", calls["count"] + 1) or np.array([[5.0 + 0.0j], [6.0 + 0.0j]]),
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        interval_path,
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert calls["count"] == 1
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([5.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))


def test_residual_field_interval_chunk_task_accepts_direct_interval_payloads(monkeypatch, tmp_path):
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    reducer_backend = _CapturingReducerBackend("manifest")

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[7.0 + 0.0j], [8.0 + 0.0j]]),
    )

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([7.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([8.0 + 0.0j]))


def test_residual_field_interval_chunk_task_uses_super_batch_for_same_geometry(monkeypatch, tmp_path):
    interval_path_1 = tmp_path / "interval_1.npz"
    interval_path_2 = tmp_path / "interval_2.npz"
    for path, interval_id, q_amp in (
        (interval_path_1, 1, 2.0 + 0.0j),
        (interval_path_2, 2, 4.0 + 0.0j),
    ):
        np.savez(
            path,
            irecip_id=np.array([interval_id], dtype=np.int64),
            element=np.array(["All"]),
            q_grid=np.array([[0.0]], dtype=np.float64),
            q_amp=np.array([q_amp]),
            q_amp_av=np.array([1.0 + 0.0j]),
        )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    calls = {"count": 0}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: calls.__setitem__("count", calls["count"] + 1)
        or np.array(
            [
                [1.0 + 0.0j],
                [2.0 + 0.0j],
                [3.0 + 0.0j],
                [4.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(1, 2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        (interval_path_1, interval_path_2),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert calls["count"] == 1
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([4.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))


def test_residual_field_interval_chunk_task_groups_mixed_q_grid_batches(monkeypatch, tmp_path):
    interval_path_1 = tmp_path / "interval_1.npz"
    interval_path_2 = tmp_path / "interval_2.npz"
    np.savez(
        interval_path_1,
        irecip_id=np.array([1], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    np.savez(
        interval_path_2,
        irecip_id=np.array([2], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[1.0]], dtype=np.float64),
        q_amp=np.array([4.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    calls = []

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )

    def fake_super_batch(**kwargs):
        calls.append(np.asarray(kwargs["q_coords"]).copy())
        if np.allclose(kwargs["q_coords"], np.array([[0.0]], dtype=np.float64)):
            return np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128)
        return np.array([[3.0 + 0.0j], [4.0 + 0.0j]], dtype=np.complex128)

    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        fake_super_batch,
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(1, 2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        (interval_path_1, interval_path_2),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert len(calls) == 2
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([4.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))
