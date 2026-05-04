from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    load_stage2_replacement_expected_manifest,
    persist_residual_field_shard_checkpoint,
    write_stage2_replacement_expected_manifest,
)
from core.scattering.execution import (
    _resolve_scattering_interval_artifact_policy,
    run_stage2_replacement_execution,
)
from core.storage.database_manager import DatabaseManager


def _db_with_replacement_work(tmp_path):
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


def test_stage2_replacement_forces_durable_interval_transport():
    assert (
        _resolve_scattering_interval_artifact_policy(
            parameters={
                "runtime_info": {
                    "scattering_stage2_mode": "replacement",
                    "scattering_interval_artifact_policy": "optional_output",
                }
            },
            client=None,
        )
        == "required_transport"
    )


def test_stage2_replacement_execution_reduces_to_residual_outputs(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _db_with_replacement_work(tmp_path)
    try:
        point_rows = db.get_point_data_for_chunk(3)

        def fake_residual_batch_task(
            work_unit,
            interval_inputs,
            atoms,
            *,
            total_reciprocal_points,
            output_dir,
            db_path,
            scratch_root,
            reducer_backend,
            owner_local_reducer,
            quiet_logs,
            **kwargs,
        ):
            del interval_inputs, atoms, db_path, scratch_root
            del reducer_backend, owner_local_reducer, quiet_logs, kwargs
            return persist_residual_field_shard_checkpoint(
                work_unit,
                grid_shape_nd=np.array([[1]], dtype=np.int64),
                total_reciprocal_points=int(total_reciprocal_points),
                contribution_reciprocal_points=5,
                amplitudes_delta=np.array([5 + 0j]),
                amplitudes_average=np.array([7 + 0j]),
                point_ids=np.array([10]),
                output_dir=output_dir,
                quiet_logs=True,
            )

        monkeypatch.setattr(
            "core.residual_field.tasks.run_residual_field_interval_chunk_task",
            fake_residual_batch_task,
        )

        expected = run_stage2_replacement_execution(
            unsaved_interval_chunks=db.get_unsaved_interval_chunks(),
            total_reciprocal_points=11,
            point_data_list=point_rows,
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
            parameter_digest="abc123",
            max_intervals_per_shard=2,
            max_inflight=4,
        )

        store = ResidualFieldArtifactStore(str(tmp_path))
        amplitudes, amplitudes_av, nrec, _ = store.load_chunk_payloads(3)
        manifest_expected = load_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest="abc123",
        )
        assert expected == {3: interval_ids}
        assert manifest_expected == {3: interval_ids}
        assert db.get_unsaved_interval_chunks() == []
        assert nrec == 5
        np.testing.assert_allclose(amplitudes[:, 1], np.array([5 + 0j]))
        np.testing.assert_allclose(amplitudes_av[:, 1], np.array([7 + 0j]))
    finally:
        db.close()


def test_stage2_replacement_writes_expected_manifest_before_reduce_failure(
    tmp_path,
    monkeypatch,
):
    db, interval_ids = _db_with_replacement_work(tmp_path)
    try:
        point_rows = db.get_point_data_for_chunk(3)

        def fake_residual_batch_task(
            work_unit,
            interval_inputs,
            atoms,
            *,
            total_reciprocal_points,
            output_dir,
            db_path,
            scratch_root,
            reducer_backend,
            owner_local_reducer,
            quiet_logs,
            **kwargs,
        ):
            del interval_inputs, atoms, db_path, scratch_root
            del reducer_backend, owner_local_reducer, quiet_logs, kwargs
            return persist_residual_field_shard_checkpoint(
                work_unit,
                grid_shape_nd=np.array([[1]], dtype=np.int64),
                total_reciprocal_points=int(total_reciprocal_points),
                contribution_reciprocal_points=5,
                amplitudes_delta=np.array([5 + 0j]),
                amplitudes_average=np.array([7 + 0j]),
                point_ids=np.array([10]),
                output_dir=output_dir,
                quiet_logs=True,
            )

        def fail_reduce(**kwargs):
            del kwargs
            raise RuntimeError("reduce failed")

        monkeypatch.setattr(
            "core.residual_field.tasks.run_residual_field_interval_chunk_task",
            fake_residual_batch_task,
        )
        monkeypatch.setattr(
            "core.scattering.execution.reduce_residual_field_shards_for_chunk",
            fail_reduce,
        )

        try:
            run_stage2_replacement_execution(
                unsaved_interval_chunks=db.get_unsaved_interval_chunks(),
                total_reciprocal_points=11,
                point_data_list=point_rows,
                db_manager=db,
                client=None,
                output_dir=str(tmp_path),
                parameter_digest="abc123",
                max_intervals_per_shard=2,
                max_inflight=4,
            )
        except RuntimeError as exc:
            assert "reduce failed" in str(exc)
        else:
            raise AssertionError("Stage-2 replacement reduce should fail")

        assert load_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest="abc123",
        ) == {3: interval_ids}
    finally:
        db.close()


def test_stage2_replacement_no_work_keeps_existing_expected_manifest(tmp_path):
    db, interval_ids = _db_with_replacement_work(tmp_path)
    try:
        write_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest="abc123",
            expected_by_chunk={3: interval_ids},
        )

        expected = run_stage2_replacement_execution(
            unsaved_interval_chunks=[],
            total_reciprocal_points=11,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
            parameter_digest="abc123",
            max_intervals_per_shard=2,
            max_inflight=4,
        )

        assert expected == {3: interval_ids}
        assert load_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest="abc123",
        ) == {3: interval_ids}
    finally:
        db.close()


def test_stage2_replacement_no_work_writes_empty_expected_manifest(tmp_path):
    db, _ = _db_with_replacement_work(tmp_path)
    try:
        expected = run_stage2_replacement_execution(
            unsaved_interval_chunks=[],
            total_reciprocal_points=11,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
            parameter_digest="abc123",
            max_intervals_per_shard=2,
            max_inflight=4,
        )

        assert expected == {}
        assert load_stage2_replacement_expected_manifest(
            output_dir=str(tmp_path),
            parameter_digest="abc123",
        ) == {}
    finally:
        db.close()


def test_stage2_replacement_uses_strict_chunk_owner_affinity(
    tmp_path,
    monkeypatch,
):
    db, _ = _db_with_replacement_work(tmp_path)
    try:
        submitted = []
        scattered = []

        class FakeFuture:
            status = "finished"

            def result(self):
                return object()

            def exception(self, timeout=None):
                return None

        class FakeClient:
            loop = SimpleNamespace(asyncio_loop=object())

            def scheduler_info(self):
                return {
                    "workers": {
                        "worker-a": {"resources": {"nufft": 1}},
                        "worker-b": {"resources": {"nufft": 1}},
                    }
                }

            def scatter(self, data, **kwargs):
                scattered.append(dict(kwargs))
                return data

            def submit(self, func, *args, **kwargs):
                submitted.append((func, dict(kwargs)))
                return FakeFuture()

        monkeypatch.setattr(
            "core.scattering.execution.yield_futures_with_results",
            lambda futures, client: ((future, True) for future in futures),
        )

        run_stage2_replacement_execution(
            unsaved_interval_chunks=db.get_unsaved_interval_chunks(),
            total_reciprocal_points=11,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=FakeClient(),
            output_dir=str(tmp_path),
            parameter_digest="abc123",
            max_intervals_per_shard=2,
            max_inflight=4,
        )

        task_submits = [
            kwargs
            for _func, kwargs in submitted
            if str(kwargs.get("key", "")).startswith("stage2-replacement-")
        ]
        reducer_submits = [
            kwargs
            for _func, kwargs in submitted
            if kwargs.get("chunk_id") == 3
        ]
        assert task_submits
        assert reducer_submits
        assert all(kwargs["workers"] == ["worker-a"] for kwargs in task_submits)
        assert all(kwargs["allow_other_workers"] is False for kwargs in task_submits)
        assert reducer_submits[0]["workers"] == ["worker-a"]
        assert reducer_submits[0]["allow_other_workers"] is False
        assert scattered[0]["workers"] == ["worker-a"]
    finally:
        db.close()
