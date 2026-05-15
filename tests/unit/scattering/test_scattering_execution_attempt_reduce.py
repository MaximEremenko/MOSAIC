from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.scattering import execution as scattering_execution
from core.scattering.artifacts import persist_scattering_interval_chunk_shard
from core.scattering.commit import (
    discover_scattering_attempts,
)
from core.scattering.planning import (
    ScatteringWorkIdentity,
    build_scattering_interval_chunk_work_units,
)
from core.scattering.kernels import IntervalTask
from core.storage.attempt_store import chunk_commit_path, stage_commit_path, stage_plan_path
from core.storage.database_manager import DatabaseManager
from core.runtime.dask_helpers import SyncClient


IDENTITY = ScatteringWorkIdentity(
    run_digest="run123",
    scientific_digest="1" * 64,
    execution_digest="2" * 64,
    qspace_plan_digest="3" * 64,
    backend_policy_digest="4" * 64,
    source_structure_digest="5" * 64,
)


def _db_with_chunk(tmp_path, *, interval_count: int = 2):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [{"h_range": (float(i), float(i + 1))} for i in range(interval_count)]
    )
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 10 + row,
                "coordinates": [0.0],
                "dist_from_atom_center": [0.0],
                "step_in_frac": [0.0],
                "chunk_id": 3,
                "grid_amplitude_initialized": 1,
            }
            for row in range(2)
        ]
    )
    db.insert_interval_chunk_status_batch(
        [(interval_id, 3, 0) for interval_id in interval_ids]
    )
    return db, tuple(interval_ids)


def _work_units(tmp_path, db):
    return build_scattering_interval_chunk_work_units(
        list(db.get_unsaved_interval_chunks()),
        dimension=1,
        output_dir=str(tmp_path),
        work_identity=IDENTITY,
    )


def _fake_chunk_task(
    work_unit,
    interval_path,
    atoms,
    *,
    total_reciprocal_points,
    output_dir,
    db_path,
    quiet_logs,
):
    del interval_path, atoms, total_reciprocal_points, db_path, quiet_logs
    scale = int(work_unit.interval_id)
    return persist_scattering_interval_chunk_shard(
        work_unit,
        grid_shape_nd=np.array([[2]], dtype=np.int64),
        total_reciprocal_points=9,
        contribution_reciprocal_points=scale,
        amplitudes_delta=np.array([scale + 0j, scale + 1j]),
        amplitudes_average=np.array([10 * scale + 0j, 20 * scale + 0j]),
        output_dir=output_dir,
        quiet_logs=True,
    )


def test_stage2_sync_reduces_attempts_to_commits_without_public_outputs(
    monkeypatch,
    tmp_path,
):
    db, interval_ids = _db_with_chunk(tmp_path)
    monkeypatch.setattr(
        scattering_execution,
        "run_scattering_interval_chunk_task",
        _fake_chunk_task,
    )
    try:
        scattering_execution.run_interval_chunk_execution(
            _work_units(tmp_path, db),
            total_reciprocal_points=9,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
        )

        attempts = discover_scattering_attempts(
            output_dir=tmp_path,
            run_digest=IDENTITY.run_digest,
            chunk_id=3,
        )
        assert [attempt.interval_id for attempt in attempts] == list(interval_ids)
        assert db.get_unsaved_interval_chunks() == []
        assert stage_plan_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
        assert chunk_commit_path(tmp_path, IDENTITY.run_digest, "scattering", 3).exists()
        assert stage_commit_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
        assert not (tmp_path / "scattering_shards").exists()
        assert not (tmp_path / "point_data_chunk_3_amplitudes.hdf5").exists()
    finally:
        db.close()


def test_stage2_sync_uses_transient_interval_payloads_when_artifacts_are_optional(
    monkeypatch,
    tmp_path,
):
    db, _interval_ids = _db_with_chunk(tmp_path, interval_count=1)
    payload = IntervalTask(
        1,
        "All",
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j]),
        np.array([0.0 + 0.0j]),
    )

    def _fake_chunk_task_requires_payload(
        work_unit,
        interval_input,
        atoms,
        *,
        total_reciprocal_points,
        output_dir,
        db_path,
        quiet_logs,
    ):
        assert isinstance(interval_input, IntervalTask)
        return _fake_chunk_task(
            work_unit,
            interval_input,
            atoms,
            total_reciprocal_points=total_reciprocal_points,
            output_dir=output_dir,
            db_path=db_path,
            quiet_logs=quiet_logs,
        )

    monkeypatch.setattr(
        scattering_execution,
        "run_scattering_interval_chunk_task",
        _fake_chunk_task_requires_payload,
    )
    try:
        scattering_execution.run_interval_chunk_execution(
            _work_units(tmp_path, db),
            total_reciprocal_points=9,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
            transient_interval_payloads={1: payload},
        )

        attempts = discover_scattering_attempts(
            output_dir=tmp_path,
            run_digest=IDENTITY.run_digest,
            chunk_id=3,
        )
        assert len(attempts) == 1
        assert not (tmp_path / "precomputed_intervals" / "interval_1.hdf5").exists()
    finally:
        db.close()


def test_stage2_sync_client_uses_serial_completion_path(monkeypatch, tmp_path):
    db, _interval_ids = _db_with_chunk(tmp_path, interval_count=1)
    payload = IntervalTask(
        1,
        "All",
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j]),
        np.array([0.0 + 0.0j]),
    )
    monkeypatch.setattr(
        scattering_execution,
        "run_scattering_interval_chunk_task",
        _fake_chunk_task,
    )
    try:
        scattering_execution.run_interval_chunk_execution(
            _work_units(tmp_path, db),
            total_reciprocal_points=9,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=SyncClient(),
            output_dir=str(tmp_path),
            transient_interval_payloads={1: payload},
        )

        assert discover_scattering_attempts(
            output_dir=tmp_path,
            run_digest=IDENTITY.run_digest,
            chunk_id=3,
        )
    finally:
        db.close()


def test_stage2_execution_requires_identity_complete_work_units(tmp_path):
    db, _ = _db_with_chunk(tmp_path, interval_count=1)
    bare_units = build_scattering_interval_chunk_work_units(
        list(db.get_unsaved_interval_chunks()),
        dimension=1,
        output_dir=str(tmp_path),
    )
    try:
        with pytest.raises(ValueError, match="complete scientific"):
            scattering_execution.run_interval_chunk_execution(
                bare_units,
                total_reciprocal_points=9,
                point_data_list=db.get_point_data_for_chunk(3),
                db_manager=db,
                client=None,
                output_dir=str(tmp_path),
            )
    finally:
        db.close()


def test_stage2_async_submits_commit_reducer_not_shard_reducer(
    monkeypatch,
    tmp_path,
):
    db, _ = _db_with_chunk(tmp_path)
    submitted = []

    class FakeFuture:
        status = "finished"

        def __init__(self, value):
            self._value = value

        def done(self):
            return True

        def result(self):
            return self._value

        def exception(self, timeout=None):
            del timeout
            return None

    class FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {"workers": {"worker-a": {"resources": {"nufft": 1}}}}

        def scatter(self, value, **kwargs):
            del kwargs
            return value

        def submit(self, func, *args, **kwargs):
            submitted.append((func, dict(kwargs)))
            call_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "allow_other_workers",
                    "key",
                    "pure",
                    "resources",
                    "retries",
                    "workers",
                }
            }
            return FakeFuture(func(*args, **call_kwargs))

    monkeypatch.setattr(
        scattering_execution,
        "run_scattering_interval_chunk_task",
        _fake_chunk_task,
    )
    monkeypatch.setattr(
        scattering_execution,
        "yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    try:
        scattering_execution.run_interval_chunk_execution(
            _work_units(tmp_path, db),
            total_reciprocal_points=9,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=FakeClient(),
            output_dir=str(tmp_path),
            max_inflight=4,
        )

        reducer_functions = [
            func.__name__
            for func, kwargs in submitted
            if kwargs.get("run_digest") == IDENTITY.run_digest
        ]
        assert reducer_functions == [
            scattering_execution.commit_scattering_attempts_for_chunk.__name__
        ]
        assert all(func.__name__ != "reduce_scattering_shards_for_chunk" for func, _ in submitted)
        assert db.get_unsaved_interval_chunks() == []
        assert stage_plan_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
        assert stage_commit_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
    finally:
        db.close()


def test_stage2_retry_after_chunk_commit_before_db_mark_is_idempotent(
    monkeypatch,
    tmp_path,
):
    db, interval_ids = _db_with_chunk(tmp_path, interval_count=1)
    monkeypatch.setattr(
        scattering_execution,
        "run_scattering_interval_chunk_task",
        _fake_chunk_task,
    )
    real_update = db.update_interval_chunk_status
    crashed = False

    def crash_once(interval_id, chunk_id, saved=1):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("db mark failed")
        return real_update(interval_id, chunk_id, saved=saved)

    monkeypatch.setattr(db, "update_interval_chunk_status", crash_once)
    try:
        with pytest.raises(RuntimeError, match="db mark failed"):
            scattering_execution.run_interval_chunk_execution(
                _work_units(tmp_path, db),
                total_reciprocal_points=9,
                point_data_list=db.get_point_data_for_chunk(3),
                db_manager=db,
                client=None,
                output_dir=str(tmp_path),
            )

        assert chunk_commit_path(tmp_path, IDENTITY.run_digest, "scattering", 3).exists()
        assert db.get_unsaved_interval_chunks() == [(interval_ids[0], 3)]

        scattering_execution.run_interval_chunk_execution(
            _work_units(tmp_path, db),
            total_reciprocal_points=9,
            point_data_list=db.get_point_data_for_chunk(3),
            db_manager=db,
            client=None,
            output_dir=str(tmp_path),
        )

        assert db.get_unsaved_interval_chunks() == []
        assert stage_plan_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
        assert stage_commit_path(tmp_path, IDENTITY.run_digest, "scattering").exists()
    finally:
        db.close()
