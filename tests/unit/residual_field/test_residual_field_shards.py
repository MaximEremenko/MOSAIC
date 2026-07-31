from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.contracts import CompletionStatus
from core.residual_field.backend import (
    ResidualFieldLocalAccumulatorPartial,
    build_residual_field_reducer_backend,
    clear_process_local_residual_field_backends,
    flush_process_local_residual_reducer_target,
    get_process_local_residual_field_backend,
    inspect_process_local_residual_reducer_target,
)
from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    assess_residual_field_shard_manifest,
    build_residual_field_generation_artifacts,
    build_residual_field_output_artifact_refs,
    build_residual_field_reducer_progress_artifact,
    build_residual_field_shard_manifest,
    delete_reclaimable_residual_field_shards,
    discover_stale_residual_field_generation_manifests,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    is_residual_field_replacement_complete,
    is_residual_field_shard_reclaimable,
    list_reclaimable_residual_field_shards,
    load_residual_field_generation_metadata,
    persist_residual_field_shard_checkpoint,
    reconcile_residual_field_reducer_progress,
    reduce_residual_field_shards_for_chunk,
    write_residual_field_reducer_progress_manifest,
)
from core.residual_field.local_accumulator import (
    LiveLocalAccumulator,
    build_local_accumulator_snapshot_path,
)
from core.residual_field.contracts import (
    ResidualFieldReducerProgressManifest,
    ResidualFieldWorkUnit,
    make_residual_field_reducer_key,
)
from core.storage.database_manager import DatabaseManager


def _seed_db_for_chunk(db: DatabaseManager, chunk_id: int = 3) -> tuple[int, int]:
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
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [{"h_range": (0.0, 1.0)}, {"h_range": (1.0, 2.0)}]
    )
    db.insert_interval_chunk_status_batch([(interval_ids[0], chunk_id, 0), (interval_ids[1], chunk_id, 0)])
    return interval_ids[0], interval_ids[1]


def test_residual_field_shard_checkpoint_is_immutable_and_complete(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    manifest = persist_residual_field_shard_checkpoint(
        work_unit,
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        contribution_reciprocal_points=5,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path),
        quiet_logs=True,
    )

    assessment = assess_residual_field_shard_manifest(manifest)
    assert assessment.is_complete is True
    assert assessment.can_resume is False
    assert Path(manifest.artifacts[0].path).exists()
    assert Path(manifest.artifacts[1].path).exists()
    assert not (tmp_path / "point_data_chunk_3_amplitudes.hdf5").exists()


def test_residual_field_shard_checkpoint_replay_is_idempotent(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    kwargs = dict(
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        contribution_reciprocal_points=5,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path),
        quiet_logs=True,
    )
    persist_residual_field_shard_checkpoint(work_unit, **kwargs)
    persist_residual_field_shard_checkpoint(work_unit, **kwargs)

    manifests = discover_residual_field_shard_manifests(
        output_dir=str(tmp_path),
        chunk_id=3,
        parameter_digest="abc123",
    )
    assert len(manifests) == 1
    assert manifests[0].interval_id == 7


def test_residual_field_batch_shard_checkpoint_uses_single_durable_manifest(tmp_path):
    scratch_root = tmp_path / "scratch"
    work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
        interval_ids=(7, 8),
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    manifest = persist_residual_field_shard_checkpoint(
        work_unit,
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        contribution_reciprocal_points=9,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path),
        scratch_root=str(scratch_root),
        quiet_logs=True,
    )

    manifests = discover_residual_field_shard_manifests(
        output_dir=str(tmp_path),
        chunk_id=3,
        parameter_digest="abc123",
    )
    assert len(manifests) == 1
    assert manifest.contributing_interval_ids == (7, 8)
    assert manifest.scratch_root == str(scratch_root)
    assert Path(manifest.artifacts[0].path).exists()
    assert Path(manifest.artifacts[1].path).exists()
    scratch_shard_dir = scratch_root / "residual_checkpoints" / "chunk_3"
    assert list(scratch_shard_dir.glob("*.npz")) == []
    assert list(scratch_shard_dir.glob("*.hdf5")) == []


def test_residual_field_reducer_is_single_writer_for_final_chunk_artifacts(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit_1 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        work_unit_2 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_2,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )

        persist_residual_field_shard_checkpoint(
            work_unit_1,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )
        persist_residual_field_shard_checkpoint(
            work_unit_2,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        manifest = reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        assert manifest is not None
        assert manifest.completion_status is CompletionStatus.COMMITTED
        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([4 + 0j, 6 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j]))
        assert nrec == 12
        np.testing.assert_allclose(shape_nd, np.array([[2]]))
        assert applied == {interval_id_1, interval_id_2}
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_residual_field_backend_round_trip_preserves_reducer_progress(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    backend = build_residual_field_reducer_backend("durable_shared_restartable")
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        partial_1 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_1,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )
        partial_2 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_2,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial_1,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=2,
        )
        backend.accept_partial(
            partial_2,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=2,
        )

        manifest = backend.finalize_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="delete_reclaimable",
            scratch_root=scratch_root,
            quiet_logs=True,
        )

        progress = backend.load_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        deleted = backend.cleanup_reclaimable_shards(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            db_path=db.db_path,
        )

        assert manifest is not None
        assert progress is not None
        assert progress.completion_status is CompletionStatus.COMMITTED
        assert progress.incorporated_interval_ids == (interval_id_1, interval_id_2)
        assert deleted == ()
        durable_shard_dir = tmp_path / "residual_checkpoints" / "chunk_3"
        assert durable_shard_dir.exists()
    finally:
        db.close()


def test_local_backend_accept_partial_snapshots_without_shard_per_batch(tmp_path, monkeypatch):
    # These pin the SYNCHRONOUS snapshot contract (file exists right
    # after the cadence fold); the async writer has its own suite in
    # test_async_snapshot_writer.py.
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "0")
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    backend = build_residual_field_reducer_backend("local_restartable")
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        partial_1 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_1,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )
        partial_2 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_2,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial_1,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )
        assert discover_residual_field_shard_manifests(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            shard_storage_root=scratch_root,
        ) == []

        backend.accept_partial(
            partial_2,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )

        snapshot_path = build_local_accumulator_snapshot_path(
            str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            snapshot_seq=1,
        )
        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )

        assert snapshot_path.exists()
        assert progress is not None
        assert progress.completion_status is CompletionStatus.MATERIALIZED
        assert progress.durable_truth_unit == "committed_local_snapshot_generation"
        assert progress.incorporated_interval_ids == (interval_id_1, interval_id_2)
        assert db.get_unsaved_interval_chunks() == []
        assert discover_residual_field_shard_manifests(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            shard_storage_root=scratch_root,
        ) == []
    finally:
        db.close()


def test_local_backend_restores_from_snapshot_and_finalizes(tmp_path, monkeypatch):
    # These pin the SYNCHRONOUS snapshot contract (file exists right
    # after the cadence fold); the async writer has its own suite in
    # test_async_snapshot_writer.py.
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "0")
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("local_restartable")
        partial_1 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_1,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )
        backend.accept_partial(
            partial_1,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=1,
        )

        restarted_backend = build_residual_field_reducer_backend("local_restartable")
        partial_2 = restarted_backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_2,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
        )
        restarted_backend.accept_partial(
            partial_2,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=4,
        )
        manifest = restarted_backend.finalize_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
        )

        store = ResidualFieldArtifactStore(str(tmp_path))
        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )

        assert manifest is not None
        np.testing.assert_allclose(current[:, 1], np.array([4 + 0j, 6 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j]))
        assert nrec == 12
        np.testing.assert_allclose(shape_nd, np.array([[2]]))
        assert progress is not None
        assert progress.completion_status is CompletionStatus.COMMITTED
    finally:
        db.close()


def test_local_backend_accepts_direct_local_contributions_without_partial_wrapper(tmp_path, monkeypatch):
    # These pin the SYNCHRONOUS snapshot contract (file exists right
    # after the cadence fold); the async writer has its own suite in
    # test_async_snapshot_writer.py.
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "0")
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    backend = build_residual_field_reducer_backend("local_restartable")
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit_1 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        work_unit_2 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_2,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )

        backend.accept_local_contribution(
            work_unit_1,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )
        backend.accept_local_contribution(
            work_unit_2,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )

        snapshot_path = build_local_accumulator_snapshot_path(
            str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            snapshot_seq=1,
        )
        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )

        assert snapshot_path.exists()
        assert progress is not None
        assert progress.incorporated_interval_ids == (interval_id_1, interval_id_2)
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_local_backend_target_flush_and_process_local_inspection_helpers(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    template_backend = build_residual_field_reducer_backend("local_restartable")
    backend = get_process_local_residual_field_backend(template_backend)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id, _ = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=7, point_start=0, point_stop=1)

        backend.accept_local_contribution(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )

        before = inspect_process_local_residual_reducer_target(
            template_backend,
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
        )
        flushed = flush_process_local_residual_reducer_target(
            template_backend,
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
            db_path=db.db_path,
        )
        after = inspect_process_local_residual_reducer_target(
            template_backend,
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
        )

        assert before is not None
        assert before["has_live_accumulator"] is True
        assert before["live_dirty"] is True
        assert before["durable_snapshot_seq"] == 0
        assert flushed is True
        assert flush_process_local_residual_reducer_target(
            template_backend,
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
            db_path=db.db_path,
        ) is False
        assert after is not None
        assert after["has_live_accumulator"] is True
        assert after["live_dirty"] is False
        assert after["durable_snapshot_seq"] == 1
        assert after["durable_interval_ids"] == (interval_id,)
        assert Path(str(after["durable_snapshot_path"])).exists()
    finally:
        db.close()


def test_local_backend_flush_keeps_and_reuses_accumulator(tmp_path):
    clear_process_local_residual_field_backends()
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    template_backend = build_residual_field_reducer_backend("local_restartable")
    backend = get_process_local_residual_field_backend(template_backend)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit_1 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=7, point_start=0, point_stop=1)
        work_unit_2 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_2,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=7, point_start=0, point_stop=1)

        common_kwargs = dict(
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )
        backend.accept_local_contribution(
            work_unit_1,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            **common_kwargs,
        )
        assert backend.flush_local_reducer_target(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
            db_path=db.db_path,
        ) is True
        key = (3, "abc123", 7)
        assert key in backend._local_accumulators
        accumulator = backend._local_accumulators[key]
        assert accumulator.durable_interval_ids == {interval_id_1}

        backend.accept_local_contribution(
            work_unit_2,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            **common_kwargs,
        )
        assert backend.flush_local_reducer_target(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
            db_path=db.db_path,
        ) is True
        assert backend._local_accumulators[key] is accumulator

        target_state = backend.inspect_local_reducer_target(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=7,
            output_dir=str(tmp_path),
            include_payload=True,
        )

        assert target_state is not None
        assert target_state["has_live_accumulator"] is True
        assert target_state["live_dirty"] is False
        assert target_state["durable_interval_ids"] == (interval_id_1, interval_id_2)
        payload = target_state["durable_snapshot_payload"]
        np.testing.assert_allclose(
            payload["amplitudes_delta"],
            np.array([4 + 0j, 6 + 0j]),
        )
        np.testing.assert_allclose(
            payload["amplitudes_average"],
            np.array([0.75 + 0j, 1.25 + 0j]),
        )
    finally:
        clear_process_local_residual_field_backends()
        db.close()


def test_live_local_accumulator_cleanup_closes_memmaps(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    accumulator = LiveLocalAccumulator.from_arrays(
        work_unit,
        point_ids=np.array([10, 11]),
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        scratch_root=str(tmp_path / "scratch"),
        max_ram_bytes=0,
    )
    delta_path = accumulator.live_dir / "amplitudes_delta.npy"
    average_path = accumulator.live_dir / "amplitudes_average.npy"
    delta_mmap = accumulator.amplitudes_delta._mmap
    average_mmap = accumulator.amplitudes_average._mmap

    assert isinstance(accumulator.amplitudes_delta, np.memmap)
    assert delta_path.exists()
    assert average_path.exists()

    accumulator.cleanup_live_files()

    assert accumulator.amplitudes_delta is None
    assert accumulator.amplitudes_average is None
    assert getattr(delta_mmap, "closed", True) is True
    assert getattr(average_mmap, "closed", True) is True
    assert not delta_path.exists()
    assert not average_path.exists()


def test_local_backend_default_accumulator_keeps_small_targets_in_ram(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id, _ = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("local_restartable")
        partial = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )

        accumulator = next(iter(backend._local_accumulators.values()))
        assert backend.local_accumulator_max_ram_bytes == 256 * 1024 * 1024
        assert accumulator.storage_mode == "ram"
        assert accumulator.live_dir is None
        assert not isinstance(accumulator.amplitudes_delta, np.memmap)
    finally:
        db.close()


def test_distributed_backend_inspect_reports_generation_paths_and_metrics(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id, _interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("durable_shared_restartable")
        partial = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=1,
        )

        target_state = backend.inspect_local_reducer_target(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=None,
            output_dir=str(tmp_path),
        )

        assert target_state is not None
        assert target_state["durable_snapshot_seq"] == 1
        assert target_state["durable_interval_ids"] == (interval_id,)
        assert Path(str(target_state["durable_snapshot_path"])).exists()
        assert Path(str(target_state["durable_snapshot_manifest_path"])).exists()
        assert target_state["checkpoint_metrics"]["total_checkpoint_writes"] == 1
        assert target_state["total_checkpoint_writes"] == 1
        assert target_state["total_checkpoint_bytes_written"] > 0
    finally:
        db.close()


def test_distributed_backend_local_intervals_already_durable_falls_back_to_interval_id(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id, _interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("durable_shared_restartable")
        partial = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=1,
        )

        fallback_work_unit = SimpleNamespace(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=None,
            interval_id=interval_id,
            interval_ids=(),
        )

        assert backend.local_intervals_already_durable(
            fallback_work_unit,
            output_dir=str(tmp_path),
        ) is True
    finally:
        db.close()


def test_distributed_backend_inspect_and_discovery_prefer_latest_generation(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("durable_shared_restartable")
        partial_1 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_1,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )
        partial_2 = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id_2,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial_1,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=4,
        )
        backend.accept_partial(
            partial_2,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=4,
        )

        target_state = backend.inspect_local_reducer_target(
            chunk_id=3,
            parameter_digest="abc123",
            partition_id=None,
            output_dir=str(tmp_path),
        )
        latest_generations = [
            manifest
            for manifest in discover_residual_field_shard_manifests(
                output_dir=str(tmp_path),
                chunk_id=3,
                parameter_digest="abc123",
            )
            if "generation" in manifest.artifact_key
        ]
        stale_generations = discover_stale_residual_field_generation_manifests(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )

        assert target_state is not None
        assert target_state["durable_snapshot_seq"] == 2
        assert target_state["durable_interval_ids"] == (interval_id_1, interval_id_2)
        assert target_state["total_checkpoint_writes"] == 2
        assert len(latest_generations) == 1
        assert len(stale_generations) == 1
    finally:
        db.close()


def test_local_backend_finalize_chunk_concatenates_disjoint_partition_snapshots(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("local_restartable")
        # Partitions split disjoint real-space atom ranges, but every
        # partition accumulates every interval: finalize refuses to
        # concatenate partitions with mismatched interval coverage.
        partition_specs = {0: (0, 1), 1: (1, 2)}
        interval_payloads = {
            interval_id_1: {
                0: (5, np.array([1 + 0j, 2 + 0j]), np.array([0.5 + 0j, 0.75 + 0j])),
                1: (5, np.array([3 + 0j, 4 + 0j]), np.array([0.25 + 0j, 0.5 + 0j])),
            },
            interval_id_2: {
                0: (7, np.array([10 + 0j, 20 + 0j]), np.array([0.25 + 0j, 0.5 + 0j])),
                1: (7, np.array([30 + 0j, 40 + 0j]), np.array([0.75 + 0j, 1.0 + 0j])),
            },
        }
        for interval_id, per_partition in interval_payloads.items():
            for partition_id, (nrec_contrib, delta, average) in per_partition.items():
                point_start, point_stop = partition_specs[partition_id]
                partial = backend.build_local_partial(
                    ResidualFieldWorkUnit.interval_chunk(
                        interval_id=interval_id,
                        chunk_id=3,
                        parameter_digest="abc123",
                        output_dir=str(tmp_path),
                    ).with_partition(
                        partition_id=partition_id,
                        point_start=point_start,
                        point_stop=point_stop,
                    ),
                    grid_shape_nd=np.array([[2]]),
                    total_reciprocal_points=11,
                    contribution_reciprocal_points=nrec_contrib,
                    amplitudes_delta=delta,
                    amplitudes_average=average,
                    point_ids=np.array([0, 1]),
                )
                backend.accept_partial(
                    partial,
                    output_dir=str(tmp_path),
                    scratch_root=scratch_root,
                    db_path=db.db_path,
                    total_expected_partials=2,
                )

        manifest = backend.finalize_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
        )

        assert manifest is not None
        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([11 + 0j, 22 + 0j, 33 + 0j, 44 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j, 1.0 + 0j, 1.5 + 0j]))
        assert nrec == 24
        np.testing.assert_allclose(shape_nd, np.array([[2], [2]]))
        assert applied == {interval_id_1, interval_id_2}
        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        assert progress is not None
        assert progress.completion_status is CompletionStatus.COMMITTED
        assert set(progress.incorporated_shard_keys)
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_local_backend_repairs_db_from_committed_progress_without_snapshots(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend("local_restartable")
        partition_specs = {0: (0, 1), 1: (1, 2)}
        for interval_id, nrec_contrib in ((interval_id_1, 5), (interval_id_2, 7)):
            for partition_id, (point_start, point_stop) in partition_specs.items():
                partial = backend.build_local_partial(
                    ResidualFieldWorkUnit.interval_chunk(
                        interval_id=interval_id,
                        chunk_id=3,
                        parameter_digest="abc123",
                        output_dir=str(tmp_path),
                    ).with_partition(
                        partition_id=partition_id,
                        point_start=point_start,
                        point_stop=point_stop,
                    ),
                    grid_shape_nd=np.array([[2]]),
                    total_reciprocal_points=11,
                    contribution_reciprocal_points=nrec_contrib,
                    amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
                    amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
                    point_ids=np.array([0, 1]),
                )
                backend.accept_partial(
                    partial,
                    output_dir=str(tmp_path),
                    scratch_root=scratch_root,
                    db_path=db.db_path,
                    total_expected_partials=2,
                )
        first_manifest = backend.finalize_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
        )
        assert first_manifest is not None

        db.update_interval_chunk_status(interval_id_1, 3, saved=0)
        db.update_interval_chunk_status(interval_id_2, 3, saved=0)
        assert db.get_unsaved_interval_chunks() == [(interval_id_1, 3), (interval_id_2, 3)]
        assert list((tmp_path / "residual_checkpoints" / "chunk_3").glob("local_accumulator*.npz")) == []

        repaired_manifest = backend.finalize_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
        )

        assert repaired_manifest is not None
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_local_backend_file_backed_accumulator_path_works(tmp_path, monkeypatch):
    # These pin the SYNCHRONOUS snapshot contract (file exists right
    # after the cadence fold); the async writer has its own suite in
    # test_async_snapshot_writer.py.
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "0")
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id, _ = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend(
            "local_restartable",
            local_accumulator_max_ram_bytes=1,
        )
        partial = backend.build_local_partial(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=interval_id,
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
            ),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
        )

        backend.accept_partial(
            partial,
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=1,
        )

        live_dir = (
            Path(scratch_root)
            / "residual_accumulators"
            / "chunk_3"
            / "params_abc123"
        )
        snapshot_path = build_local_accumulator_snapshot_path(
            str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            snapshot_seq=1,
        )

        assert live_dir.exists()
        assert any(live_dir.glob("*.npy"))
        assert snapshot_path.exists()
        assert backend._local_accumulators
    finally:
        db.close()


def test_local_backend_file_backed_partition_targets_use_distinct_live_dirs(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    scratch_root = str(tmp_path / "scratch")
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        backend = build_residual_field_reducer_backend(
            "local_restartable",
            local_accumulator_max_ram_bytes=1,
        )
        work_unit_1 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=0, point_start=0, point_stop=1)
        work_unit_2 = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_2,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=1, point_start=1, point_stop=2)

        backend.accept_local_contribution(
            work_unit_1,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([0, 1]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )
        backend.accept_local_contribution(
            work_unit_2,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=7,
            amplitudes_delta=np.array([3 + 0j, 4 + 0j]),
            amplitudes_average=np.array([0.25 + 0j, 0.5 + 0j]),
            point_ids=np.array([0, 1]),
            output_dir=str(tmp_path),
            scratch_root=scratch_root,
            db_path=db.db_path,
            total_expected_partials=8,
        )

        live_dir_1 = (
            Path(scratch_root)
            / "residual_accumulators"
            / "chunk_3"
            / "params_abc123_partition_0"
        )
        live_dir_2 = (
            Path(scratch_root)
            / "residual_accumulators"
            / "chunk_3"
            / "params_abc123_partition_1"
        )

        assert live_dir_1.exists()
        assert live_dir_2.exists()
        assert live_dir_1 != live_dir_2
        assert any(live_dir_1.glob("*.npy"))
        assert any(live_dir_2.glob("*.npy"))
    finally:
        db.close()


def test_durable_residual_field_backend_uses_shared_shard_root_override(tmp_path):
    backend = build_residual_field_reducer_backend(
        "durable_shared_restartable",
        shard_storage_root_override=str(tmp_path / "shared-shards"),
    )
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path / "out"),
    )

    manifest = backend.persist_shard_checkpoint(
        work_unit,
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        contribution_reciprocal_points=5,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path / "out"),
        scratch_root=str(tmp_path / "scratch"),
        quiet_logs=True,
    )

    assert all(
        Path(str(artifact.path)).is_relative_to(tmp_path / "shared-shards")
        for artifact in manifest.artifacts
        if artifact.path is not None
    )


def test_residual_field_reducer_progress_and_reclaimability_are_persisted(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persisted_shard = persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="delete_reclaimable",
            quiet_logs=True,
        )

        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        reclaimable = list_reclaimable_residual_field_shards(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )

        assert progress is not None
        assert progress.incorporated_shard_keys == (persisted_shard.artifact_key,)
        assert progress.incorporated_interval_ids == (interval_id_1, interval_id_2)
        assert progress.reclaimable_shard_keys == (persisted_shard.artifact_key,)
        assert is_residual_field_shard_reclaimable(
            persisted_shard,
            output_dir=str(tmp_path),
            db_path=db.db_path,
        )
        assert [manifest.artifact_key for manifest in reclaimable] == [persisted_shard.artifact_key]
    finally:
        db.close()


def test_stage2_replacement_residual_reduce_requires_expected_coverage(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        with pytest.raises(RuntimeError, match="missing expected replacement shard coverage"):
            reduce_residual_field_shards_for_chunk(
                chunk_id=3,
                parameter_digest="abc123",
                expected_interval_ids=(interval_id_1, interval_id_2),
                output_dir=str(tmp_path),
                db_path=db.db_path,
                quiet_logs=True,
            )

        assert set(db.get_unsaved_interval_chunks()) == {
            (interval_id_1, 3),
            (interval_id_2, 3),
        }
    finally:
        db.close()


def test_stage2_replacement_residual_reduce_commits_canonical_outputs(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            expected_interval_ids=(interval_id_1, interval_id_2),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            quiet_logs=True,
        )

        assert (tmp_path / "residual_chunk_3_amplitudes.hdf5").exists()
        assert not (tmp_path / "point_data_chunk_3_amplitudes.hdf5").exists()
        assert is_residual_field_replacement_complete(
            chunk_id=3,
            parameter_digest="abc123",
            expected_interval_ids=(interval_id_1, interval_id_2),
            output_dir=str(tmp_path),
            db_path=db.db_path,
        )
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_stage2_replacement_commits_progress_before_sqlite_mark_saved(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        class FailingDb:
            def update_interval_chunk_status(self, *args, **kwargs):
                raise RuntimeError("sqlite unavailable")

            def close(self):
                return None

        with pytest.raises(RuntimeError, match="sqlite unavailable"):
            reduce_residual_field_shards_for_chunk(
                chunk_id=3,
                parameter_digest="abc123",
                expected_interval_ids=(interval_id_1, interval_id_2),
                output_dir=str(tmp_path),
                db_path=db.db_path,
                db_manager_factory=lambda path: FailingDb(),
                quiet_logs=True,
            )

        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        assert progress is not None
        assert progress.completion_status is CompletionStatus.COMMITTED
        assert set(db.get_unsaved_interval_chunks()) == {
            (interval_id_1, 3),
            (interval_id_2, 3),
        }

        assert is_residual_field_replacement_complete(
            chunk_id=3,
            parameter_digest="abc123",
            expected_interval_ids=(interval_id_1, interval_id_2),
            output_dir=str(tmp_path),
            db_path=db.db_path,
        )
        assert db.get_unsaved_interval_chunks() == []
    finally:
        db.close()


def test_residual_field_reducer_resume_prefers_progress_manifest(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="delete_reclaimable",
            quiet_logs=True,
        )

        store.save_applied_interval_ids(3, set())
        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="delete_reclaimable",
            quiet_logs=True,
        )

        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        np.testing.assert_allclose(current[:, 1], np.array([4 + 0j, 6 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j]))
        assert nrec == 12
        np.testing.assert_allclose(shape_nd, np.array([[2]]))
    finally:
        db.close()


def test_residual_field_reclaimable_shard_cleanup_is_opt_in_and_safe(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persisted_shard = persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        assert delete_reclaimable_residual_field_shards(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            db_path=db.db_path,
        ) == ()
        assert Path(persisted_shard.artifacts[0].path).exists()

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="delete_reclaimable",
            quiet_logs=True,
        )
        deleted = delete_reclaimable_residual_field_shards(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            db_path=db.db_path,
        )

        assert deleted == (persisted_shard.artifact_key,)
        assert not Path(persisted_shard.artifacts[0].path).exists()
        assert not Path(persisted_shard.artifacts[1].path).exists()
    finally:
        db.close()


def test_residual_field_reducer_reconciles_after_crash_before_committed_progress(
    tmp_path,
    monkeypatch,
):
    import core.residual_field.artifacts as artifacts_module

    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    real_write = write_residual_field_reducer_progress_manifest
    calls = {"count": 0}
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        def crash_after_pending(manifest):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("crash after final payload write, before committed progress")
            return real_write(manifest)

        monkeypatch.setattr(
            artifacts_module,
            "write_residual_field_reducer_progress_manifest",
            crash_after_pending,
        )

        with pytest.raises(RuntimeError, match="before committed progress"):
            reduce_residual_field_shards_for_chunk(
                chunk_id=3,
                parameter_digest="abc123",
                output_dir=str(tmp_path),
                db_path=db.db_path,
                cleanup_policy="delete_reclaimable",
                quiet_logs=True,
            )

        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        assert progress is not None
        assert progress.completion_status is CompletionStatus.MATERIALIZED
        assert progress.pending_shard_keys
        assert Path(progress.final_artifacts[0].path).exists()

        monkeypatch.setattr(
            artifacts_module,
            "write_residual_field_reducer_progress_manifest",
            real_write,
        )
        reconciled = reconcile_residual_field_reducer_progress(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
        )

        assert reconciled is not None
        assert reconciled.completion_status is CompletionStatus.COMMITTED
        assert reconciled.durable_truth_unit == "committed_shard_checkpoint"
        assert reconciled.pending_shard_keys == ()
        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        np.testing.assert_allclose(current[:, 1], np.array([4 + 0j, 6 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j]))
        assert nrec == 12
        np.testing.assert_allclose(shape_nd, np.array([[2]]))
    finally:
        db.close()


def test_residual_field_cleanup_policy_blocks_deletion_when_policy_is_off(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persisted_shard = persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            quiet_logs=True,
        )
        deleted = delete_reclaimable_residual_field_shards(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
            db_path=db.db_path,
        )

        assert deleted == ()
        assert Path(persisted_shard.artifacts[0].path).exists()
        progress = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        assert progress is not None
        assert progress.cleanup_policy == "off"
    finally:
        db.close()


def test_reduce_shards_blocks_when_pending_durable_coverage_is_missing(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id_1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persisted_shard = persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )
        progress = ResidualFieldReducerProgressManifest(
            artifact=build_residual_field_reducer_progress_artifact(
                str(tmp_path),
                chunk_id=3,
                parameter_digest="abc123",
            ),
            reducer_key=make_residual_field_reducer_key(
                chunk_id=3,
                parameter_digest="abc123",
            ),
            chunk_id=3,
            parameter_digest="abc123",
            completion_status=CompletionStatus.MATERIALIZED,
            durable_truth_unit="committed_shard_checkpoint",
            incorporated_shard_keys=(persisted_shard.artifact_key,),
            incorporated_interval_ids=(interval_id_1,),
            pending_shard_keys=("missing-shard",),
            pending_interval_ids=(interval_id_2,),
            reclaimable_shard_keys=(),
            final_artifacts=build_residual_field_output_artifact_refs(str(tmp_path), 3),
            cleanup_policy="off",
        )
        write_residual_field_reducer_progress_manifest(progress)

        result = reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            cleanup_policy="off",
            quiet_logs=True,
        )

        assert result is None
        progress_after = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest="abc123",
        )
        assert progress_after is not None
        assert progress_after.pending_shard_keys == ("missing-shard",)
        assert progress_after.durable_truth_unit == "committed_shard_checkpoint"
    finally:
        db.close()


def test_residual_field_batch_reducer_replay_remains_idempotent(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(interval_id_1, interval_id_2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=12,
            amplitudes_delta=np.array([4 + 0j, 6 + 0j]),
            amplitudes_average=np.array([0.75 + 0j, 1.25 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )
        reduce_residual_field_shards_for_chunk(
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([4 + 0j, 6 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.75 + 0j, 1.25 + 0j]))
        assert nrec == 12
        np.testing.assert_allclose(shape_nd, np.array([[2]]))
        assert applied == {interval_id_1, interval_id_2}
    finally:
        db.close()
