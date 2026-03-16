from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dask.distributed import Client, LocalCluster
import pytest

import core.residual_field.execution as residual_execution
import core.residual_field.tasks as residual_tasks
from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    parse_residual_field_generation_ref,
)
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.contracts import (
    ResidualFieldAccumulatorStatus,
    ResidualFieldWorkUnit,
)
from core.scattering.kernels import IntervalTask
from core.residual_field.planning import build_residual_field_parameter_digest
from core.storage.database_manager import DatabaseManager


pytestmark = pytest.mark.skipif(
    os.getenv("MOSAIC_ENABLE_DISTRIBUTED_TESTS") != "1",
    reason=(
        "Set MOSAIC_ENABLE_DISTRIBUTED_TESTS=1 to run bounded LocalCluster HPC "
        "integration coverage on a host where Dask LocalCluster startup is supported."
    ),
)


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
    db.insert_interval_chunk_status_batch(
        [(interval_ids[0], chunk_id, 0), (interval_ids[1], chunk_id, 0)]
    )
    return interval_ids[0], interval_ids[1]


def _workflow_parameters(*, cleanup_policy: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_info={
            "residual_field_reducer_backend": "durable_shared_restartable",
            "residual_shard_batch_size": 1,
            "residual_shard_cleanup_policy": cleanup_policy,
        },
        postprocessing_mode="displacement",
        supercell=np.array([1]),
        rspace_info={"mode": "displacement"},
    )


def _run_stage_once(
    *,
    tmp_path: Path,
    db: DatabaseManager,
    workflow_parameters,
    transient_interval_payloads: dict[int, IntervalTask],
) -> None:
    artifacts = SimpleNamespace(
        db_manager=db,
        padded_intervals=[{"h_range": (0.0, 1.0)}, {"h_range": (1.0, 2.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads=transient_interval_payloads,
    )
    structure = SimpleNamespace(supercell=np.array([1]))

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
        resources={"nufft": 1},
    )
    try:
        with Client(cluster) as client:
            run_residual_field_stage(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
    finally:
        cluster.close()


def test_distributed_durable_residual_field_owner_local_generations_restart_and_cleanup(
    tmp_path,
    monkeypatch,
):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    payload_types: list[type[object]] = []
    original_record = residual_execution._record_residual_task_result

    def _capture_task_result(*, payload, work_unit, manifests_by_chunk):
        payload_types.append(type(payload))
        return original_record(
            payload=payload,
            work_unit=work_unit,
            manifests_by_chunk=manifests_by_chunk,
        )

    monkeypatch.setattr(
        residual_tasks,
        "build_rifft_grid_for_chunk",
        lambda chunk_data: (
            np.array([[0.0]], dtype=np.float64),
            np.array([[1]], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        residual_tasks,
        "execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.asarray(kwargs["weights"], dtype=np.complex128),
    )
    monkeypatch.setattr(
        residual_execution,
        "_record_residual_task_result",
        _capture_task_result,
    )
    monkeypatch.setattr(
        residual_execution,
        "reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )

    try:
        interval_id_1, interval_id_2 = _seed_db_for_chunk(db)
        workflow_parameters = _workflow_parameters(cleanup_policy="off")
        parameter_digest = build_residual_field_parameter_digest(workflow_parameters)
        transient_interval_payloads = {
            interval_id_1: IntervalTask(
                interval_id_1,
                "All",
                np.array([[0.0]], dtype=np.float64),
                np.array([2.0 + 0.0j], dtype=np.complex128),
                np.array([1.0 + 0.0j], dtype=np.complex128),
            ),
            interval_id_2: IntervalTask(
                interval_id_2,
                "All",
                np.array([[0.0]], dtype=np.float64),
                np.array([4.0 + 0.0j], dtype=np.complex128),
                np.array([1.0 + 0.0j], dtype=np.complex128),
            ),
        }

        _run_stage_once(
            tmp_path=tmp_path,
            db=db,
            workflow_parameters=workflow_parameters,
            transient_interval_payloads=transient_interval_payloads,
        )

        assert payload_types
        assert all(payload_type is ResidualFieldAccumulatorStatus for payload_type in payload_types)

        current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current[:, 1], np.array([4.0 + 0.0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([2.0 + 0.0j]))
        assert nrec == 2
        np.testing.assert_allclose(shape_nd, np.array([[1]]))
        assert applied == {interval_id_1, interval_id_2}

        generation_manifests = discover_residual_field_shard_manifests(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest=parameter_digest,
            include_stale_generations=True,
        )
        assert generation_manifests
        assert all(parse_residual_field_generation_ref(manifest) is not None for manifest in generation_manifests)
        shard_dir = tmp_path / "residual_shards" / "chunk_3"
        assert sorted(path.name for path in shard_dir.glob("batch_*")) == []
        generation_paths_before = sorted(path.name for path in shard_dir.glob("generation_*"))
        assert generation_paths_before

        progress_before = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest=parameter_digest,
        )
        assert progress_before is not None
        assert progress_before.durable_truth_unit == "committed_local_snapshot_generation"
        assert progress_before.incorporated_interval_ids == (interval_id_1, interval_id_2)

        db.update_interval_chunk_status(interval_id_1, 3, 0)
        db.update_interval_chunk_status(interval_id_2, 3, 0)
        payload_types.clear()

        _run_stage_once(
            tmp_path=tmp_path,
            db=db,
            workflow_parameters=_workflow_parameters(cleanup_policy="delete_reclaimable"),
            transient_interval_payloads=transient_interval_payloads,
        )

        assert payload_types == []

        current_after, current_av_after, nrec_after, shape_after = store.load_chunk_payloads(3)
        applied_after = store.load_applied_interval_ids(3)
        np.testing.assert_allclose(current_after[:, 1], np.array([4.0 + 0.0j]))
        np.testing.assert_allclose(current_av_after[:, 1], np.array([2.0 + 0.0j]))
        assert nrec_after == 2
        np.testing.assert_allclose(shape_after, np.array([[1]]))
        assert applied_after == {interval_id_1, interval_id_2}

        assert sorted(path.name for path in shard_dir.glob("generation_*")) == []
        progress_after = discover_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=3,
            parameter_digest=parameter_digest,
        )
        assert progress_after is not None
        assert progress_after.cleanup_policy == "delete_reclaimable"
        assert progress_after.durable_truth_unit == "committed_local_snapshot_generation"
        assert progress_after.incorporated_interval_ids == (interval_id_1, interval_id_2)
    finally:
        db.close()
