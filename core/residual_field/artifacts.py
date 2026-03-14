from __future__ import annotations

import logging

import numpy as np

from core.scattering.artifacts import ScatteringArtifactStore
from core.residual_field.accumulation import (
    build_existing_materialized_residual_field_state,
    build_materialized_residual_field_state,
    materialize_scattering_payload,
    merge_materialized_residual_field_states,
)
from core.residual_field.contracts import (
    RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    ResidualFieldArtifactManifest,
    ResidualFieldWorkUnit,
)
from core.contracts import ArtifactRef, CompletionStatus
from core.storage.database_manager import create_db_manager_for_thread


logger = logging.getLogger(__name__)


class _ResidualFieldChunkStatusUpdater:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def mark_saved(self, interval_id: int, chunk_id: int) -> None:
        db = create_db_manager_for_thread(self.db_path)
        try:
            db.update_interval_chunk_status(interval_id, chunk_id, saved=True)
        finally:
            db.close()


class ResidualFieldArtifactStore(ScatteringArtifactStore):
    """Current residual-field extraction reuses the existing chunk artifact layout."""


def build_residual_field_output_artifact_refs(
    output_dir: str,
    chunk_id: int,
) -> tuple[ArtifactRef, ...]:
    chunk_prefix = ResidualFieldArtifactStore(output_dir)._filename(chunk_id, "")
    base = chunk_prefix.rsplit(".", 1)[0]
    return (
        ArtifactRef(
            stage="residual_field",
            kind="chunk-amplitudes",
            key=f"residual-field:chunk-amplitudes:chunk-{chunk_id}",
            path=f"{base}_amplitudes.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-amplitudes-average",
            key=f"residual-field:chunk-amplitudes-average:chunk-{chunk_id}",
            path=f"{base}_amplitudes_av.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-grid-shape",
            key=f"residual-field:chunk-grid-shape:chunk-{chunk_id}",
            path=f"{base}_shapeNd.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-reciprocal-point-count",
            key=f"residual-field:chunk-reciprocal-point-count:chunk-{chunk_id}",
            path=f"{base}_amplitudes_nreciprocal_space_points.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-total-reciprocal-point-count",
            key=f"residual-field:chunk-total-reciprocal-point-count:chunk-{chunk_id}",
            path=f"{base}_amplitudes_ntotal_reciprocal_space_points.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-applied-interval-ids",
            key=f"residual-field:chunk-applied-interval-ids:chunk-{chunk_id}",
            path=f"{base}_applied_interval_ids.hdf5",
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
    )


def build_residual_field_chunk_manifest(
    work_unit: ResidualFieldWorkUnit,
    *,
    output_dir: str,
    completion_status: CompletionStatus,
) -> ResidualFieldArtifactManifest:
    return ResidualFieldArtifactManifest.from_work_unit(
        work_unit,
        artifacts=build_residual_field_output_artifact_refs(output_dir, work_unit.chunk_id),
        completion_status=completion_status,
        consumer_stage="decoding",
    )


def load_existing_materialized_state(
    chunk_id: int,
    *,
    output_dir: str,
    parameter_digest: str,
):
    store = ResidualFieldArtifactStore(output_dir)
    current, current_av, reciprocal_point_count, grid_shape_nd = store.load_chunk_payloads(chunk_id)
    applied_set = store.load_applied_interval_ids(chunk_id)
    if current is None or current_av is None:
        return None, applied_set, current, current_av
    state = build_existing_materialized_residual_field_state(
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        output_artifacts=build_residual_field_output_artifact_refs(output_dir, chunk_id),
        amplitudes_payload=current,
        amplitudes_average_payload=current_av,
        grid_shape_nd=grid_shape_nd,
        reciprocal_point_count=reciprocal_point_count,
        applied_interval_ids=tuple(sorted(applied_set)),
    )
    return state, applied_set, current, current_av


def persist_residual_field_chunk_result(
    work_unit: ResidualFieldWorkUnit,
    *,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    contribution_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    point_ids: np.ndarray | None = None,
    output_dir: str,
    db_path: str,
    quiet_logs: bool = False,
) -> ResidualFieldArtifactManifest:
    if work_unit.interval_id is None:
        raise ValueError("Residual-field chunk persistence requires interval_id.")

    store = ResidualFieldArtifactStore(output_dir)
    store.ensure_grid_shape(work_unit.chunk_id, grid_shape_nd)
    store.ensure_total_reciprocal_points(work_unit.chunk_id, total_reciprocal_points)

    existing_state, applied_set, current_payload, current_average_payload = (
        load_existing_materialized_state(
            work_unit.chunk_id,
            output_dir=output_dir,
            parameter_digest=work_unit.parameter_digest,
        )
    )
    already_applied = work_unit.interval_id in applied_set

    if not already_applied:
        effective_point_ids = (
            existing_state.payload.point_ids if existing_state is not None else point_ids
        )
        new_state = build_materialized_residual_field_state(
            work_unit,
            output_artifacts=build_residual_field_output_artifact_refs(output_dir, work_unit.chunk_id),
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            grid_shape_nd=grid_shape_nd,
            reciprocal_point_count=contribution_reciprocal_points,
            point_ids=effective_point_ids,
        )
        merged_state = (
            merge_materialized_residual_field_states(existing_state, new_state)
            if existing_state is not None
            else new_state
        )
        amplitudes_payload = materialize_scattering_payload(
            current_payload,
            merged_state.payload.point_ids,
            merged_state.payload.amplitudes_delta,
        )
        amplitudes_average_payload = materialize_scattering_payload(
            current_average_payload,
            merged_state.payload.point_ids,
            merged_state.payload.amplitudes_average,
        )
        store.save_chunk_payloads(
            work_unit.chunk_id,
            amplitudes_payload=amplitudes_payload,
            amplitudes_average_payload=amplitudes_average_payload,
            reciprocal_point_count=merged_state.payload.reciprocal_point_count,
        )
        applied_set.add(work_unit.interval_id)
        store.save_applied_interval_ids(work_unit.chunk_id, applied_set)

    _ResidualFieldChunkStatusUpdater(db_path).mark_saved(work_unit.interval_id, work_unit.chunk_id)
    manifest = build_residual_field_chunk_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )

    if quiet_logs:
        logger.debug(
            "write-HDF5 | chunk %d | iv %d %s",
            work_unit.chunk_id,
            work_unit.interval_id,
            "already applied (idempotent skip)" if already_applied else "applied",
        )
    else:
        if already_applied:
            logger.info(
                "write-HDF5 | chunk %d | iv %d already applied (idempotent skip)",
                work_unit.chunk_id,
                work_unit.interval_id,
            )
        else:
            logger.info(
                "write-HDF5 | chunk %d | iv %d applied",
                work_unit.chunk_id,
                work_unit.interval_id,
            )

    return manifest


load_existing_residual_field_partial_result = load_existing_materialized_state
persist_residual_field_interval_chunk_result = persist_residual_field_chunk_result


__all__ = [
    "ResidualFieldArtifactStore",
    "build_residual_field_chunk_manifest",
    "build_residual_field_output_artifact_refs",
    "load_existing_residual_field_partial_result",
    "load_existing_materialized_state",
    "persist_residual_field_chunk_result",
    "persist_residual_field_interval_chunk_result",
]
