from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from core.scattering.artifacts import ScatteringArtifactStore
from core.residual_field.accumulation import (
    build_existing_materialized_residual_field_state,
    build_materialized_residual_field_state,
    materialize_scattering_payload,
    merge_materialized_residual_field_states,
)
from core.residual_field.contracts import (
    RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA,
    RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    ResidualFieldArtifactManifest,
    ResidualFieldWorkUnit,
    build_residual_field_output_artifacts,
    validate_residual_field_artifact_manifest,
)
from core.contracts import ArtifactManifestAssessment, ArtifactRef, CompletionStatus
from core.runtime import chunk_mutex
from core.storage.database_manager import create_db_manager_for_thread


logger = logging.getLogger(__name__)


class _ResidualFieldChunkStatusUpdater:
    def __init__(
        self,
        db_path: str,
        *,
        db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
    ) -> None:
        self.db_path = db_path
        self.db_manager_factory = db_manager_factory

    def mark_saved(self, interval_id: int, chunk_id: int) -> None:
        db = self.db_manager_factory(self.db_path)
        try:
            db.update_interval_chunk_status(interval_id, chunk_id, saved=True)
        finally:
            db.close()

    def is_saved(self, interval_id: int, chunk_id: int) -> bool:
        db = self.db_manager_factory(self.db_path)
        try:
            return (int(interval_id), int(chunk_id)) not in {
                (int(iv), int(ch))
                for iv, ch in db.get_unsaved_interval_chunks()
            }
        finally:
            db.close()


class ResidualFieldArtifactStore(ScatteringArtifactStore):
    """Current residual-field extraction reuses the existing chunk artifact layout."""


def build_residual_field_output_artifact_refs(
    output_dir: str,
    chunk_id: int,
) -> tuple[ArtifactRef, ...]:
    return build_residual_field_output_artifacts(output_dir, chunk_id)


def build_residual_field_chunk_manifest(
    work_unit: ResidualFieldWorkUnit,
    *,
    output_dir: str,
    completion_status: CompletionStatus,
) -> ResidualFieldArtifactManifest:
    manifest = ResidualFieldArtifactManifest.from_work_unit(
        work_unit,
        artifacts=build_residual_field_output_artifact_refs(output_dir, work_unit.chunk_id),
        completion_status=completion_status,
        consumer_stage="decoding",
    )
    validate_residual_field_artifact_manifest(manifest)
    return manifest


def _missing_artifact_kinds(
    manifest: ResidualFieldArtifactManifest,
) -> tuple[str, ...]:
    present_kinds = {artifact.kind for artifact in manifest.artifacts}
    return tuple(
        kind
        for kind in RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.required_artifact_kinds
        if kind not in present_kinds
    )


def _missing_artifact_paths(artifacts: tuple[ArtifactRef, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for artifact in artifacts:
        if artifact.path is None or not Path(artifact.path).exists():
            missing.append(artifact.key)
    return tuple(sorted(missing))


def assess_residual_field_manifest(
    manifest: ResidualFieldArtifactManifest,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ArtifactManifestAssessment:
    validate_residual_field_artifact_manifest(manifest)
    missing_kinds = _missing_artifact_kinds(manifest)
    missing_paths = _missing_artifact_paths(manifest.artifacts)
    all_required_artifacts_present = not missing_kinds and not missing_paths
    upstream_paths_missing = _missing_artifact_paths(manifest.upstream_artifacts)
    applied_ids = ResidualFieldArtifactStore(
        str(Path(manifest.artifacts[0].path).parent)
    ).load_applied_interval_ids(manifest.chunk_id)
    committed_state_consistent = (
        manifest.interval_id is not None
        and _ResidualFieldChunkStatusUpdater(
            db_path,
            db_manager_factory=db_manager_factory,
        ).is_saved(
            manifest.interval_id,
            manifest.chunk_id,
        )
        and manifest.interval_id in applied_ids
    )
    is_complete = (
        all_required_artifacts_present
        and committed_state_consistent
        and manifest.completion_status is CompletionStatus.COMMITTED
    )
    can_resume = bool(not upstream_paths_missing) and not is_complete
    detail = (
        "committed"
        if is_complete
        else RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.resume_rule
        if can_resume
        else RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.completeness_rule
    )
    return ArtifactManifestAssessment(
        schema=RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA,
        artifact_key=manifest.artifact_key,
        completion_status=manifest.completion_status,
        missing_artifact_kinds=missing_kinds,
        missing_artifact_paths=missing_paths,
        all_required_artifacts_present=all_required_artifacts_present,
        committed_state_consistent=committed_state_consistent,
        is_complete=is_complete,
        can_resume=can_resume,
        detail=detail,
    )


def is_residual_field_manifest_complete(
    manifest: ResidualFieldArtifactManifest,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    return assess_residual_field_manifest(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    ).is_complete


def can_resume_residual_field_work_unit(
    work_unit: ResidualFieldWorkUnit,
    *,
    output_dir: str,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    manifest = build_residual_field_chunk_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
    return assess_residual_field_manifest(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    ).can_resume


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
    artifact_store_factory: Callable[[str], ResidualFieldArtifactStore] = ResidualFieldArtifactStore,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ResidualFieldArtifactManifest:
    if work_unit.interval_id is None:
        raise ValueError("Residual-field chunk persistence requires interval_id.")

    with chunk_mutex(work_unit.chunk_id):
        store = artifact_store_factory(output_dir)
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

    _ResidualFieldChunkStatusUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    ).mark_saved(work_unit.interval_id, work_unit.chunk_id)
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
    "assess_residual_field_manifest",
    "build_residual_field_chunk_manifest",
    "build_residual_field_output_artifact_refs",
    "can_resume_residual_field_work_unit",
    "is_residual_field_manifest_complete",
    "load_existing_residual_field_partial_result",
    "load_existing_materialized_state",
    "persist_residual_field_chunk_result",
    "persist_residual_field_interval_chunk_result",
]
