from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from core.scattering.artifacts import ScatteringArtifactStore
from core.residual_field.accumulation import (
    build_existing_materialized_residual_field_state,
    build_materialized_residual_field_state,
    build_materialized_residual_field_state_from_shard,
    materialize_scattering_payload,
    merge_materialized_residual_field_states,
)
from core.residual_field.contracts import (
    RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA,
    RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA,
    ResidualFieldArtifactManifest,
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
    build_residual_field_output_artifacts,
    build_residual_field_shard_artifacts,
    make_residual_field_artifact_key,
    make_residual_field_reducer_key,
    validate_residual_field_artifact_manifest,
    validate_residual_field_reducer_progress_manifest,
    validate_residual_field_shard_manifest,
)
from core.contracts import (
    ArtifactManifestAssessment,
    ArtifactRef,
    CompletionStatus,
    RetryDisposition,
    RetryIdempotencySemantics,
)
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


def build_residual_field_reducer_progress_artifact(
    output_dir: str,
    *,
    chunk_id: int,
    parameter_digest: str,
) -> ArtifactRef:
    shard_dir = Path(output_dir) / "residual_shards" / f"chunk_{chunk_id}"
    return ArtifactRef(
        stage="residual_field",
        kind="residual-reducer-progress-manifest",
        key=make_residual_field_artifact_key(
            "residual-reducer-progress-manifest",
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        path=str(shard_dir / f"reducer_progress_params_{parameter_digest}.manifest.json"),
        schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    )


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
        artifact_schema_name=RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.name,
    )
    validate_residual_field_artifact_manifest(manifest)
    return manifest


def build_residual_field_shard_manifest(
    work_unit: ResidualFieldWorkUnit,
    *,
    output_dir: str,
    completion_status: CompletionStatus,
    point_count: int,
    contribution_reciprocal_point_count: int,
    total_reciprocal_point_count: int,
    shard_storage_root: str | None = None,
) -> ResidualFieldShardManifest:
    manifest = ResidualFieldShardManifest.from_work_unit(
        work_unit,
        artifacts=build_residual_field_shard_artifacts(
            output_dir,
            chunk_id=work_unit.chunk_id,
            interval_ids=tuple(work_unit.interval_ids or (int(work_unit.interval_id),)),
            parameter_digest=work_unit.parameter_digest,
            shard_storage_root=shard_storage_root,
        ),
        completion_status=completion_status,
        point_count=point_count,
        contribution_reciprocal_point_count=contribution_reciprocal_point_count,
        total_reciprocal_point_count=total_reciprocal_point_count,
    )
    validate_residual_field_shard_manifest(manifest)
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


def _residual_field_shard_manifest_to_payload(
    manifest: ResidualFieldShardManifest,
) -> dict[str, object]:
    return {
        "artifact_key": manifest.artifact_key,
        "completion_status": manifest.completion_status.value,
        "retry": {
            "failure_unit": manifest.retry.failure_unit,
            "retry_unit": manifest.retry.retry_unit,
            "idempotency_key": manifest.retry.idempotency_key,
            "replay_disposition": manifest.retry.replay_disposition.value,
            "crash_recovery_rule": manifest.retry.crash_recovery_rule,
        },
        "interval_id": manifest.interval_id,
        "contributing_interval_ids": list(manifest.contributing_interval_ids),
        "chunk_id": manifest.chunk_id,
        "parameter_digest": manifest.parameter_digest,
        "point_count": manifest.point_count,
        "contribution_reciprocal_point_count": manifest.contribution_reciprocal_point_count,
        "total_reciprocal_point_count": manifest.total_reciprocal_point_count,
        "scratch_root": manifest.scratch_root,
        "producer_stage": manifest.producer_stage,
        "consumer_stage": manifest.consumer_stage,
        "artifact_schema_name": manifest.artifact_schema_name,
        "schema_version": manifest.schema_version,
        "artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.artifacts
        ],
        "upstream_artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.upstream_artifacts
        ],
    }


def _residual_field_reducer_progress_manifest_to_payload(
    manifest: ResidualFieldReducerProgressManifest,
) -> dict[str, object]:
    return {
        "artifact": {
            "stage": manifest.artifact.stage,
            "kind": manifest.artifact.kind,
            "key": manifest.artifact.key,
            "path": manifest.artifact.path,
            "schema_version": manifest.artifact.schema_version,
        },
        "reducer_key": manifest.reducer_key,
        "chunk_id": manifest.chunk_id,
        "parameter_digest": manifest.parameter_digest,
        "completion_status": manifest.completion_status.value,
        "durable_truth_unit": manifest.durable_truth_unit,
        "incorporated_shard_keys": list(manifest.incorporated_shard_keys),
        "incorporated_interval_ids": list(manifest.incorporated_interval_ids),
        "pending_shard_keys": list(manifest.pending_shard_keys),
        "pending_interval_ids": list(manifest.pending_interval_ids),
        "reclaimable_shard_keys": list(manifest.reclaimable_shard_keys),
        "cleanup_policy": manifest.cleanup_policy,
        "final_artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.final_artifacts
        ],
        "schema_version": manifest.schema_version,
    }


def _artifact_ref_from_payload(payload: dict[str, object]) -> ArtifactRef:
    return ArtifactRef(
        stage=str(payload["stage"]),
        kind=str(payload["kind"]),
        key=str(payload["key"]),
        path=str(payload["path"]) if payload.get("path") is not None else None,
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )


def _write_json_atomic(target_path: Path, payload: dict[str, object]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f"{target_path.stem}_",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    Path(handle.name).replace(target_path)


def load_residual_field_shard_manifest(manifest_path: str | Path) -> ResidualFieldShardManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    retry_payload = payload["retry"]
    manifest = ResidualFieldShardManifest(
        artifact_key=str(payload["artifact_key"]),
        artifacts=tuple(_artifact_ref_from_payload(item) for item in payload["artifacts"]),
        completion_status=CompletionStatus(str(payload["completion_status"])),
        retry=RetryIdempotencySemantics(
            failure_unit=str(retry_payload["failure_unit"]),
            retry_unit=str(retry_payload["retry_unit"]),
            idempotency_key=str(retry_payload["idempotency_key"]),
            replay_disposition=RetryDisposition(str(retry_payload["replay_disposition"])),
            crash_recovery_rule=str(retry_payload["crash_recovery_rule"]),
        ),
        interval_id=int(payload["interval_id"]),
        contributing_interval_ids=tuple(
            int(interval_id)
            for interval_id in payload.get(
                "contributing_interval_ids",
                [payload["interval_id"]],
            )
        ),
        chunk_id=int(payload["chunk_id"]),
        parameter_digest=str(payload["parameter_digest"]),
        point_count=int(payload["point_count"]),
        contribution_reciprocal_point_count=int(payload["contribution_reciprocal_point_count"]),
        total_reciprocal_point_count=int(payload["total_reciprocal_point_count"]),
        scratch_root=(
            str(payload["scratch_root"])
            if payload.get("scratch_root") is not None
            else None
        ),
        producer_stage=str(payload.get("producer_stage", "residual_field")),
        consumer_stage=payload.get("consumer_stage"),
        upstream_artifacts=tuple(
            _artifact_ref_from_payload(item) for item in payload.get("upstream_artifacts", [])
        ),
        artifact_schema_name=str(payload.get("artifact_schema_name", RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.name)),
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )
    validate_residual_field_shard_manifest(manifest)
    return manifest


def _write_residual_field_shard_manifest_json(
    manifest: ResidualFieldShardManifest,
) -> None:
    manifest_ref = next(
        artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-manifest"
    )
    if manifest_ref.path is None:
        raise ValueError("Residual-field shard manifest path is required.")
    _write_json_atomic(
        Path(manifest_ref.path),
        _residual_field_shard_manifest_to_payload(manifest),
    )


def load_residual_field_reducer_progress_manifest(
    manifest_path: str | Path,
) -> ResidualFieldReducerProgressManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = ResidualFieldReducerProgressManifest(
        artifact=_artifact_ref_from_payload(payload["artifact"]),
        reducer_key=str(
            payload.get(
                "reducer_key",
                make_residual_field_reducer_key(
                    chunk_id=int(payload["chunk_id"]),
                    parameter_digest=str(payload["parameter_digest"]),
                ),
            )
        ),
        chunk_id=int(payload["chunk_id"]),
        parameter_digest=str(payload["parameter_digest"]),
        completion_status=CompletionStatus(str(payload["completion_status"])),
        durable_truth_unit=str(
            payload.get("durable_truth_unit", "committed_shard_checkpoint")
        ),
        incorporated_shard_keys=tuple(str(key) for key in payload["incorporated_shard_keys"]),
        incorporated_interval_ids=tuple(int(interval_id) for interval_id in payload["incorporated_interval_ids"]),
        pending_shard_keys=tuple(str(key) for key in payload.get("pending_shard_keys", [])),
        pending_interval_ids=tuple(int(interval_id) for interval_id in payload.get("pending_interval_ids", [])),
        reclaimable_shard_keys=tuple(str(key) for key in payload.get("reclaimable_shard_keys", [])),
        cleanup_policy=str(payload.get("cleanup_policy", "off")),
        final_artifacts=tuple(
            _artifact_ref_from_payload(item) for item in payload.get("final_artifacts", [])
        ),
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )
    validate_residual_field_reducer_progress_manifest(manifest)
    return manifest


def write_residual_field_reducer_progress_manifest(
    manifest: ResidualFieldReducerProgressManifest,
) -> ResidualFieldReducerProgressManifest:
    validate_residual_field_reducer_progress_manifest(manifest)
    if manifest.artifact.path is None:
        raise ValueError("Reducer progress manifest path is required.")
    _write_json_atomic(
        Path(manifest.artifact.path),
        _residual_field_reducer_progress_manifest_to_payload(manifest),
    )
    return manifest


def _build_residual_field_reducer_progress_manifest(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    completion_status: CompletionStatus,
    durable_truth_unit: str = "committed_shard_checkpoint",
    incorporated_shard_keys: tuple[str, ...],
    incorporated_interval_ids: tuple[int, ...],
    reclaimable_shard_keys: tuple[str, ...],
    final_artifacts: tuple[ArtifactRef, ...],
    pending_shard_keys: tuple[str, ...] = (),
    pending_interval_ids: tuple[int, ...] = (),
    cleanup_policy: str = "off",
) -> ResidualFieldReducerProgressManifest:
    return ResidualFieldReducerProgressManifest(
        artifact=build_residual_field_reducer_progress_artifact(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        reducer_key=make_residual_field_reducer_key(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        completion_status=completion_status,
        durable_truth_unit=str(durable_truth_unit),
        incorporated_shard_keys=tuple(sorted(set(str(key) for key in incorporated_shard_keys))),
        incorporated_interval_ids=tuple(
            sorted(set(int(interval_id) for interval_id in incorporated_interval_ids))
        ),
        reclaimable_shard_keys=tuple(sorted(set(str(key) for key in reclaimable_shard_keys))),
        final_artifacts=final_artifacts,
        pending_shard_keys=tuple(sorted(set(str(key) for key in pending_shard_keys))),
        pending_interval_ids=tuple(
            sorted(set(int(interval_id) for interval_id in pending_interval_ids))
        ),
        cleanup_policy=_normalize_residual_shard_cleanup_policy(cleanup_policy),
    )


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


def assess_residual_field_shard_manifest(
    manifest: ResidualFieldShardManifest,
) -> ArtifactManifestAssessment:
    validate_residual_field_shard_manifest(manifest)
    present_kinds = {artifact.kind for artifact in manifest.artifacts}
    missing_kinds = tuple(
        kind
        for kind in RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.required_artifact_kinds
        if kind not in present_kinds
    )
    missing_paths = _missing_artifact_paths(manifest.artifacts)
    all_required_artifacts_present = not missing_kinds and not missing_paths
    committed_state_consistent = manifest.completion_status is CompletionStatus.COMMITTED
    is_complete = all_required_artifacts_present and committed_state_consistent
    can_resume = not is_complete
    detail = (
        "committed"
        if is_complete
        else RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.resume_rule
        if can_resume
        else RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.completeness_rule
    )
    return ArtifactManifestAssessment(
        schema=RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA,
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


def _build_reduced_state_from_shards(
    *,
    shard_manifests: list[ResidualFieldShardManifest],
    output_dir: str,
    chunk_id: int,
) -> tuple[object | None, int | None]:
    merged_state = None
    total_reciprocal_points: int | None = None
    for manifest in sorted(
        shard_manifests,
        key=lambda item: (tuple(item.contributing_interval_ids), item.artifact_key),
    ):
        shard_payload = load_residual_field_shard_payload(manifest)
        total_reciprocal_points = manifest.total_reciprocal_point_count
        new_state = build_materialized_residual_field_state_from_shard(
            manifest,
            output_artifacts=build_residual_field_output_artifact_refs(output_dir, chunk_id),
            point_ids=shard_payload["point_ids"],
            grid_shape_nd=shard_payload["grid_shape_nd"],
            amplitudes_delta=shard_payload["amplitudes_delta"],
            amplitudes_average=shard_payload["amplitudes_average"],
        )
        merged_state = (
            merge_materialized_residual_field_states(merged_state, new_state)
            if merged_state is not None
            else new_state
        )
    return merged_state, total_reciprocal_points


def _write_residual_field_chunk_state(
    *,
    store: ResidualFieldArtifactStore,
    chunk_id: int,
    merged_state,
    total_reciprocal_points: int | None,
    applied_set: set[int],
) -> None:
    store.ensure_grid_shape(chunk_id, np.asarray(merged_state.payload.grid_shape_nd))
    if total_reciprocal_points is not None:
        store.ensure_total_reciprocal_points(chunk_id, total_reciprocal_points)
    amplitudes_payload = materialize_scattering_payload(
        None,
        merged_state.payload.point_ids,
        merged_state.payload.amplitudes_delta,
    )
    amplitudes_average_payload = materialize_scattering_payload(
        None,
        merged_state.payload.point_ids,
        merged_state.payload.amplitudes_average,
    )
    store.save_chunk_payloads(
        chunk_id,
        amplitudes_payload=amplitudes_payload,
        amplitudes_average_payload=amplitudes_average_payload,
        reciprocal_point_count=merged_state.payload.reciprocal_point_count,
    )
    store.save_applied_interval_ids(chunk_id, applied_set)


def reconcile_residual_field_reducer_progress(
    *,
    chunk_id: int,
    parameter_digest: str,
    output_dir: str,
    db_path: str,
    manifests: list[ResidualFieldShardManifest] | None = None,
    shard_storage_root: str | None = None,
    artifact_store_factory: Callable[[str], ResidualFieldArtifactStore] = ResidualFieldArtifactStore,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ResidualFieldReducerProgressManifest | None:
    progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if progress is None or progress.completion_status is CompletionStatus.COMMITTED:
        return progress

    shard_manifests = _merge_residual_field_shard_manifests(
        manifests,
        discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=shard_storage_root,
        ),
    )
    shard_by_key = _shard_manifests_by_key(shard_manifests)
    target_shard_keys = tuple(
        sorted(set(progress.incorporated_shard_keys) | set(progress.pending_shard_keys))
    )
    target_shards = [
        shard_by_key[key]
        for key in target_shard_keys
        if key in shard_by_key and assess_residual_field_shard_manifest(shard_by_key[key]).is_complete
    ]
    missing_target_shard_keys = tuple(
        sorted(set(target_shard_keys) - {manifest.artifact_key for manifest in target_shards})
    )
    if missing_target_shard_keys:
        blocked_progress = _build_residual_field_reducer_progress_manifest(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            completion_status=CompletionStatus.MATERIALIZED,
            durable_truth_unit=progress.durable_truth_unit,
            incorporated_shard_keys=progress.incorporated_shard_keys,
            incorporated_interval_ids=progress.incorporated_interval_ids,
            reclaimable_shard_keys=progress.reclaimable_shard_keys,
            final_artifacts=progress.final_artifacts,
            pending_shard_keys=missing_target_shard_keys,
            pending_interval_ids=progress.pending_interval_ids,
            cleanup_policy=progress.cleanup_policy,
        )
        return write_residual_field_reducer_progress_manifest(blocked_progress)
    if not target_shards:
        return progress

    target_interval_ids = tuple(
        sorted(set(progress.incorporated_interval_ids) | set(progress.pending_interval_ids))
    )
    merged_state, total_reciprocal_points = _build_reduced_state_from_shards(
        shard_manifests=target_shards,
        output_dir=output_dir,
        chunk_id=chunk_id,
    )
    if merged_state is None:
        return progress

    with chunk_mutex(chunk_id):
        store = artifact_store_factory(output_dir)
        _write_residual_field_chunk_state(
            store=store,
            chunk_id=chunk_id,
            merged_state=merged_state,
            total_reciprocal_points=total_reciprocal_points,
            applied_set=set(int(interval_id) for interval_id in target_interval_ids),
        )

    status_updater = _ResidualFieldChunkStatusUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    )
    for interval_id in target_interval_ids:
        status_updater.mark_saved(int(interval_id), int(chunk_id))

    representative_interval_id = (
        max(int(interval_id) for interval_id in target_interval_ids)
        if target_interval_ids
        else int(target_shards[-1].interval_id)
    )
    final_manifest = build_residual_field_chunk_manifest(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=representative_interval_id,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
        ),
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
    final_assessment = assess_residual_field_manifest(
        final_manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    )
    reconciled_progress = _build_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        completion_status=(
            CompletionStatus.COMMITTED
            if final_assessment.is_complete
            else CompletionStatus.MATERIALIZED
        ),
        durable_truth_unit="committed_shard_checkpoint",
        incorporated_shard_keys=target_shard_keys,
        incorporated_interval_ids=target_interval_ids,
        reclaimable_shard_keys=(
            target_shard_keys if final_assessment.is_complete else ()
        ),
        final_artifacts=final_manifest.artifacts,
        pending_shard_keys=(),
        pending_interval_ids=(),
        cleanup_policy=progress.cleanup_policy,
    )
    return write_residual_field_reducer_progress_manifest(reconciled_progress)


def load_residual_field_shard_payload(
    manifest: ResidualFieldShardManifest,
) -> dict[str, np.ndarray]:
    shard_ref = next(
        artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-data"
    )
    if shard_ref.path is None:
        raise ValueError("Residual-field shard data path is required.")
    with np.load(shard_ref.path, allow_pickle=False) as data:
        return {
            "point_ids": np.asarray(data["point_ids"]),
            "grid_shape_nd": np.asarray(data["grid_shape_nd"]),
            "amplitudes_delta": np.asarray(data["amplitudes_delta"]),
            "amplitudes_average": np.asarray(data["amplitudes_average"]),
        }


def discover_residual_field_shard_manifests(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    shard_storage_root: str | None = None,
) -> list[ResidualFieldShardManifest]:
    shard_dir = Path(shard_storage_root or output_dir) / "residual_shards" / f"chunk_{chunk_id}"
    if not shard_dir.exists():
        return []
    manifests: list[ResidualFieldShardManifest] = []
    for path in sorted(shard_dir.glob(f"batch_*_params_{parameter_digest}.manifest.json")):
        manifests.append(load_residual_field_shard_manifest(path))
    return manifests


def _merge_residual_field_shard_manifests(
    *manifest_groups: list[ResidualFieldShardManifest] | tuple[ResidualFieldShardManifest, ...] | None,
) -> list[ResidualFieldShardManifest]:
    merged: dict[str, ResidualFieldShardManifest] = {}
    for manifest_group in manifest_groups:
        if not manifest_group:
            continue
        for manifest in manifest_group:
            merged[manifest.artifact_key] = manifest
    return [merged[key] for key in sorted(merged)]


def _shard_manifests_by_key(
    manifests: list[ResidualFieldShardManifest],
) -> dict[str, ResidualFieldShardManifest]:
    return {manifest.artifact_key: manifest for manifest in manifests}


def _normalize_residual_shard_cleanup_policy(policy: str | bool | None) -> str:
    if isinstance(policy, bool):
        return "delete_reclaimable" if policy else "off"
    normalized = str(policy or "off").strip().lower()
    if normalized in {"off", "false", "0", "keep"}:
        return "off"
    if normalized in {"delete_reclaimable", "cleanup", "on", "true", "1"}:
        return "delete_reclaimable"
    raise ValueError(
        "Residual-field cleanup policy must be 'off' or 'delete_reclaimable'."
    )


def discover_residual_field_reducer_progress_manifest(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
) -> ResidualFieldReducerProgressManifest | None:
    artifact = build_residual_field_reducer_progress_artifact(
        output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if artifact.path is None or not Path(artifact.path).exists():
        return None
    return load_residual_field_reducer_progress_manifest(artifact.path)


def is_residual_field_shard_reclaimable(
    manifest: ResidualFieldShardManifest,
    *,
    output_dir: str,
    db_path: str,
    progress_manifest: ResidualFieldReducerProgressManifest | None = None,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    progress = progress_manifest or discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=int(manifest.chunk_id),
        parameter_digest=manifest.parameter_digest,
    )
    if progress is None:
        return False
    if manifest.artifact_key not in set(progress.reclaimable_shard_keys):
        return False
    representative_interval_id = (
        max(int(interval_id) for interval_id in progress.incorporated_interval_ids)
        if progress.incorporated_interval_ids
        else int(manifest.interval_id)
    )
    final_manifest = build_residual_field_chunk_manifest(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=representative_interval_id,
            chunk_id=int(manifest.chunk_id),
            parameter_digest=manifest.parameter_digest,
            output_dir=output_dir,
        ),
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
    final_assessment = assess_residual_field_manifest(
        final_manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    )
    return (
        final_assessment.is_complete
        and set(int(interval_id) for interval_id in manifest.contributing_interval_ids).issubset(
            set(int(interval_id) for interval_id in progress.incorporated_interval_ids)
        )
    )


def list_reclaimable_residual_field_shards(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    shard_storage_root: str | None = None,
) -> list[ResidualFieldShardManifest]:
    progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if progress is None:
        return []
    reclaimable = set(progress.reclaimable_shard_keys)
    if not reclaimable:
        return []
    return [
        manifest
        for manifest in discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=shard_storage_root,
        )
        if manifest.artifact_key in reclaimable
    ]


def delete_reclaimable_residual_field_shards(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    db_path: str,
    manifests: list[ResidualFieldShardManifest] | None = None,
    shard_storage_root: str | None = None,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> tuple[str, ...]:
    progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if (
        progress is None
        or progress.completion_status is not CompletionStatus.COMMITTED
        or progress.cleanup_policy != "delete_reclaimable"
    ):
        return ()
    shard_manifests = _merge_residual_field_shard_manifests(
        manifests,
        discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=shard_storage_root,
        ),
    )
    deleted: list[str] = []
    for manifest in shard_manifests:
        if not is_residual_field_shard_reclaimable(
            manifest,
            output_dir=output_dir,
            db_path=db_path,
            progress_manifest=progress,
            db_manager_factory=db_manager_factory,
        ):
            continue
        for artifact in manifest.artifacts:
            if artifact.path is None:
                continue
            Path(artifact.path).unlink(missing_ok=True)
        deleted.append(manifest.artifact_key)
    shard_dir = Path(shard_storage_root or output_dir) / "residual_shards" / f"chunk_{chunk_id}"
    if shard_dir.exists() and not any(shard_dir.iterdir()):
        shard_dir.rmdir()
    return tuple(sorted(deleted))


def persist_residual_field_shard_checkpoint(
    work_unit: ResidualFieldWorkUnit,
    *,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    contribution_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    point_ids: np.ndarray | None = None,
    output_dir: str,
    scratch_root: str | None = None,
    shard_storage_root: str | None = None,
    compress: bool = True,
    quiet_logs: bool = False,
) -> ResidualFieldShardManifest:
    start_time = time.perf_counter()
    if work_unit.interval_id is None:
        raise ValueError("Residual-field shard checkpoint requires interval_id.")

    shard_artifacts = build_residual_field_shard_artifacts(
        output_dir,
        chunk_id=work_unit.chunk_id,
        interval_ids=tuple(work_unit.interval_ids or (work_unit.interval_id,)),
        parameter_digest=work_unit.parameter_digest,
        shard_storage_root=shard_storage_root,
    )
    manifest = build_residual_field_shard_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
        point_count=int(np.asarray(amplitudes_delta).reshape(-1).shape[0]),
        contribution_reciprocal_point_count=contribution_reciprocal_points,
        total_reciprocal_point_count=total_reciprocal_points,
        shard_storage_root=shard_storage_root,
    )
    manifest = ResidualFieldShardManifest(
        **{
            **manifest.__dict__,
            "scratch_root": str(Path(scratch_root).expanduser()) if scratch_root else None,
        }
    )
    assessment = assess_residual_field_shard_manifest(manifest)
    if assessment.is_complete:
        if quiet_logs:
            logger.debug(
                "write-shard | chunk %d | batch %s already committed (idempotent skip)",
                work_unit.chunk_id,
                ",".join(str(interval_id) for interval_id in manifest.contributing_interval_ids),
            )
        else:
            logger.info(
                "write-shard | chunk %d | batch %s already committed (idempotent skip)",
                work_unit.chunk_id,
                ",".join(str(interval_id) for interval_id in manifest.contributing_interval_ids),
            )
        return manifest

    shard_ref = next(
        artifact for artifact in shard_artifacts if artifact.kind == "residual-shard-data"
    )
    if shard_ref.path is None:
        raise ValueError("Residual-field shard data path is required.")
    shard_path = Path(shard_ref.path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = (
        Path(manifest.scratch_root).expanduser()
        / "residual_shards"
        / f"chunk_{work_unit.chunk_id}"
        if manifest.scratch_root
        else shard_path.parent
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=scratch_dir,
        prefix=f"{shard_path.stem}_",
        suffix=".npz",
        delete=False,
    ) as handle:
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(
            handle,
            interval_id=np.array([work_unit.interval_id], dtype=np.int64),
            contributing_interval_ids=np.asarray(manifest.contributing_interval_ids, dtype=np.int64),
            chunk_id=np.array([work_unit.chunk_id], dtype=np.int64),
            parameter_digest=np.array([work_unit.parameter_digest]),
            point_ids=np.asarray(point_ids if point_ids is not None else np.arange(np.asarray(amplitudes_delta).reshape(-1).shape[0], dtype=np.int64)),
            grid_shape_nd=np.asarray(grid_shape_nd),
            amplitudes_delta=np.asarray(amplitudes_delta),
            amplitudes_average=np.asarray(amplitudes_average),
            contribution_reciprocal_points=np.array([int(contribution_reciprocal_points)], dtype=np.int64),
            total_reciprocal_points=np.array([int(total_reciprocal_points)], dtype=np.int64),
        )
    scratch_path = Path(handle.name)
    if scratch_path.parent == shard_path.parent:
        scratch_path.replace(shard_path)
    else:
        with tempfile.NamedTemporaryFile(
            dir=shard_path.parent,
            prefix=f"{shard_path.stem}_commit_",
            suffix=".npz",
            delete=False,
        ) as durable_handle:
            durable_tmp_path = Path(durable_handle.name)
        shutil.copyfile(scratch_path, durable_tmp_path)
        durable_tmp_path.replace(shard_path)
        scratch_path.unlink(missing_ok=True)
    _write_residual_field_shard_manifest_json(manifest)
    shard_bytes = sum(
        int(Path(artifact.path).stat().st_size)
        for artifact in manifest.artifacts
        if artifact.path is not None and Path(artifact.path).exists()
    )

    if quiet_logs:
        logger.debug(
            "write-shard | chunk %d | batch %s committed | bytes=%d | duration=%.3fs",
            work_unit.chunk_id,
            ",".join(str(interval_id) for interval_id in manifest.contributing_interval_ids),
            shard_bytes,
            time.perf_counter() - start_time,
        )
    else:
        logger.info(
            "write-shard | chunk %d | batch %s committed | bytes=%d | duration=%.3fs",
            work_unit.chunk_id,
            ",".join(str(interval_id) for interval_id in manifest.contributing_interval_ids),
            shard_bytes,
            time.perf_counter() - start_time,
        )
    return manifest


def summarize_residual_field_shards(
    manifests: list[ResidualFieldShardManifest],
) -> dict[str, int]:
    shard_bytes = 0
    point_count = 0
    for manifest in manifests:
        point_count += int(manifest.point_count)
        for artifact in manifest.artifacts:
            if artifact.kind != "residual-shard-data" or artifact.path is None:
                continue
            path = Path(artifact.path)
            if path.exists():
                shard_bytes += int(path.stat().st_size)
    return {
        "committed_shard_count": int(len(manifests)),
        "committed_shard_bytes": int(shard_bytes),
        "committed_point_count": int(point_count),
    }


def summarize_residual_field_output_artifacts(
    artifacts: tuple[ArtifactRef, ...],
) -> dict[str, int]:
    total_bytes = 0
    for artifact in artifacts:
        if artifact.path is None:
            continue
        path = Path(artifact.path)
        if path.exists():
            total_bytes += int(path.stat().st_size)
    return {
        "final_artifact_count": int(len(artifacts)),
        "final_artifact_bytes": int(total_bytes),
    }


def reduce_residual_field_shards_for_chunk(
    *,
    chunk_id: int,
    parameter_digest: str,
    output_dir: str,
    db_path: str,
    manifests: list[ResidualFieldShardManifest] | None = None,
    cleanup_policy: str | bool | None = None,
    shard_storage_root: str | None = None,
    artifact_store_factory: Callable[[str], ResidualFieldArtifactStore] = ResidualFieldArtifactStore,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
    quiet_logs: bool = False,
) -> ResidualFieldArtifactManifest | None:
    start_time = time.perf_counter()
    shard_manifests = _merge_residual_field_shard_manifests(
        manifests,
        discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=shard_storage_root,
        ),
    )
    existing_progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    resolved_cleanup_policy = _normalize_residual_shard_cleanup_policy(
        cleanup_policy
        if cleanup_policy is not None
        else existing_progress.cleanup_policy if existing_progress is not None else "off"
    )
    if existing_progress is not None and existing_progress.completion_status is CompletionStatus.MATERIALIZED:
        existing_progress = reconcile_residual_field_reducer_progress(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=shard_manifests,
            shard_storage_root=shard_storage_root,
            artifact_store_factory=artifact_store_factory,
            db_manager_factory=db_manager_factory,
        )
    committed_shards = [
        manifest
        for manifest in shard_manifests
        if assess_residual_field_shard_manifest(manifest).is_complete
    ]
    if existing_progress is not None and existing_progress.pending_shard_keys:
        committed_shard_keys = {
            manifest.artifact_key for manifest in committed_shards
        }
        missing_pending = tuple(
            sorted(set(existing_progress.pending_shard_keys) - committed_shard_keys)
        )
        if missing_pending:
            if quiet_logs:
                logger.debug(
                    "reduce-shards | chunk %d blocked by missing durable shard coverage %s",
                    chunk_id,
                    missing_pending,
                )
            else:
                logger.warning(
                    "reduce-shards | chunk %d blocked by missing durable shard coverage %s",
                    chunk_id,
                    missing_pending,
                )
            return None
    if not committed_shards:
        return None
    shard_summary = summarize_residual_field_shards(committed_shards)

    with chunk_mutex(chunk_id):
        store = artifact_store_factory(output_dir)
        existing_state, applied_set, current_payload, current_average_payload = (
            load_existing_materialized_state(
                chunk_id,
                output_dir=output_dir,
                parameter_digest=parameter_digest,
            )
        )
        incorporated_shard_keys = set(
            existing_progress.incorporated_shard_keys if existing_progress is not None else ()
        )
        if not incorporated_shard_keys:
            incorporated_shard_keys.update(
                manifest.artifact_key
                for manifest in committed_shards
                if set(manifest.contributing_interval_ids).issubset(applied_set)
            )
        incorporated_interval_ids = set(
            existing_progress.incorporated_interval_ids if existing_progress is not None else ()
        )
        incorporated_interval_ids.update(int(interval_id) for interval_id in applied_set)
        reduced_interval_ids: list[int] = []
        reduced_shard_keys: list[str] = []
        total_reciprocal_points: int | None = None
        merged_state = existing_state
        final_artifacts = build_residual_field_output_artifact_refs(output_dir, chunk_id)

        for manifest in sorted(
            committed_shards,
            key=lambda item: (tuple(item.contributing_interval_ids), item.artifact_key),
        ):
            manifest_intervals = tuple(int(interval_id) for interval_id in manifest.contributing_interval_ids)
            if (
                manifest.artifact_key in incorporated_shard_keys
                or set(manifest_intervals).issubset(applied_set)
            ):
                incorporated_shard_keys.add(manifest.artifact_key)
                incorporated_interval_ids.update(manifest_intervals)
                continue
            shard_payload = load_residual_field_shard_payload(manifest)
            total_reciprocal_points = manifest.total_reciprocal_point_count
            new_state = build_materialized_residual_field_state_from_shard(
                manifest,
                output_artifacts=build_residual_field_output_artifact_refs(output_dir, chunk_id),
                point_ids=shard_payload["point_ids"],
                grid_shape_nd=shard_payload["grid_shape_nd"],
                amplitudes_delta=shard_payload["amplitudes_delta"],
                amplitudes_average=shard_payload["amplitudes_average"],
            )
            merged_state = (
                merge_materialized_residual_field_states(merged_state, new_state)
                if merged_state is not None
                else new_state
            )
            applied_set.update(manifest_intervals)
            incorporated_shard_keys.add(manifest.artifact_key)
            incorporated_interval_ids.update(manifest_intervals)
            reduced_interval_ids.extend(manifest_intervals)
            reduced_shard_keys.append(manifest.artifact_key)

        if merged_state is not None and reduced_shard_keys:
            pending_progress = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                completion_status=CompletionStatus.MATERIALIZED,
                durable_truth_unit="committed_shard_checkpoint",
                incorporated_shard_keys=tuple(sorted(set(existing_progress.incorporated_shard_keys))) if existing_progress is not None else (),
                incorporated_interval_ids=tuple(
                    sorted(set(existing_progress.incorporated_interval_ids))
                ) if existing_progress is not None else (),
                reclaimable_shard_keys=(
                    existing_progress.reclaimable_shard_keys if existing_progress is not None else ()
                ),
                final_artifacts=final_artifacts,
                pending_shard_keys=tuple(sorted(set(reduced_shard_keys))),
                pending_interval_ids=tuple(sorted(set(reduced_interval_ids))),
                cleanup_policy=resolved_cleanup_policy,
            )
            write_residual_field_reducer_progress_manifest(pending_progress)
            store.ensure_grid_shape(chunk_id, np.asarray(merged_state.payload.grid_shape_nd))
            if total_reciprocal_points is not None:
                store.ensure_total_reciprocal_points(chunk_id, total_reciprocal_points)
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
                chunk_id,
                amplitudes_payload=amplitudes_payload,
                amplitudes_average_payload=amplitudes_average_payload,
                reciprocal_point_count=merged_state.payload.reciprocal_point_count,
            )
            store.save_applied_interval_ids(chunk_id, applied_set)

    status_updater = _ResidualFieldChunkStatusUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    )
    for interval_id in sorted(set(reduced_interval_ids)):
        status_updater.mark_saved(interval_id, chunk_id)

    representative_interval_id = (
        max(int(interval_id) for interval_id in applied_set)
        if applied_set
        else int(committed_shards[-1].interval_id)
    )
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=representative_interval_id,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        output_dir=output_dir,
    )
    manifest = build_residual_field_chunk_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
    manifest_assessment = assess_residual_field_manifest(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    )
    reclaimable_shard_keys = tuple(
        sorted(
            manifest_item.artifact_key
            for manifest_item in committed_shards
            if set(int(interval_id) for interval_id in manifest_item.contributing_interval_ids).issubset(applied_set)
        )
    )
    progress_manifest = _build_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        completion_status=(
            CompletionStatus.COMMITTED
            if manifest_assessment.is_complete
            else CompletionStatus.MATERIALIZED
        ),
        durable_truth_unit="committed_shard_checkpoint",
        incorporated_shard_keys=tuple(sorted(incorporated_shard_keys)),
        incorporated_interval_ids=tuple(
            sorted(
                set(int(interval_id) for interval_id in applied_set)
                | set(int(interval_id) for interval_id in incorporated_interval_ids)
            )
        ),
        reclaimable_shard_keys=(
            reclaimable_shard_keys if manifest_assessment.is_complete else ()
        ),
        final_artifacts=manifest.artifacts,
        pending_shard_keys=(),
        pending_interval_ids=(),
        cleanup_policy=resolved_cleanup_policy,
    )
    write_residual_field_reducer_progress_manifest(progress_manifest)
    if quiet_logs:
        logger.debug(
            "reduce-shards | chunk %d | reduced %d shard(s) | committed_shards=%d | shard_bytes=%d | point_count=%d | duration=%.3fs",
            chunk_id,
            len(reduced_shard_keys),
            shard_summary["committed_shard_count"],
            shard_summary["committed_shard_bytes"],
            shard_summary["committed_point_count"],
            time.perf_counter() - start_time,
        )
    else:
        logger.info(
            "reduce-shards | chunk %d | reduced %d shard(s) | committed_shards=%d | shard_bytes=%d | point_count=%d | duration=%.3fs",
            chunk_id,
            len(reduced_shard_keys),
            shard_summary["committed_shard_count"],
            shard_summary["committed_shard_bytes"],
            shard_summary["committed_point_count"],
            time.perf_counter() - start_time,
        )
    return manifest


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
    "assess_residual_field_shard_manifest",
    "build_residual_field_chunk_manifest",
    "build_residual_field_output_artifact_refs",
    "build_residual_field_reducer_progress_artifact",
    "build_residual_field_shard_manifest",
    "can_resume_residual_field_work_unit",
    "delete_reclaimable_residual_field_shards",
    "discover_residual_field_reducer_progress_manifest",
    "discover_residual_field_shard_manifests",
    "is_residual_field_manifest_complete",
    "is_residual_field_shard_reclaimable",
    "list_reclaimable_residual_field_shards",
    "load_existing_residual_field_partial_result",
    "load_existing_materialized_state",
    "load_residual_field_reducer_progress_manifest",
    "load_residual_field_shard_manifest",
    "load_residual_field_shard_payload",
    "persist_residual_field_shard_checkpoint",
    "persist_residual_field_chunk_result",
    "persist_residual_field_interval_chunk_result",
    "reconcile_residual_field_reducer_progress",
    "reduce_residual_field_shards_for_chunk",
    "write_residual_field_reducer_progress_manifest",
]
