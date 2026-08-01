from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from core.scattering.artifacts import ScatteringArtifactStore
from core.residual_field.accumulation import (
    build_existing_materialized_residual_field_state,
    build_materialized_residual_field_state,
    build_materialized_residual_field_state_from_shard,
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
    validate_residual_field_artifact_manifest,
    validate_residual_field_shard_manifest,
)
from core.contracts import (
    ArtifactManifestAssessment,
    ArtifactRef,
    CompletionStatus,
)
from core.runtime import chunk_mutex
from core.storage.atomic import fsync_parent, fsync_path
from core.storage.database_manager import create_db_manager_for_thread

# ---------------------------------------------------------------------------
# Re-exports from extracted modules so all public names remain importable from
# core.residual_field.artifacts.
# ---------------------------------------------------------------------------
from core.residual_field.manifest_io import (  # noqa: E402
    _artifact_ref_from_payload,
    _build_residual_field_reducer_progress_manifest,
    _load_array_payload,
    _normalize_residual_shard_cleanup_policy,
    _residual_field_reducer_progress_manifest_to_payload,
    _residual_field_shard_manifest_to_payload,
    _write_json_atomic,
    _write_residual_field_shard_manifest_json,
    build_residual_field_reducer_progress_artifact,
    load_residual_field_reducer_progress_manifest,
    load_residual_field_shard_manifest,
    write_residual_field_reducer_progress_manifest,
)
from core.residual_field.shard_discovery import (  # noqa: E402
    _merge_residual_field_shard_manifests,
    _shard_manifests_by_key,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    list_reclaimable_residual_field_shards,
)


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

    def mark_saved_many(self, interval_ids, chunk_id: int) -> None:
        """One connection + one transaction for a whole chunk's rows."""
        rows = [(int(iv), int(chunk_id), 1) for iv in interval_ids]
        if not rows:
            return
        db = self.db_manager_factory(self.db_path)
        try:
            db.update_interval_chunk_status_batch(rows)
        finally:
            db.close()

    def unsaved_pairs(self) -> set[tuple[int, int]]:
        db = self.db_manager_factory(self.db_path)
        try:
            return {
                (int(iv), int(ch)) for iv, ch in db.get_unsaved_interval_chunks()
            }
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
    """Residual-field chunk artifacts use their own namespace."""

    def build_chunk_artifact_refs(self, chunk_id: int):
        return build_residual_field_output_artifacts(self.output_dir, chunk_id)

    def chunk_amplitudes_kind(self) -> str:
        return "chunk-residual-values"

    def chunk_amplitudes_average_kind(self) -> str:
        return "chunk-residual-average-values"

    def save_chunk_payload_components(
        self,
        chunk_id: int,
        *,
        point_ids: np.ndarray,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
        reciprocal_point_count: int,
    ) -> None:
        ref_by_kind = self._ref_by_kind(chunk_id)
        self._save_two_column_complex_payload(
            self._artifact_filename(ref_by_kind[self.chunk_amplitudes_kind()].path),
            "amplitudes",
            point_ids=point_ids,
            values=amplitudes_delta,
        )
        self._save_two_column_complex_payload(
            self._artifact_filename(
                ref_by_kind[self.chunk_amplitudes_average_kind()].path
            ),
            "amplitudes_av",
            point_ids=point_ids,
            values=amplitudes_average,
        )
        self.saver.save_data(
            {"nreciprocal_space_points": np.array([int(reciprocal_point_count)], dtype=np.int64)},
            self._artifact_filename(
                ref_by_kind[self.chunk_reciprocal_point_count_kind()].path
            ),
        )

    def _save_two_column_complex_payload(
        self,
        filename: str,
        dataset_name: str,
        *,
        point_ids: np.ndarray,
        values: np.ndarray,
    ) -> None:
        point_ids_arr = np.asarray(point_ids)
        values_arr = np.asarray(values).reshape(-1)
        if point_ids_arr.shape[0] != values_arr.shape[0]:
            raise ValueError(
                f"{dataset_name} point ids and values must have matching length."
            )
        file_path = Path(self.output_dir) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_name(
            f".{file_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        row_count = int(values_arr.shape[0])
        row_bytes = np.dtype(np.complex128).itemsize * 2
        block_rows = max(1, (64 * 1024 * 1024) // row_bytes)
        chunk_rows = max(1, min(row_count or 1, (4 * 1024 * 1024) // row_bytes))
        try:
            with h5py.File(tmp_path, "w") as h5file:
                dataset = h5file.create_dataset(
                    dataset_name,
                    shape=(row_count, 2),
                    maxshape=(None, 2),
                    dtype=np.complex128,
                    chunks=(chunk_rows, 2),
                )
                for start in range(0, row_count, block_rows):
                    stop = min(row_count, start + block_rows)
                    block = np.empty((stop - start, 2), dtype=np.complex128)
                    block[:, 0] = point_ids_arr[start:stop]
                    block[:, 1] = values_arr[start:stop]
                    dataset[start:stop] = block
            tmp_path.replace(file_path)
        finally:
            tmp_path.unlink(missing_ok=True)


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
        if artifact.path is None:
            missing.append(artifact.key)
            continue
        path = Path(artifact.path)
        if path.exists():
            continue
        missing.append(artifact.key)
    return tuple(sorted(missing))



def _write_hdf5_payload_atomic(
    target_path: Path,
    payload: dict[str, np.ndarray],
    *,
    attrs: dict[str, object] | None = None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f"{target_path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        with h5py.File(temp_path, "w") as h5file:
            for name, value in payload.items():
                data = np.asarray(value)
                if data.dtype.kind in {"U", "O"}:
                    dtype = h5py.string_dtype("utf-8")
                    h5file.create_dataset(name, data=data.astype(dtype), dtype=dtype)
                else:
                    h5file.create_dataset(name, data=data)
            if attrs:
                for name, value in attrs.items():
                    h5file.attrs[name] = value
            h5file.flush()
        fsync_path(temp_path)
        with h5py.File(temp_path, "r") as h5file:
            for name, value in payload.items():
                if name not in h5file:
                    raise OSError(f"HDF5 payload validation failed: missing {name!r}")
                if h5file[name].shape != np.asarray(value).shape:
                    raise OSError(
                        f"HDF5 payload validation failed for {name!r}: "
                        f"{h5file[name].shape} != {np.asarray(value).shape}"
                    )
        temp_path.replace(target_path)
        fsync_parent(target_path)
    finally:
        temp_path.unlink(missing_ok=True)



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


def is_residual_field_replacement_complete(
    *,
    chunk_id: int,
    parameter_digest: str,
    expected_interval_ids: tuple[int, ...],
    output_dir: str,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    expected_set = set(int(interval_id) for interval_id in expected_interval_ids)
    if not expected_set:
        return True
    progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=int(chunk_id),
        parameter_digest=parameter_digest,
    )
    if progress is None or progress.completion_status is not CompletionStatus.COMMITTED:
        return False
    if set(int(interval_id) for interval_id in progress.incorporated_interval_ids) != expected_set:
        return False
    store = ResidualFieldArtifactStore(output_dir)
    applied_set = set(int(interval_id) for interval_id in store.load_applied_interval_ids(chunk_id))
    if applied_set != expected_set:
        return False
    current, current_av, reciprocal_point_count, grid_shape_nd = store.load_chunk_payloads(chunk_id)
    if current is None or current_av is None or grid_shape_nd is None:
        return False
    current_arr = np.asarray(current)
    current_av_arr = np.asarray(current_av)
    if current_arr.shape != current_av_arr.shape:
        return False
    if current_arr.size == 0 or int(reciprocal_point_count) < 0:
        return False
    status_updater = _ResidualFieldChunkStatusUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    )
    unsaved = status_updater.unsaved_pairs()
    missing_db_rows = [
        int(interval_id)
        for interval_id in sorted(int(v) for v in expected_set)
        if (int(interval_id), int(chunk_id)) in unsaved
    ]
    status_updater.mark_saved_many(missing_db_rows, int(chunk_id))
    representative = ResidualFieldWorkUnit.interval_chunk(
        interval_id=max(expected_set),
        chunk_id=int(chunk_id),
        parameter_digest=parameter_digest,
        output_dir=output_dir,
    )
    manifest = build_residual_field_chunk_manifest(
        representative,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
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
    _write_residual_field_chunk_payload_components(
        store=store,
        chunk_id=chunk_id,
        point_ids=merged_state.payload.point_ids,
        grid_shape_nd=merged_state.payload.grid_shape_nd,
        amplitudes_delta=merged_state.payload.amplitudes_delta,
        amplitudes_average=merged_state.payload.amplitudes_average,
        reciprocal_point_count=merged_state.payload.reciprocal_point_count,
        total_reciprocal_points=total_reciprocal_points,
        applied_set=applied_set,
    )


def _write_residual_field_chunk_payload_components(
    *,
    store: ResidualFieldArtifactStore,
    chunk_id: int,
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    reciprocal_point_count: int,
    total_reciprocal_points: int | None,
    applied_set: set[int],
) -> None:
    store.ensure_grid_shape(chunk_id, np.asarray(grid_shape_nd))
    if total_reciprocal_points is not None:
        store.ensure_total_reciprocal_points(chunk_id, total_reciprocal_points)
    point_ids_arr = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    amplitudes_delta_arr = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
    amplitudes_average_arr = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
    if point_ids_arr.shape[0] != amplitudes_delta_arr.shape[0]:
        raise ValueError("Residual-field point_ids must match amplitudes_delta length.")
    if amplitudes_delta_arr.shape != amplitudes_average_arr.shape:
        raise ValueError("Residual-field delta and average payloads must match.")
    store.save_chunk_payload_components(
        chunk_id,
        point_ids=point_ids_arr,
        amplitudes_delta=amplitudes_delta_arr,
        amplitudes_average=amplitudes_average_arr,
        reciprocal_point_count=reciprocal_point_count,
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

    with chunk_mutex(chunk_id, lock_root=output_dir):
        store = artifact_store_factory(output_dir)
        _write_residual_field_chunk_state(
            store=store,
            chunk_id=chunk_id,
            merged_state=merged_state,
            total_reciprocal_points=total_reciprocal_points,
            applied_set=set(int(interval_id) for interval_id in target_interval_ids),
        )

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
    final_artifacts_present = (
        not _missing_artifact_kinds(final_manifest)
        and not _missing_artifact_paths(final_manifest.artifacts)
    )
    reconciled_progress = _build_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        completion_status=(
            CompletionStatus.COMMITTED
            if final_artifacts_present
            else CompletionStatus.MATERIALIZED
        ),
        durable_truth_unit="committed_shard_checkpoint",
        incorporated_shard_keys=target_shard_keys,
        incorporated_interval_ids=target_interval_ids,
        reclaimable_shard_keys=(
            target_shard_keys if final_artifacts_present else ()
        ),
        final_artifacts=final_manifest.artifacts,
        pending_shard_keys=(),
        pending_interval_ids=(),
        cleanup_policy=progress.cleanup_policy,
    )
    written_progress = write_residual_field_reducer_progress_manifest(reconciled_progress)
    if written_progress.completion_status is CompletionStatus.COMMITTED:
        status_updater = _ResidualFieldChunkStatusUpdater(
            db_path,
            db_manager_factory=db_manager_factory,
        )
        status_updater.mark_saved_many(
            sorted({int(v) for v in target_interval_ids}), int(chunk_id)
        )
    return written_progress


def load_residual_field_shard_payload(
    manifest: ResidualFieldShardManifest,
) -> dict[str, np.ndarray]:
    shard_ref = next(
        artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-data"
    )
    if shard_ref.path is None:
        raise ValueError("Residual-field shard data path is required.")
    data = _load_array_payload(shard_ref.path)
    return {
        "point_ids": np.asarray(data["point_ids"]),
        "grid_shape_nd": np.asarray(data["grid_shape_nd"]),
        "amplitudes_delta": np.asarray(data["amplitudes_delta"]),
        "amplitudes_average": np.asarray(data["amplitudes_average"]),
    }


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
        if progress.cleanup_policy != "delete_reclaimable":
            continue
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
    shard_dir = Path(shard_storage_root or output_dir) / "residual_checkpoints" / f"chunk_{chunk_id}"
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
    compress: bool = False,
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
        / "residual_checkpoints"
        / f"chunk_{work_unit.chunk_id}"
        if manifest.scratch_root
        else shard_path.parent
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "interval_id": np.array([work_unit.interval_id], dtype=np.int64),
        "contributing_interval_ids": np.asarray(
            manifest.contributing_interval_ids,
            dtype=np.int64,
        ),
        "chunk_id": np.array([work_unit.chunk_id], dtype=np.int64),
        "parameter_digest": np.array([work_unit.parameter_digest]),
        "point_ids": np.asarray(
            point_ids
            if point_ids is not None
            else np.arange(
                np.asarray(amplitudes_delta).reshape(-1).shape[0],
                dtype=np.int64,
            )
        ),
        "grid_shape_nd": np.asarray(grid_shape_nd),
        "amplitudes_delta": np.asarray(amplitudes_delta),
        "amplitudes_average": np.asarray(amplitudes_average),
        "contribution_reciprocal_points": np.array(
            [int(contribution_reciprocal_points)],
            dtype=np.int64,
        ),
        "total_reciprocal_points": np.array(
            [int(total_reciprocal_points)],
            dtype=np.int64,
        ),
    }
    scratch_path = scratch_dir / shard_path.name
    if shard_path.suffix in {".h5", ".hdf5"}:
        _write_hdf5_payload_atomic(
            scratch_path,
            payload,
            attrs={
                "schema_version": 2,
                "format": "mosaic.residual_field.shard",
            },
        )
    else:
        with tempfile.NamedTemporaryFile(
            dir=scratch_dir,
            prefix=f"{shard_path.stem}_",
            suffix=".npz",
            delete=False,
        ) as handle:
            save_fn = np.savez_compressed if compress else np.savez
            save_fn(handle, **payload)
        scratch_path = Path(handle.name)
    if scratch_path == shard_path:
        pass
    elif scratch_path.parent == shard_path.parent:
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

    with chunk_mutex(work_unit.chunk_id, lock_root=output_dir):
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
            store.save_chunk_payload_components(
                work_unit.chunk_id,
                point_ids=merged_state.payload.point_ids,
                amplitudes_delta=merged_state.payload.amplitudes_delta,
                amplitudes_average=merged_state.payload.amplitudes_average,
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
    "discover_residual_field_reducer_progress_manifest",
    "is_residual_field_manifest_complete",
    "is_residual_field_replacement_complete",
    "load_residual_field_reducer_progress_manifest",
    "reconcile_residual_field_reducer_progress",
    "write_residual_field_reducer_progress_manifest",
]
