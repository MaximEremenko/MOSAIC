from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from core.scattering.accumulation import (
    build_scattering_partial_result,
    build_scattering_partial_result_from_payloads,
    materialize_scattering_payload,
    merge_scattering_partial_results,
)
from core.scattering.contracts import (
    SCATTERING_CHUNK_ARTIFACT_SCHEMA,
    SCATTERING_INTERVAL_ARTIFACT_SCHEMA,
    ScatteringArtifactManifest,
    ScatteringWorkUnit,
    build_chunk_artifact_refs,
    validate_scattering_artifact_manifest,
)
from core.scattering.kernels import IntervalTask
from core.contracts import ArtifactManifestAssessment, CompletionStatus
from core.runtime import TIMER
from core.storage.database_manager import create_db_manager_for_thread
from core.storage import RIFFTInDataSaver


logger = logging.getLogger(__name__)


class _IntervalPrecomputeStateUpdater:
    def __init__(
        self,
        db_path: str,
        *,
        db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
    ) -> None:
        self.db_path = db_path
        self.db_manager_factory = db_manager_factory

    def is_precomputed(self, interval_id: int) -> bool:
        db = self.db_manager_factory(self.db_path)
        try:
            return db.is_interval_precomputed(interval_id)
        finally:
            db.close()

    def mark_precomputed(self, interval_id: int) -> None:
        db = self.db_manager_factory(self.db_path)
        try:
            db.mark_interval_precomputed(interval_id, True)
        finally:
            db.close()


class _IntervalChunkStatusUpdater:
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


class ScatteringArtifactStore:
    def __init__(
        self,
        output_dir: str,
        *,
        saver: RIFFTInDataSaver | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.saver = saver or RIFFTInDataSaver(output_dir, "hdf5")

    def _filename(self, chunk_id: int, suffix: str) -> str:
        return self.saver.generate_filename(chunk_id, suffix)

    def _artifact_filename(self, artifact_path: str) -> str:
        return Path(artifact_path).name

    def ensure_grid_shape(self, chunk_id: int, grid_shape_nd: np.ndarray) -> None:
        fn_shape = self._filename(chunk_id, "_shapeNd")
        try:
            self.saver.load_data(fn_shape)
        except FileNotFoundError:
            self.saver.save_data({"shapeNd": np.asarray(grid_shape_nd)}, fn_shape)

    def load_grid_shape(self, chunk_id: int) -> np.ndarray | None:
        fn_shape = self._filename(chunk_id, "_shapeNd")
        try:
            return np.asarray(self.saver.load_data(fn_shape)["shapeNd"])
        except FileNotFoundError:
            return None

    def ensure_total_reciprocal_points(
        self,
        chunk_id: int,
        total_reciprocal_points: int,
    ) -> None:
        fn_tot = self._filename(chunk_id, "_amplitudes_ntotal_reciprocal_space_points")
        val = int(total_reciprocal_points)

        def _write_total_points() -> None:
            self.saver.save_data(
                {
                    "ntotal_reciprocal_space_points": np.array([val], dtype=np.int64),
                    "ntotal_reciprocal_points": np.array([val], dtype=np.int64),
                },
                fn_tot,
            )

        try:
            data = self.saver.load_data(fn_tot)

            def _needs_update(store: dict, key: str) -> bool:
                arr = store.get(key, None)
                if arr is None:
                    return True
                try:
                    return int(np.array(arr).ravel()[0]) == -1
                except Exception:
                    return True

            if _needs_update(data, "ntotal_reciprocal_space_points") or _needs_update(
                data, "ntotal_reciprocal_points"
            ):
                data["ntotal_reciprocal_space_points"] = np.array([val], dtype=np.int64)
                data["ntotal_reciprocal_points"] = np.array([val], dtype=np.int64)
                self.saver.save_data(data, fn_tot)
        except FileNotFoundError:
            _write_total_points()
        except Exception as exc:
            logger.warning(
                "Recreating corrupted total reciprocal-point artifact for chunk %d: %s",
                chunk_id,
                exc,
            )
            _write_total_points()

    def load_applied_interval_ids(self, chunk_id: int) -> set[int]:
        fn_applied = self._filename(chunk_id, "_applied_interval_ids")
        try:
            applied_arr = self.saver.load_data(fn_applied)["ids"]
        except FileNotFoundError:
            return set()
        return set(int(item) for item in np.asarray(applied_arr).ravel().tolist())

    def save_applied_interval_ids(self, chunk_id: int, applied_set: set[int]) -> None:
        fn_applied = self._filename(chunk_id, "_applied_interval_ids")
        self.saver.save_data(
            {"ids": np.array(sorted(applied_set), dtype=np.int64)},
            fn_applied,
        )

    def load_chunk_payloads(
        self,
        chunk_id: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int, np.ndarray | None]:
        refs = build_chunk_artifact_refs(self.output_dir, chunk_id)
        ref_by_kind = {ref.kind: ref for ref in refs}
        try:
            current = self.saver.load_data(
                self._artifact_filename(ref_by_kind["chunk-amplitudes"].path)
            )["amplitudes"]
            current_av = self.saver.load_data(
                self._artifact_filename(ref_by_kind["chunk-amplitudes-average"].path)
            )["amplitudes_av"]
            nrec = self.saver.load_data(
                self._artifact_filename(ref_by_kind["chunk-reciprocal-point-count"].path)
            )["nreciprocal_space_points"]
            shape_nd = self.saver.load_data(
                self._artifact_filename(ref_by_kind["chunk-grid-shape"].path)
            )["shapeNd"]
        except FileNotFoundError:
            return None, None, 0, None
        try:
            reciprocal_point_count = int(np.asarray(nrec).ravel()[0])
        except Exception:
            reciprocal_point_count = 0
        return current, current_av, reciprocal_point_count, np.asarray(shape_nd)

    def save_chunk_payloads(
        self,
        chunk_id: int,
        *,
        amplitudes_payload: np.ndarray,
        amplitudes_average_payload: np.ndarray,
        reciprocal_point_count: int,
    ) -> None:
        refs = build_chunk_artifact_refs(self.output_dir, chunk_id)
        ref_by_kind = {ref.kind: ref for ref in refs}
        self.saver.save_data(
            {"amplitudes": amplitudes_payload},
            self._artifact_filename(ref_by_kind["chunk-amplitudes"].path),
        )
        self.saver.save_data(
            {"amplitudes_av": amplitudes_average_payload},
            self._artifact_filename(ref_by_kind["chunk-amplitudes-average"].path),
        )
        self.saver.save_data(
            {"nreciprocal_space_points": np.array([int(reciprocal_point_count)], dtype=np.int64)},
            self._artifact_filename(ref_by_kind["chunk-reciprocal-point-count"].path),
        )


def build_scattering_interval_manifest(
    work_unit: ScatteringWorkUnit,
    *,
    completion_status: CompletionStatus,
) -> ScatteringArtifactManifest:
    artifacts = (
        (work_unit.interval_artifact,)
        if work_unit.interval_artifact is not None
        else ()
    )
    manifest = ScatteringArtifactManifest.from_work_unit(
        work_unit,
        artifacts=artifacts,
        completion_status=completion_status,
        consumer_stage="residual_field",
    )
    validate_scattering_artifact_manifest(manifest)
    return manifest


def build_scattering_chunk_manifest(
    work_unit: ScatteringWorkUnit,
    *,
    output_dir: str,
    completion_status: CompletionStatus,
) -> ScatteringArtifactManifest:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk-scattering manifest requires a chunk-scoped work unit.")
    manifest = ScatteringArtifactManifest.from_work_unit(
        work_unit,
        artifacts=build_chunk_artifact_refs(output_dir, work_unit.chunk_id),
        completion_status=completion_status,
        consumer_stage="residual_field",
    )
    validate_scattering_artifact_manifest(manifest)
    return manifest


def _missing_artifact_kinds(
    manifest: ScatteringArtifactManifest,
) -> tuple[str, ...]:
    schema = (
        SCATTERING_INTERVAL_ARTIFACT_SCHEMA
        if manifest.chunk_id is None
        else SCATTERING_CHUNK_ARTIFACT_SCHEMA
    )
    present_kinds = {artifact.kind for artifact in manifest.artifacts}
    return tuple(
        kind for kind in schema.required_artifact_kinds if kind not in present_kinds
    )


def _missing_artifact_paths(artifacts: tuple) -> tuple[str, ...]:
    missing: list[str] = []
    for artifact in artifacts:
        if artifact.path is None or not Path(artifact.path).exists():
            missing.append(artifact.key)
    return tuple(sorted(missing))


def assess_scattering_manifest(
    manifest: ScatteringArtifactManifest,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ArtifactManifestAssessment:
    validate_scattering_artifact_manifest(manifest)
    schema = (
        SCATTERING_INTERVAL_ARTIFACT_SCHEMA
        if manifest.chunk_id is None
        else SCATTERING_CHUNK_ARTIFACT_SCHEMA
    )
    missing_kinds = _missing_artifact_kinds(manifest)
    missing_paths = _missing_artifact_paths(manifest.artifacts)
    all_required_artifacts_present = not missing_kinds and not missing_paths

    if manifest.chunk_id is None:
        committed_state_consistent = (
            manifest.interval_id is not None
            and _IntervalPrecomputeStateUpdater(
                db_path,
                db_manager_factory=db_manager_factory,
            ).is_precomputed(manifest.interval_id)
        )
        can_resume = not (
            all_required_artifacts_present
            and committed_state_consistent
            and manifest.completion_status is CompletionStatus.COMMITTED
        )
    else:
        upstream_paths_missing = _missing_artifact_paths(manifest.upstream_artifacts)
        applied_ids = ScatteringArtifactStore(
            str(Path(manifest.artifacts[0].path).parent)
        ).load_applied_interval_ids(manifest.chunk_id)
        committed_state_consistent = (
            manifest.interval_id is not None
            and _IntervalChunkStatusUpdater(
                db_path,
                db_manager_factory=db_manager_factory,
            ).is_saved(manifest.interval_id, manifest.chunk_id)
            and manifest.interval_id in applied_ids
        )
        can_resume = bool(not upstream_paths_missing) and not (
            all_required_artifacts_present
            and committed_state_consistent
            and manifest.completion_status is CompletionStatus.COMMITTED
        )

    is_complete = (
        all_required_artifacts_present
        and committed_state_consistent
        and manifest.completion_status is CompletionStatus.COMMITTED
    )
    detail = (
        "committed"
        if is_complete
        else schema.resume_rule if can_resume else schema.completeness_rule
    )
    return ArtifactManifestAssessment(
        schema=schema,
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


def is_scattering_manifest_complete(
    manifest: ScatteringArtifactManifest,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    return assess_scattering_manifest(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    ).is_complete


def can_resume_scattering_work_unit(
    work_unit: ScatteringWorkUnit,
    *,
    output_dir: str,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    manifest = (
        build_scattering_interval_manifest(
            work_unit,
            completion_status=CompletionStatus.COMMITTED,
        )
        if work_unit.chunk_id is None
        else build_scattering_chunk_manifest(
            work_unit,
            output_dir=output_dir,
            completion_status=CompletionStatus.COMMITTED,
        )
    )
    return assess_scattering_manifest(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    ).can_resume


def is_interval_artifact_committed(
    work_unit: ScatteringWorkUnit,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> bool:
    manifest = build_scattering_interval_manifest(
        work_unit,
        completion_status=CompletionStatus.COMMITTED,
    )
    return is_scattering_manifest_complete(
        manifest,
        db_path=db_path,
        db_manager_factory=db_manager_factory,
    )


def persist_precomputed_interval_artifact(
    work_unit: ScatteringWorkUnit,
    interval_task: IntervalTask,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ScatteringArtifactManifest:
    if work_unit.interval_artifact is None or work_unit.interval_artifact.path is None:
        raise ValueError("Precompute work unit must include an interval artifact path.")
    out_path = Path(work_unit.interval_artifact.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent,
        prefix=f"interval_{interval_task.irecip_id}_",
        suffix=".npz",
        delete=False,
    ) as handle:
        np.savez_compressed(
            handle,
            irecip_id=interval_task.irecip_id,
            element=interval_task.element,
            q_grid=interval_task.q_grid,
            q_amp=interval_task.q_amp,
            q_amp_av=interval_task.q_amp_av,
        )
    Path(handle.name).replace(out_path)
    _IntervalPrecomputeStateUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    ).mark_precomputed(interval_task.irecip_id)
    return build_scattering_interval_manifest(
        work_unit,
        completion_status=CompletionStatus.COMMITTED,
    )


def load_existing_scattering_partial_result(
    chunk_id: int,
    *,
    output_dir: str,
) -> tuple[object | None, set[int], np.ndarray | None, np.ndarray | None]:
    store = ScatteringArtifactStore(output_dir)
    current, current_av, reciprocal_point_count, grid_shape_nd = store.load_chunk_payloads(chunk_id)
    applied_set = store.load_applied_interval_ids(chunk_id)
    if current is None or current_av is None:
        return None, applied_set, current, current_av
    partial = build_scattering_partial_result_from_payloads(
        chunk_id=chunk_id,
        contributing_interval_ids=tuple(sorted(applied_set)),
        amplitudes_payload=current,
        amplitudes_average_payload=current_av,
        grid_shape_nd=(
            grid_shape_nd if grid_shape_nd is not None else np.array([], dtype=int)
        ),
        reciprocal_point_count=reciprocal_point_count,
    )
    return partial, applied_set, current, current_av


def persist_scattering_interval_chunk_result(
    work_unit: ScatteringWorkUnit,
    *,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    contribution_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    output_dir: str,
    db_path: str,
    quiet_logs: bool = False,
    artifact_store_factory: Callable[[str], ScatteringArtifactStore] = ScatteringArtifactStore,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ScatteringArtifactManifest:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk accumulation requires a chunk-scoped work unit.")

    t0 = TIMER()
    store = artifact_store_factory(output_dir)
    store.ensure_grid_shape(work_unit.chunk_id, grid_shape_nd)
    store.ensure_total_reciprocal_points(work_unit.chunk_id, total_reciprocal_points)

    existing_partial, applied_set, current_payload, current_average_payload = (
        load_existing_scattering_partial_result(work_unit.chunk_id, output_dir=output_dir)
    )
    already_applied = work_unit.interval_id in applied_set

    if not already_applied:
        point_ids = (
            existing_partial.point_ids
            if existing_partial is not None
            else None
        )
        new_partial = build_scattering_partial_result(
            chunk_id=work_unit.chunk_id,
            interval_id=work_unit.interval_id,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            grid_shape_nd=grid_shape_nd,
            reciprocal_point_count=contribution_reciprocal_points,
            point_ids=point_ids,
        )
        merged_partial = (
            merge_scattering_partial_results(existing_partial, new_partial)
            if existing_partial is not None
            else new_partial
        )
        amplitudes_payload = materialize_scattering_payload(
            current_payload,
            merged_partial.point_ids,
            merged_partial.amplitudes_delta,
        )
        amplitudes_average_payload = materialize_scattering_payload(
            current_average_payload,
            merged_partial.point_ids,
            merged_partial.amplitudes_average,
        )
        store.save_chunk_payloads(
            work_unit.chunk_id,
            amplitudes_payload=amplitudes_payload,
            amplitudes_average_payload=amplitudes_average_payload,
            reciprocal_point_count=merged_partial.reciprocal_point_count,
        )
        applied_set.add(work_unit.interval_id)
        store.save_applied_interval_ids(work_unit.chunk_id, applied_set)

    _IntervalChunkStatusUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    ).mark_saved(work_unit.interval_id, work_unit.chunk_id)
    manifest = build_scattering_chunk_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )

    if quiet_logs:
        logger.debug(
            "write-HDF5 | chunk %d | iv %d %s | %.3f s",
            work_unit.chunk_id,
            work_unit.interval_id,
            "already applied (idempotent skip)" if already_applied else "applied",
            TIMER() - t0,
        )
    else:
        if already_applied:
            logger.info(
                "write-HDF5 | chunk %d | iv %d already applied (idempotent skip) | %.3f s",
                work_unit.chunk_id,
                work_unit.interval_id,
                TIMER() - t0,
            )
        else:
            logger.info(
                "write-HDF5 | chunk %d | iv %d applied | %.3f s",
                work_unit.chunk_id,
                work_unit.interval_id,
                TIMER() - t0,
            )
    return manifest


__all__ = [
    "ScatteringArtifactStore",
    "assess_scattering_manifest",
    "build_scattering_chunk_manifest",
    "build_scattering_interval_manifest",
    "can_resume_scattering_work_unit",
    "is_interval_artifact_committed",
    "is_scattering_manifest_complete",
    "load_existing_scattering_partial_result",
    "persist_precomputed_interval_artifact",
    "persist_scattering_interval_chunk_result",
]
