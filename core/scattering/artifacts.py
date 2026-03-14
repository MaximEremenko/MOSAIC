from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from core.scattering.accumulation import (
    build_scattering_partial_result,
    build_scattering_partial_result_from_payloads,
    materialize_scattering_payload,
    merge_scattering_partial_results,
)
from core.scattering.contracts import (
    ScatteringArtifactManifest,
    ScatteringWorkUnit,
    build_chunk_artifact_refs,
)
from core.scattering.kernels import IntervalTask
from core.scattering.runtime import TIMER
from core.contracts import CompletionStatus
from core.storage.database_manager import create_db_manager_for_thread
from core.storage import RIFFTInDataSaver


logger = logging.getLogger(__name__)


class _IntervalPrecomputeStateUpdater:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def is_precomputed(self, interval_id: int) -> bool:
        db = create_db_manager_for_thread(self.db_path)
        try:
            return db.is_interval_precomputed(interval_id)
        finally:
            db.close()

    def mark_precomputed(self, interval_id: int) -> None:
        db = create_db_manager_for_thread(self.db_path)
        try:
            db.mark_interval_precomputed(interval_id, True)
        finally:
            db.close()


class _IntervalChunkStatusUpdater:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def mark_saved(self, interval_id: int, chunk_id: int) -> None:
        db = create_db_manager_for_thread(self.db_path)
        try:
            db.update_interval_chunk_status(interval_id, chunk_id, saved=True)
        finally:
            db.close()


class ScatteringArtifactStore:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.saver = RIFFTInDataSaver(output_dir, "hdf5")

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
            self.saver.save_data(
                {
                    "ntotal_reciprocal_space_points": np.array([val], dtype=np.int64),
                    "ntotal_reciprocal_points": np.array([val], dtype=np.int64),
                },
                fn_tot,
            )

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
    return ScatteringArtifactManifest.from_work_unit(
        work_unit,
        artifacts=artifacts,
        completion_status=completion_status,
        consumer_stage="residual_field",
    )


def build_scattering_chunk_manifest(
    work_unit: ScatteringWorkUnit,
    *,
    output_dir: str,
    completion_status: CompletionStatus,
) -> ScatteringArtifactManifest:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk-scattering manifest requires a chunk-scoped work unit.")
    return ScatteringArtifactManifest.from_work_unit(
        work_unit,
        artifacts=build_chunk_artifact_refs(output_dir, work_unit.chunk_id),
        completion_status=completion_status,
        consumer_stage="residual_field",
    )


def is_interval_artifact_committed(
    work_unit: ScatteringWorkUnit,
    *,
    db_path: str,
) -> bool:
    if work_unit.interval_artifact is None or work_unit.interval_artifact.path is None:
        return False
    artifact_exists = Path(work_unit.interval_artifact.path).exists()
    state_updater = _IntervalPrecomputeStateUpdater(db_path)
    return artifact_exists and state_updater.is_precomputed(work_unit.interval_id)


def persist_precomputed_interval_artifact(
    work_unit: ScatteringWorkUnit,
    interval_task: IntervalTask,
    *,
    db_path: str,
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
    _IntervalPrecomputeStateUpdater(db_path).mark_precomputed(interval_task.irecip_id)
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
) -> ScatteringArtifactManifest:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk accumulation requires a chunk-scoped work unit.")

    t0 = TIMER()
    store = ScatteringArtifactStore(output_dir)
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

    _IntervalChunkStatusUpdater(db_path).mark_saved(work_unit.interval_id, work_unit.chunk_id)
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
    "build_scattering_chunk_manifest",
    "build_scattering_interval_manifest",
    "is_interval_artifact_committed",
    "load_existing_scattering_partial_result",
    "persist_precomputed_interval_artifact",
    "persist_scattering_interval_chunk_result",
]
