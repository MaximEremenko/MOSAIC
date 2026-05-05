from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

import h5py
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
from core.runtime import TIMER, chunk_mutex
from core.storage.database_manager import create_db_manager_for_thread
from core.storage.rifft_in_data_saver import RIFFTInDataSaver


logger = logging.getLogger(__name__)


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _q_grid_digest(q_grid: np.ndarray) -> str:
    import hashlib

    arr = np.ascontiguousarray(q_grid)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(memoryview(arr.view(np.uint8)))
    return digest.hexdigest()


def _atomic_hdf5_write(
    out_path: Path,
    datasets: dict[str, np.ndarray],
    *,
    attrs: dict[str, object] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with h5py.File(temp_path, "w") as h5file:
            for name, values in datasets.items():
                data = np.asarray(values)
                h5file.create_dataset(name, data=data)
            if attrs:
                for name, value in attrs.items():
                    h5file.attrs[name] = value
            h5file.flush()
        _fsync_path(temp_path)
        with h5py.File(temp_path, "r") as h5file:
            for name, expected in datasets.items():
                if name not in h5file:
                    raise OSError(f"HDF5 validation failed: missing dataset {name!r}")
                if h5file[name].shape != np.asarray(expected).shape:
                    raise OSError(
                        f"HDF5 validation failed for {name!r}: "
                        f"{h5file[name].shape} != {np.asarray(expected).shape}"
                    )
        os.replace(temp_path, out_path)
        _fsync_parent(out_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


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

    def build_chunk_artifact_refs(self, chunk_id: int):
        return build_chunk_artifact_refs(self.output_dir, chunk_id)

    def build_legacy_chunk_artifact_refs(self, chunk_id: int):
        return ()

    def chunk_amplitudes_kind(self) -> str:
        return "chunk-amplitudes"

    def chunk_amplitudes_average_kind(self) -> str:
        return "chunk-amplitudes-average"

    def chunk_grid_shape_kind(self) -> str:
        return "chunk-grid-shape"

    def chunk_reciprocal_point_count_kind(self) -> str:
        return "chunk-reciprocal-point-count"

    def chunk_total_reciprocal_point_count_kind(self) -> str:
        return "chunk-total-reciprocal-point-count"

    def chunk_applied_interval_ids_kind(self) -> str:
        return "chunk-applied-interval-ids"

    def _ref_by_kind(self, chunk_id: int, *, legacy: bool = False) -> dict[str, object]:
        refs = (
            self.build_legacy_chunk_artifact_refs(chunk_id)
            if legacy
            else self.build_chunk_artifact_refs(chunk_id)
        )
        return {ref.kind: ref for ref in refs}

    def _filename_for_kind(self, chunk_id: int, kind: str, *, legacy: bool = False) -> str:
        ref_by_kind = self._ref_by_kind(chunk_id, legacy=legacy)
        return self._artifact_filename(ref_by_kind[kind].path)

    def ensure_grid_shape(self, chunk_id: int, grid_shape_nd: np.ndarray) -> None:
        fn_shape = self._filename_for_kind(chunk_id, self.chunk_grid_shape_kind())
        try:
            self.saver.load_data(fn_shape)
        except FileNotFoundError:
            self.saver.save_data({"shapeNd": np.asarray(grid_shape_nd)}, fn_shape)

    def load_grid_shape(self, chunk_id: int) -> np.ndarray | None:
        for legacy in (False, True):
            ref_by_kind = self._ref_by_kind(chunk_id, legacy=legacy)
            if not ref_by_kind:
                continue
            fn_shape = self._artifact_filename(
                ref_by_kind[self.chunk_grid_shape_kind()].path
            )
            try:
                return np.asarray(self.saver.load_data(fn_shape)["shapeNd"])
            except FileNotFoundError:
                continue
        return None

    def ensure_total_reciprocal_points(
        self,
        chunk_id: int,
        total_reciprocal_points: int,
    ) -> None:
        fn_tot = self._filename_for_kind(
            chunk_id,
            self.chunk_total_reciprocal_point_count_kind(),
        )
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
        for legacy in (False, True):
            ref_by_kind = self._ref_by_kind(chunk_id, legacy=legacy)
            if not ref_by_kind:
                continue
            fn_applied = self._artifact_filename(
                ref_by_kind[self.chunk_applied_interval_ids_kind()].path
            )
            try:
                applied_arr = self.saver.load_data(fn_applied)["ids"]
            except FileNotFoundError:
                continue
            return set(int(item) for item in np.asarray(applied_arr).ravel().tolist())
        return set()

    def save_applied_interval_ids(self, chunk_id: int, applied_set: set[int]) -> None:
        fn_applied = self._filename_for_kind(
            chunk_id,
            self.chunk_applied_interval_ids_kind(),
        )
        self.saver.save_data(
            {"ids": np.array(sorted(applied_set), dtype=np.int64)},
            fn_applied,
        )

    def load_chunk_payloads(
        self,
        chunk_id: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int, np.ndarray | None]:
        for legacy in (False, True):
            ref_by_kind = self._ref_by_kind(chunk_id, legacy=legacy)
            if not ref_by_kind:
                continue
            try:
                current = self.saver.load_data(
                    self._artifact_filename(
                        ref_by_kind[self.chunk_amplitudes_kind()].path
                    )
                )["amplitudes"]
                current_av = self.saver.load_data(
                    self._artifact_filename(
                        ref_by_kind[self.chunk_amplitudes_average_kind()].path
                    )
                )["amplitudes_av"]
                nrec = self.saver.load_data(
                    self._artifact_filename(
                        ref_by_kind[self.chunk_reciprocal_point_count_kind()].path
                    )
                )["nreciprocal_space_points"]
                shape_nd = self.saver.load_data(
                    self._artifact_filename(
                        ref_by_kind[self.chunk_grid_shape_kind()].path
                    )
                )["shapeNd"]
            except FileNotFoundError:
                continue
            try:
                reciprocal_point_count = int(np.asarray(nrec).ravel()[0])
            except Exception:
                reciprocal_point_count = 0
            return current, current_av, reciprocal_point_count, np.asarray(shape_nd)
        return None, None, 0, None

    def save_chunk_payloads(
        self,
        chunk_id: int,
        *,
        amplitudes_payload: np.ndarray,
        amplitudes_average_payload: np.ndarray,
        reciprocal_point_count: int,
    ) -> None:
        ref_by_kind = self._ref_by_kind(chunk_id)
        self.saver.save_data(
            {"amplitudes": amplitudes_payload},
            self._artifact_filename(ref_by_kind[self.chunk_amplitudes_kind()].path),
        )
        self.saver.save_data(
            {"amplitudes_av": amplitudes_average_payload},
            self._artifact_filename(
                ref_by_kind[self.chunk_amplitudes_average_kind()].path
            ),
        )
        self.saver.save_data(
            {"nreciprocal_space_points": np.array([int(reciprocal_point_count)], dtype=np.int64)},
            self._artifact_filename(
                ref_by_kind[self.chunk_reciprocal_point_count_kind()].path
            ),
        )


def mark_empty_interval_precomputed(
    interval_id: int,
    *,
    db_path: str,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> None:
    """Mark an interval whose mask produced an empty Q-grid as complete.

    When a reciprocal-space mask eliminates all Q-points in a subvolume,
    no ``.npz`` artifact is written.  This function marks the interval as
    precomputed **and** marks every ``(interval, chunk)`` pair as saved so
    that downstream stages (Stage-2 chunk accumulation and residual-field)
    do not attempt to load the non-existent artifact file.
    """
    _IntervalPrecomputeStateUpdater(
        db_path, db_manager_factory=db_manager_factory
    ).mark_precomputed(interval_id)
    db = db_manager_factory(db_path)
    try:
        unsaved = db.get_unsaved_interval_chunks()
        for iv_id, chunk_id in unsaved:
            if int(iv_id) == int(interval_id):
                db.update_interval_chunk_status(int(iv_id), int(chunk_id), saved=True)
    finally:
        db.close()
    logger.debug(
        "Empty-mask interval %d marked precomputed + all chunks saved.", interval_id
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
        if artifact.path is None:
            missing.append(artifact.key)
            continue
        path = Path(artifact.path)
        if path.exists():
            continue
        if (
            artifact.kind == "interval-precompute"
            and path.suffix == ".hdf5"
            and path.with_suffix(".npz").exists()
        ):
            continue
        else:
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
    _atomic_hdf5_write(
        out_path,
        {
            "irecip_id": np.array([int(interval_task.irecip_id)], dtype=np.int64),
            "element": np.asarray(str(interval_task.element), dtype=h5py.string_dtype("utf-8")),
            "q_grid": np.asarray(interval_task.q_grid),
            "q_amp": np.asarray(interval_task.q_amp),
            "q_amp_av": np.asarray(interval_task.q_amp_av),
            "q_grid_digest": np.asarray(_q_grid_digest(interval_task.q_grid), dtype=h5py.string_dtype("ascii")),
            "half_space_role": np.asarray(
                str(interval_task.half_space_role),
                dtype=h5py.string_dtype("ascii"),
            ),
            "reciprocal_multiplicity": np.array(
                [int(interval_task.reciprocal_multiplicity)],
                dtype=np.int64,
            ),
        },
        attrs={
            "schema_version": 2,
            "interval_id": int(interval_task.irecip_id),
            "format": "mosaic.scattering.interval",
        },
    )
    _IntervalPrecomputeStateUpdater(
        db_path,
        db_manager_factory=db_manager_factory,
    ).mark_precomputed(interval_task.irecip_id)
    return build_scattering_interval_manifest(
        work_unit,
        completion_status=CompletionStatus.COMMITTED,
    )


def build_scattering_interval_chunk_shard_path(
    output_dir: str,
    *,
    chunk_id: int,
    interval_id: int,
) -> Path:
    return (
        Path(output_dir)
        / "scattering_shards"
        / f"chunk_{int(chunk_id)}"
        / f"interval_{int(interval_id)}.hdf5"
    )


def persist_scattering_interval_chunk_shard(
    work_unit: ScatteringWorkUnit,
    *,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    contribution_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    output_dir: str,
    quiet_logs: bool = False,
) -> ScatteringArtifactManifest:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk shard persistence requires a chunk-scoped work unit.")
    t0 = TIMER()
    delta = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
    average = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
    if average.shape != delta.shape:
        raise ValueError("Scattering shard amplitudes_delta/amplitudes_average shape mismatch.")
    shard_path = build_scattering_interval_chunk_shard_path(
        output_dir,
        chunk_id=work_unit.chunk_id,
        interval_id=work_unit.interval_id,
    )
    _atomic_hdf5_write(
        shard_path,
        {
            "point_ids": np.arange(delta.shape[0], dtype=np.int64),
            "grid_shape_nd": np.asarray(grid_shape_nd, dtype=np.int64),
            "amplitudes_delta": delta,
            "amplitudes_average": average,
        },
        attrs={
            "schema_version": 1,
            "format": "mosaic.scattering.interval_chunk_shard",
            "interval_id": int(work_unit.interval_id),
            "chunk_id": int(work_unit.chunk_id),
            "contribution_reciprocal_points": int(contribution_reciprocal_points),
            "total_reciprocal_points": int(total_reciprocal_points),
        },
    )
    log_fn = logger.debug if quiet_logs else logger.info
    log_fn(
        "scattering-shard | chunk %d | iv %d | bytes=%d | %.3f s",
        int(work_unit.chunk_id),
        int(work_unit.interval_id),
        int(shard_path.stat().st_size),
        TIMER() - t0,
    )
    return build_scattering_chunk_manifest(
        work_unit,
        output_dir=output_dir,
        completion_status=CompletionStatus.MATERIALIZED,
    )


def _parse_scattering_shard_interval_id(path: Path) -> int:
    stem = path.stem
    prefix = "interval_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected scattering shard filename: {path}")
    return int(stem[len(prefix):])


def discover_scattering_interval_chunk_shards(
    output_dir: str,
    *,
    chunk_id: int,
) -> list[Path]:
    shard_dir = Path(output_dir) / "scattering_shards" / f"chunk_{int(chunk_id)}"
    if not shard_dir.exists():
        return []
    return sorted(
        shard_dir.glob("interval_*.hdf5"),
        key=_parse_scattering_shard_interval_id,
    )


def load_scattering_interval_chunk_shard(path: Path):
    with h5py.File(path, "r") as h5file:
        interval_id = int(h5file.attrs["interval_id"])
        chunk_id = int(h5file.attrs["chunk_id"])
        return build_scattering_partial_result(
            chunk_id=chunk_id,
            interval_id=interval_id,
            point_ids=np.asarray(h5file["point_ids"], dtype=np.int64),
            grid_shape_nd=np.asarray(h5file["grid_shape_nd"], dtype=np.int64),
            amplitudes_delta=np.asarray(h5file["amplitudes_delta"], dtype=np.complex128),
            amplitudes_average=np.asarray(h5file["amplitudes_average"], dtype=np.complex128),
            reciprocal_point_count=int(h5file.attrs["contribution_reciprocal_points"]),
        )


def reduce_scattering_shards_for_chunk(
    *,
    chunk_id: int,
    expected_interval_ids: tuple[int, ...],
    total_reciprocal_points: int,
    output_dir: str,
    db_path: str,
    dimension: int = 1,
    quiet_logs: bool = False,
    artifact_store_factory: Callable[[str], ScatteringArtifactStore] = ScatteringArtifactStore,
    db_manager_factory: Callable[[str], object] = create_db_manager_for_thread,
) -> ScatteringArtifactManifest:
    expected = tuple(sorted(int(interval_id) for interval_id in expected_interval_ids))
    if not expected:
        raise ValueError("Scattering reducer requires at least one expected interval id.")
    t0 = TIMER()
    shard_by_interval = {
        _parse_scattering_shard_interval_id(path): path
        for path in discover_scattering_interval_chunk_shards(output_dir, chunk_id=chunk_id)
    }
    missing = tuple(interval_id for interval_id in expected if interval_id not in shard_by_interval)
    if missing:
        raise RuntimeError(
            "Scattering reducer missing interval-chunk shard coverage: "
            f"chunk={int(chunk_id)} intervals={list(missing)}"
        )
    with chunk_mutex(chunk_id, lock_root=output_dir):
        store = artifact_store_factory(output_dir)
        existing_partial, applied_set, current_payload, current_average_payload = (
            load_existing_scattering_partial_result(int(chunk_id), output_dir=output_dir)
        )
        merged_partial = existing_partial
        for interval_id in expected:
            if interval_id in applied_set:
                continue
            partial = load_scattering_interval_chunk_shard(shard_by_interval[interval_id])
            store.ensure_grid_shape(int(chunk_id), partial.grid_shape_nd)
            store.ensure_total_reciprocal_points(int(chunk_id), int(total_reciprocal_points))
            merged_partial = (
                merge_scattering_partial_results(merged_partial, partial)
                if merged_partial is not None
                else partial
            )
            applied_set.add(interval_id)
        if merged_partial is None:
            raise RuntimeError(
                f"Scattering reducer had no materialized state for chunk={int(chunk_id)}."
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
            int(chunk_id),
            amplitudes_payload=amplitudes_payload,
            amplitudes_average_payload=amplitudes_average_payload,
            reciprocal_point_count=merged_partial.reciprocal_point_count,
        )
        store.ensure_total_reciprocal_points(int(chunk_id), int(total_reciprocal_points))
        store.save_applied_interval_ids(int(chunk_id), applied_set)
        status_updater = _IntervalChunkStatusUpdater(
            db_path,
            db_manager_factory=db_manager_factory,
        )
        for interval_id in expected:
            status_updater.mark_saved(int(interval_id), int(chunk_id))
    representative = ScatteringWorkUnit.interval_chunk(
        interval_id=max(expected),
        chunk_id=int(chunk_id),
        dimension=max(int(dimension), 1),
        output_dir=output_dir,
    )
    manifest = build_scattering_chunk_manifest(
        representative,
        output_dir=output_dir,
        completion_status=CompletionStatus.COMMITTED,
    )
    log_fn = logger.debug if quiet_logs else logger.info
    log_fn(
        "scattering-reduce | chunk %d | intervals=%d | %.3f s",
        int(chunk_id),
        int(len(expected)),
        TIMER() - t0,
    )
    return manifest


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
    "build_scattering_interval_chunk_shard_path",
    "build_scattering_chunk_manifest",
    "build_scattering_interval_manifest",
    "can_resume_scattering_work_unit",
    "discover_scattering_interval_chunk_shards",
    "is_interval_artifact_committed",
    "is_scattering_manifest_complete",
    "load_existing_scattering_partial_result",
    "load_scattering_interval_chunk_shard",
    "mark_empty_interval_precomputed",
    "persist_precomputed_interval_artifact",
    "persist_scattering_interval_chunk_shard",
    "persist_scattering_interval_chunk_result",
    "reduce_scattering_shards_for_chunk",
]
