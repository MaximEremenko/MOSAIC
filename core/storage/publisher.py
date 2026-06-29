from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from core.decoding.commit import DecoderCommitManifest, decoder_commit_path
from core.residual_field.artifacts import ResidualFieldArtifactStore
from core.residual_field.commit import (
    RESIDUAL_FIELD_STAGE,
    ResidualChunkCommitManifest,
    ResidualCommitCandidateManifest,
    ResidualNoOutputManifest,
    ResidualStageCommitManifest,
    require_residual_no_output_manifest,
    validate_residual_commit_candidate,
    write_residual_stage_commit,
)
from core.scattering.artifacts import ScatteringArtifactStore
from core.scattering.commit import (
    SCATTERING_STAGE,
    ScatteringChunkCommitManifest,
    ScatteringCommitCandidateManifest,
    ScatteringStageCommitManifest,
    validate_scattering_commit_candidate,
    write_scattering_stage_commit,
)
from core.storage.atomic import (
    assert_path_contained,
    fsync_parent,
    fsync_path,
    temp_sibling_path,
)
from core.storage.attempt_store import relative_to_output, stage_commit_path
from core.storage.fingerprint import file_sha256
from core.storage.manifest import ManifestError, read_manifest
from core.storage.public_manifest import (
    PublicManifest,
    read_public_manifest,
    write_public_manifest,
)


PUBLIC_MANIFEST_FILENAME = "public_manifest.json"


class PublicPublishError(RuntimeError):
    """Raised when a run cannot be published as a compatibility projection."""


@dataclass(frozen=True)
class _ChunkProjection:
    chunk_id: int
    candidate_payload_path: str
    payload_sha256: str
    point_ids: np.ndarray
    grid_shape_nd: np.ndarray
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray
    reciprocal_point_count: int
    applied_ids: tuple[int, ...]
    source_identity: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RunProjection:
    scattering_stage: ScatteringStageCommitManifest
    residual_stage: ResidualStageCommitManifest | None
    residual_no_output: ResidualNoOutputManifest | None
    scattering_chunks: tuple[_ChunkProjection, ...]
    residual_chunks: tuple[_ChunkProjection, ...]
    decoder_commit: DecoderCommitManifest | None


def public_manifest_path(output_dir: str | Path) -> Path:
    output_root = Path(output_dir).resolve()
    return assert_path_contained(
        output_root / PUBLIC_MANIFEST_FILENAME,
        output_dir=output_root,
    )


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in handle.keys()}


def _record_for_file(
    path: str | Path,
    *,
    output_dir: str | Path,
    stage: str,
    kind: str,
    chunk_id: int | None,
) -> dict[str, Any]:
    contained = assert_path_contained(path, output_dir=output_dir)
    digest = file_sha256(contained)
    nbytes = int(contained.stat().st_size)
    # Reopen after rename/copy so the public manifest records only visible files.
    with contained.open("rb"):
        pass
    return {
        "path": relative_to_output(contained, output_dir=output_dir),
        "stage": stage,
        "kind": kind,
        "chunk_id": None if chunk_id is None else int(chunk_id),
        "file_sha256": digest,
        "payload_nbytes": nbytes,
    }


def _copy_file_atomically(source: Path, target: Path, *, output_dir: str | Path) -> None:
    source_path = assert_path_contained(source, output_dir=output_dir)
    target_path = assert_path_contained(target, output_dir=output_dir)
    if source_path == target_path:
        fsync_path(target_path)
        return
    temp_path = temp_sibling_path(target_path, suffix=".tmp")
    try:
        with source_path.open("rb") as src, temp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temp_path, target_path)
        fsync_parent(target_path)
        fsync_path(target_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def validate_public_manifest_files(
    path: str | Path,
    *,
    output_dir: str | Path,
) -> PublicManifest:
    manifest = read_public_manifest(path, output_dir=output_dir)
    for record in manifest.published_file_records:
        file_path = assert_path_contained(record["path"], output_dir=output_dir)
        if not file_path.exists():
            raise ManifestError(f"Published file is missing: {record['path']}")
        if file_sha256(file_path) != record["file_sha256"]:
            raise ManifestError(f"Published file hash mismatch: {record['path']}")
        if int(file_path.stat().st_size) != int(record["payload_nbytes"]):
            raise ManifestError(f"Published file byte-size mismatch: {record['path']}")
    return manifest


def _require_scattering_stage(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> ScatteringStageCommitManifest:
    path = stage_commit_path(output_dir, run_digest, SCATTERING_STAGE)
    if not path.exists():
        raise PublicPublishError("Cannot publish run without scattering stage_commit.json.")
    stage = read_manifest(path, codec=ScatteringStageCommitManifest, output_dir=output_dir)
    return write_scattering_stage_commit(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_ids=stage.chunk_ids,
    )


def _require_residual_stage(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> ResidualStageCommitManifest | None:
    path = stage_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE)
    if not path.exists():
        return None
    stage = read_manifest(path, codec=ResidualStageCommitManifest, output_dir=output_dir)
    return write_residual_stage_commit(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_ids=stage.chunk_ids,
    )


def _require_residual_terminal(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> tuple[ResidualStageCommitManifest | None, ResidualNoOutputManifest | None]:
    residual_stage = _require_residual_stage(output_dir=output_dir, run_digest=run_digest)
    if residual_stage is not None:
        return residual_stage, None
    try:
        no_output = require_residual_no_output_manifest(
            output_dir=output_dir,
            run_digest=run_digest,
        )
    except RuntimeError as exc:
        raise PublicPublishError(
            "Cannot publish run without residual stage_commit.json or no_output.json."
        ) from exc
    return None, no_output


def _scattering_chunk_projection(
    *,
    output_dir: str | Path,
    chunk_commit_relpath: str,
    run_digest: str,
) -> _ChunkProjection:
    commit = read_manifest(
        Path(output_dir) / chunk_commit_relpath,
        codec=ScatteringChunkCommitManifest,
        output_dir=output_dir,
    )
    if commit.run_digest != str(run_digest):
        raise PublicPublishError("Scattering chunk commit run identity mismatch.")
    candidate = read_manifest(
        Path(output_dir) / commit.candidate_manifest_path,
        codec=ScatteringCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_scattering_commit_candidate(candidate, output_dir=output_dir)
    if candidate.candidate_id != commit.selected_candidate_id:
        raise PublicPublishError("Scattering chunk commit selected candidate mismatch.")
    if candidate.payload_path != commit.candidate_payload_path:
        raise PublicPublishError("Scattering chunk commit payload path mismatch.")
    if candidate.payload_sha256 != commit.payload_sha256:
        raise PublicPublishError("Scattering chunk commit payload hash mismatch.")
    if candidate.file_sha256 != commit.file_sha256:
        raise PublicPublishError("Scattering chunk commit file hash mismatch.")
    if int(candidate.payload_nbytes) != int(commit.payload_nbytes):
        raise PublicPublishError("Scattering chunk commit byte-size mismatch.")
    payload = _load_payload(Path(output_dir) / candidate.payload_path)
    return _ChunkProjection(
        chunk_id=int(commit.chunk_id),
        candidate_payload_path=candidate.payload_path,
        payload_sha256=candidate.payload_sha256,
        point_ids=np.asarray(payload["point_ids"], dtype=np.int64),
        grid_shape_nd=np.asarray(payload["grid_shape_nd"], dtype=np.int64),
        amplitudes_delta=np.asarray(payload["amplitudes_delta"], dtype=np.complex128),
        amplitudes_average=np.asarray(payload["amplitudes_average"], dtype=np.complex128),
        reciprocal_point_count=int(candidate.reciprocal_point_count),
        applied_ids=tuple(int(item) for item in candidate.contributing_interval_ids),
    )


def _residual_chunk_projection(
    *,
    output_dir: str | Path,
    chunk_commit_relpath: str,
    run_digest: str,
) -> _ChunkProjection:
    commit = read_manifest(
        Path(output_dir) / chunk_commit_relpath,
        codec=ResidualChunkCommitManifest,
        output_dir=output_dir,
    )
    if commit.run_digest != str(run_digest):
        raise PublicPublishError("Residual chunk commit run identity mismatch.")
    candidate = read_manifest(
        Path(output_dir) / commit.candidate_manifest_path,
        codec=ResidualCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_residual_commit_candidate(candidate, output_dir=output_dir)
    if candidate.candidate_id != commit.selected_candidate_id:
        raise PublicPublishError("Residual chunk commit selected candidate mismatch.")
    if candidate.payload_path != commit.candidate_payload_path:
        raise PublicPublishError("Residual chunk commit payload path mismatch.")
    if candidate.payload_sha256 != commit.payload_sha256:
        raise PublicPublishError("Residual chunk commit payload hash mismatch.")
    if candidate.file_sha256 != commit.file_sha256:
        raise PublicPublishError("Residual chunk commit file hash mismatch.")
    if int(candidate.payload_nbytes) != int(commit.payload_nbytes):
        raise PublicPublishError("Residual chunk commit byte-size mismatch.")
    payload = _load_payload(Path(output_dir) / candidate.payload_path)
    return _ChunkProjection(
        chunk_id=int(commit.chunk_id),
        candidate_payload_path=candidate.payload_path,
        payload_sha256=candidate.payload_sha256,
        point_ids=np.asarray(payload["point_ids"], dtype=np.int64),
        grid_shape_nd=np.asarray(payload["grid_shape_nd"], dtype=np.int64),
        amplitudes_delta=np.asarray(payload["amplitudes_delta"], dtype=np.complex128),
        amplitudes_average=np.asarray(payload["amplitudes_average"], dtype=np.complex128),
        reciprocal_point_count=int(candidate.reciprocal_point_count),
        applied_ids=tuple(int(item) for item in candidate.partition_ids),
        source_identity=None if candidate.source_identity is None else dict(candidate.source_identity),
    )


def _read_decoder_commit_if_present(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> DecoderCommitManifest | None:
    path = decoder_commit_path(output_dir, run_digest)
    if not path.exists():
        return None
    commit = read_manifest(path, codec=DecoderCommitManifest, output_dir=output_dir)
    cache_path = assert_path_contained(commit.decoder_cache_path, output_dir=output_dir)
    if file_sha256(cache_path) != commit.decoder_cache_file_sha256:
        raise PublicPublishError("decoder_commit.json cache file hash mismatch.")
    if int(cache_path.stat().st_size) != int(commit.decoder_cache_nbytes):
        raise PublicPublishError("decoder_commit.json cache byte-size mismatch.")
    return commit


def _build_run_projection(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> _RunProjection:
    scattering_stage = _require_scattering_stage(output_dir=output_dir, run_digest=run_digest)
    residual_stage, residual_no_output = _require_residual_terminal(
        output_dir=output_dir,
        run_digest=run_digest,
    )
    scattering_chunks = tuple(
        sorted(
            (
                _scattering_chunk_projection(
                    output_dir=output_dir,
                    chunk_commit_relpath=path,
                    run_digest=run_digest,
                )
                for path in scattering_stage.chunk_commit_paths
            ),
            key=lambda item: item.chunk_id,
        )
    )
    residual_chunks = (
        ()
        if residual_stage is None
        else tuple(
            sorted(
                (
                    _residual_chunk_projection(
                        output_dir=output_dir,
                        chunk_commit_relpath=path,
                        run_digest=run_digest,
                    )
                    for path in residual_stage.chunk_commit_paths
                ),
                key=lambda item: item.chunk_id,
            )
        )
    )
    return _RunProjection(
        scattering_stage=scattering_stage,
        residual_stage=residual_stage,
        residual_no_output=residual_no_output,
        scattering_chunks=scattering_chunks,
        residual_chunks=residual_chunks,
        decoder_commit=_read_decoder_commit_if_present(
            output_dir=output_dir,
            run_digest=run_digest,
        ),
    )


def _save_scattering_chunk(
    *,
    output_dir: str | Path,
    chunk: _ChunkProjection,
    total_reciprocal_points: int,
) -> tuple[dict[str, Any], ...]:
    store = ScatteringArtifactStore(str(output_dir))
    records: list[dict[str, Any]] = []

    def save(kind: str, datasets: Mapping[str, np.ndarray]) -> None:
        filename = store._filename_for_kind(chunk.chunk_id, kind)
        store.saver.save_data(datasets, filename)
        records.append(
            _record_for_file(
                Path(output_dir) / filename,
                output_dir=output_dir,
                stage=SCATTERING_STAGE,
                kind=kind,
                chunk_id=chunk.chunk_id,
            )
        )

    save(store.chunk_amplitudes_kind(), {"amplitudes": chunk.amplitudes_delta})
    save(store.chunk_amplitudes_average_kind(), {"amplitudes_av": chunk.amplitudes_average})
    save(store.chunk_grid_shape_kind(), {"shapeNd": chunk.grid_shape_nd})
    save(
        store.chunk_reciprocal_point_count_kind(),
        {
            "nreciprocal_space_points": np.array(
                [int(chunk.reciprocal_point_count)],
                dtype=np.int64,
            )
        },
    )
    save(
        store.chunk_total_reciprocal_point_count_kind(),
        {
            "ntotal_reciprocal_space_points": np.array(
                [int(total_reciprocal_points)],
                dtype=np.int64,
            ),
            "ntotal_reciprocal_points": np.array(
                [int(total_reciprocal_points)],
                dtype=np.int64,
            ),
        },
    )
    save(
        store.chunk_applied_interval_ids_kind(),
        {"ids": np.array(chunk.applied_ids, dtype=np.int64)},
    )
    return tuple(records)


def _save_residual_chunk(
    *,
    output_dir: str | Path,
    chunk: _ChunkProjection,
    total_reciprocal_points: int,
) -> tuple[dict[str, Any], ...]:
    store = ResidualFieldArtifactStore(str(output_dir))
    records: list[dict[str, Any]] = []

    def record(kind: str) -> None:
        filename = store._filename_for_kind(chunk.chunk_id, kind)
        records.append(
            _record_for_file(
                Path(output_dir) / filename,
                output_dir=output_dir,
                stage=RESIDUAL_FIELD_STAGE,
                kind=kind,
                chunk_id=chunk.chunk_id,
            )
        )

    amplitudes_filename = store._filename_for_kind(chunk.chunk_id, store.chunk_amplitudes_kind())
    store._save_two_column_complex_payload(
        amplitudes_filename,
        "amplitudes",
        point_ids=chunk.point_ids,
        values=chunk.amplitudes_delta,
    )
    record(store.chunk_amplitudes_kind())

    average_filename = store._filename_for_kind(
        chunk.chunk_id,
        store.chunk_amplitudes_average_kind(),
    )
    store._save_two_column_complex_payload(
        average_filename,
        "amplitudes_av",
        point_ids=chunk.point_ids,
        values=chunk.amplitudes_average,
    )
    record(store.chunk_amplitudes_average_kind())

    def save(kind: str, datasets: Mapping[str, np.ndarray]) -> None:
        filename = store._filename_for_kind(chunk.chunk_id, kind)
        store.saver.save_data(datasets, filename)
        record(kind)

    save(store.chunk_grid_shape_kind(), {"shapeNd": chunk.grid_shape_nd})
    save(
        store.chunk_reciprocal_point_count_kind(),
        {
            "nreciprocal_space_points": np.array(
                [int(chunk.reciprocal_point_count)],
                dtype=np.int64,
            )
        },
    )
    save(
        store.chunk_total_reciprocal_point_count_kind(),
        {
            "ntotal_reciprocal_space_points": np.array(
                [int(total_reciprocal_points)],
                dtype=np.int64,
            ),
            "ntotal_reciprocal_points": np.array(
                [int(total_reciprocal_points)],
                dtype=np.int64,
            ),
        },
    )
    save(
        store.chunk_applied_interval_ids_kind(),
        {"ids": np.array(chunk.applied_ids, dtype=np.int64)},
    )
    return tuple(records)


def _archive_previous_manifest(
    *,
    output_dir: str | Path,
    previous: PublicManifest,
    previous_path: Path,
) -> tuple[str, str, str]:
    previous_digest = file_sha256(previous_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_name = f"{timestamp}_{previous.run_digest}.json"
    history_path = assert_path_contained(
        Path(output_dir) / ".mosaic" / "public_manifest_history" / history_name,
        output_dir=output_dir,
    )
    _copy_file_atomically(previous_path, history_path, output_dir=output_dir)
    return (
        previous.run_digest,
        previous_digest,
        relative_to_output(history_path, output_dir=output_dir),
    )


def _source_identity(projection: _RunProjection) -> dict[str, Any]:
    upstream_identity = [
        dict(chunk.source_identity)
        for chunk in projection.residual_chunks
        if chunk.source_identity is not None
    ]
    if not upstream_identity:
        upstream_identity = [
            {"scattering_stage_digest": projection.scattering_stage.stage_digest}
        ]
    identity: dict[str, Any] = {
        "residual_stage_digest": (
            None if projection.residual_stage is None else projection.residual_stage.stage_digest
        ),
        "residual_stage_plan_digest": (
            None
            if projection.residual_stage is None
            else projection.residual_stage.stage_plan_digest
        ),
        "residual_payload_hashes": [
            chunk.payload_sha256 for chunk in projection.residual_chunks
        ],
        "upstream_scattering_identity": upstream_identity,
        "scattering_stage_digest": projection.scattering_stage.stage_digest,
        "scattering_payload_hashes": [
            chunk.payload_sha256 for chunk in projection.scattering_chunks
        ],
    }
    if projection.residual_no_output is not None:
        identity["residual_no_output"] = {
            "no_output_digest": projection.residual_no_output.no_output_digest,
            "source_scattering_commit_digest": (
                projection.residual_no_output.source_scattering_commit_digest
            ),
            "reason": projection.residual_no_output.reason,
        }
    if projection.decoder_commit is not None:
        identity["decoder_commit_digest"] = projection.decoder_commit.decoder_commit_digest
    return identity


def _publish_decoder_cache(
    *,
    output_dir: str | Path,
    decoder_commit: DecoderCommitManifest,
) -> dict[str, Any]:
    source = assert_path_contained(decoder_commit.decoder_cache_path, output_dir=output_dir)
    if not source.name.startswith("decoder_M_") or source.suffix != ".npz":
        raise PublicPublishError("decoder_commit.json must reference a decoder_M_*.npz cache.")
    target = assert_path_contained(Path(output_dir) / source.name, output_dir=output_dir)
    _copy_file_atomically(source, target, output_dir=output_dir)
    if file_sha256(target) != decoder_commit.decoder_cache_file_sha256:
        raise PublicPublishError("Published decoder cache hash mismatch.")
    if int(target.stat().st_size) != int(decoder_commit.decoder_cache_nbytes):
        raise PublicPublishError("Published decoder cache byte-size mismatch.")
    return _record_for_file(
        target,
        output_dir=output_dir,
        stage="decoding",
        kind="decoder-cache",
        chunk_id=None,
    )


def publish_run(
    output_dir: str | Path,
    run_digest: str,
    *,
    replace: bool = False,
) -> PublicManifest:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_target = public_manifest_path(output_root)
    existing: PublicManifest | None = None
    previous_metadata: tuple[str, str, str] | None = None

    if manifest_target.exists():
        existing = validate_public_manifest_files(manifest_target, output_dir=output_root)
        if existing.run_digest != str(run_digest) and not replace:
            raise PublicPublishError(
                "public_manifest.json already publishes a different run; pass replace=True."
            )

    projection = _build_run_projection(output_dir=output_root, run_digest=str(run_digest))

    if existing is not None and existing.run_digest != str(run_digest):
        previous_metadata = _archive_previous_manifest(
            output_dir=output_root,
            previous=existing,
            previous_path=manifest_target,
        )

    records: list[dict[str, Any]] = []
    scattering_total = sum(chunk.reciprocal_point_count for chunk in projection.scattering_chunks)
    residual_total = sum(chunk.reciprocal_point_count for chunk in projection.residual_chunks)
    for chunk in projection.scattering_chunks:
        records.extend(
            _save_scattering_chunk(
                output_dir=output_root,
                chunk=chunk,
                total_reciprocal_points=scattering_total,
            )
        )
    for chunk in projection.residual_chunks:
        records.extend(
            _save_residual_chunk(
                output_dir=output_root,
                chunk=chunk,
                total_reciprocal_points=residual_total,
            )
        )
    if projection.decoder_commit is not None:
        records.append(
            _publish_decoder_cache(
                output_dir=output_root,
                decoder_commit=projection.decoder_commit,
            )
        )

    published_files = tuple(record["path"] for record in records)
    previous_run_digest = None
    previous_public_manifest_sha256 = None
    previous_public_manifest_path = None
    if previous_metadata is not None:
        (
            previous_run_digest,
            previous_public_manifest_sha256,
            previous_public_manifest_path,
        ) = previous_metadata

    return write_public_manifest(
        manifest_target,
        output_dir=output_root,
        run_digest=str(run_digest),
        source_stage=RESIDUAL_FIELD_STAGE,
        source_identity=_source_identity(projection),
        published_files=published_files,
        published_file_records=records,
        previous_run_digest=previous_run_digest,
        previous_public_manifest_sha256=previous_public_manifest_sha256,
        previous_public_manifest_path=previous_public_manifest_path,
    )


__all__ = [
    "PUBLIC_MANIFEST_FILENAME",
    "PublicPublishError",
    "public_manifest_path",
    "publish_run",
    "validate_public_manifest_files",
]
