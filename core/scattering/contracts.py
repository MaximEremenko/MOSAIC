from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.contracts import (
    ArtifactRef,
    ArtifactSchemaSpec,
    CompletionStatus,
    MergeInvariantSpec,
    RetryDisposition,
    RetryIdempotencySemantics,
)


SCATTERING_CONTRACT_SCHEMA_VERSION = 1
SCATTERING_INTERVAL_ARTIFACT_SCHEMA = ArtifactSchemaSpec(
    stage="scattering",
    name="scattering-interval-precompute",
    schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
    required_artifact_kinds=("interval-precompute",),
    completeness_rule=(
        "the interval artifact file must exist and SQLite must mark the interval as precomputed"
    ),
    resume_rule=(
        "resume is allowed while the interval is not yet committed; replay becomes a no-op "
        "only after both the NPZ artifact and SQLite precomputed flag are present"
    ),
)
SCATTERING_CHUNK_ARTIFACT_SCHEMA = ArtifactSchemaSpec(
    stage="scattering",
    name="scattering-interval-chunk",
    schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
    required_artifact_kinds=(
        "chunk-amplitudes",
        "chunk-amplitudes-average",
        "chunk-grid-shape",
        "chunk-reciprocal-point-count",
        "chunk-total-reciprocal-point-count",
        "chunk-applied-interval-ids",
    ),
    completeness_rule=(
        "all chunk artifacts must exist, SQLite must mark the (interval, chunk) pair as saved, "
        "and _applied_interval_ids must contain the interval id"
    ),
    resume_rule=(
        "resume is allowed while the upstream interval artifact is readable and the chunk is "
        "not yet committed; duplicate replay becomes a no-op only after the committed-state "
        "predicate above is satisfied"
    ),
)


def make_scattering_retry_key(interval_id: int, chunk_id: int | None = None) -> str:
    if chunk_id is None:
        return f"scattering:interval:{interval_id}"
    return f"scattering:interval-chunk:{interval_id}:{chunk_id}"


def make_scattering_artifact_key(
    kind: str,
    *,
    interval_id: int | None = None,
    chunk_id: int | None = None,
) -> str:
    parts = ["scattering", kind]
    if interval_id is not None:
        parts.append(f"interval-{interval_id}")
    if chunk_id is not None:
        parts.append(f"chunk-{chunk_id}")
    return ":".join(parts)


def build_interval_artifact_ref(output_dir: str, interval_id: int) -> ArtifactRef:
    return ArtifactRef(
        stage="scattering",
        kind="interval-precompute",
        key=make_scattering_artifact_key(
            "interval-precompute",
            interval_id=interval_id,
        ),
        path=str(Path(output_dir) / "precomputed_intervals" / f"interval_{interval_id}.npz"),
        schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
    )


def build_chunk_artifact_refs(output_dir: str, chunk_id: int) -> tuple[ArtifactRef, ...]:
    chunk_prefix = Path(output_dir) / f"point_data_chunk_{chunk_id}"
    return (
        ArtifactRef(
            stage="scattering",
            kind="chunk-amplitudes",
            key=make_scattering_artifact_key("chunk-amplitudes", chunk_id=chunk_id),
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes.hdf5")),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-amplitudes-average",
            key=make_scattering_artifact_key(
                "chunk-amplitudes-average",
                chunk_id=chunk_id,
            ),
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes_av.hdf5")),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-grid-shape",
            key=make_scattering_artifact_key("chunk-grid-shape", chunk_id=chunk_id),
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_shapeNd.hdf5")),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-reciprocal-point-count",
            key=make_scattering_artifact_key(
                "chunk-reciprocal-point-count",
                chunk_id=chunk_id,
            ),
            path=str(
                chunk_prefix.with_name(
                    f"{chunk_prefix.name}_amplitudes_nreciprocal_space_points.hdf5"
                )
            ),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-total-reciprocal-point-count",
            key=make_scattering_artifact_key(
                "chunk-total-reciprocal-point-count",
                chunk_id=chunk_id,
            ),
            path=str(
                chunk_prefix.with_name(
                    f"{chunk_prefix.name}_amplitudes_ntotal_reciprocal_space_points.hdf5"
                )
            ),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-applied-interval-ids",
            key=make_scattering_artifact_key(
                "chunk-applied-interval-ids",
                chunk_id=chunk_id,
            ),
            path=str(
                chunk_prefix.with_name(f"{chunk_prefix.name}_applied_interval_ids.hdf5")
            ),
            schema_version=SCATTERING_CONTRACT_SCHEMA_VERSION,
        ),
    )


@dataclass(frozen=True)
class ScatteringWorkUnit:
    """Serializable identity for the current scattering execution units."""

    interval_id: int
    chunk_id: int | None
    dimension: int
    interval_artifact: ArtifactRef | None
    chunk_artifact_prefix: str | None
    artifact_key: str
    retry: RetryIdempotencySemantics
    schema_version: int = SCATTERING_CONTRACT_SCHEMA_VERSION

    @classmethod
    def precompute_interval(
        cls,
        *,
        interval_id: int,
        dimension: int,
        output_dir: str,
    ) -> "ScatteringWorkUnit":
        return cls(
            interval_id=interval_id,
            chunk_id=None,
            dimension=dimension,
            interval_artifact=build_interval_artifact_ref(output_dir, interval_id),
            chunk_artifact_prefix=None,
            artifact_key=make_scattering_artifact_key(
                "interval-precompute",
                interval_id=interval_id,
            ),
            retry=RetryIdempotencySemantics(
                failure_unit="interval-precompute",
                retry_unit="interval-precompute",
                idempotency_key=make_scattering_retry_key(interval_id),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the interval as complete only when both the interval artifact "
                    "exists and SQLite marks the interval as precomputed"
                ),
            ),
        )

    @classmethod
    def interval_chunk(
        cls,
        *,
        interval_id: int,
        chunk_id: int,
        dimension: int,
        output_dir: str,
    ) -> "ScatteringWorkUnit":
        return cls(
            interval_id=interval_id,
            chunk_id=chunk_id,
            dimension=dimension,
            interval_artifact=build_interval_artifact_ref(output_dir, interval_id),
            chunk_artifact_prefix=str(Path(output_dir) / f"point_data_chunk_{chunk_id}"),
            artifact_key=make_scattering_artifact_key(
                "interval-chunk-accumulation",
                interval_id=interval_id,
                chunk_id=chunk_id,
            ),
            retry=RetryIdempotencySemantics(
                failure_unit="interval-chunk-accumulation",
                retry_unit="interval-chunk-accumulation",
                idempotency_key=make_scattering_retry_key(interval_id, chunk_id),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the accumulation as committed only when SQLite marks the "
                    "(interval, chunk) pair as saved and _applied_interval_ids contains "
                    "the interval id"
                ),
            ),
        )

    @property
    def artifact_schema(self) -> ArtifactSchemaSpec:
        return (
            SCATTERING_INTERVAL_ARTIFACT_SCHEMA
            if self.chunk_id is None
            else SCATTERING_CHUNK_ARTIFACT_SCHEMA
        )


@dataclass(frozen=True)
class ScatteringPartialResult:
    """
    Additive scattering contribution for one chunk before persistence.

    Merge invariants are documented in `SCATTERING_PARTIAL_RESULT_MERGE_INVARIANTS`.
    Duplicate delivery is rejected by merge and must be filtered by the work-unit
    idempotency key plus `_applied_interval_ids`.
    """

    chunk_id: int
    contributing_interval_ids: tuple[int, ...]
    point_ids: np.ndarray
    grid_shape_nd: np.ndarray
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray
    reciprocal_point_count: int
    schema_version: int = SCATTERING_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contributing_interval_ids",
            tuple(sorted(int(interval_id) for interval_id in self.contributing_interval_ids)),
        )
        object.__setattr__(self, "point_ids", np.asarray(self.point_ids))
        object.__setattr__(self, "grid_shape_nd", np.asarray(self.grid_shape_nd))
        object.__setattr__(self, "amplitudes_delta", np.asarray(self.amplitudes_delta))
        object.__setattr__(
            self,
            "amplitudes_average",
            np.asarray(self.amplitudes_average),
        )
        validate_scattering_partial_result(self)


SCATTERING_PARTIAL_RESULT_MERGE_INVARIANTS = MergeInvariantSpec(
    identity=(
        "a zero-valued contribution for a fixed chunk/point-id alignment/grid shape with "
        "no contributing interval ids and reciprocal_point_count == 0"
    ),
    associative=True,
    compatibility_checks=(
        "same chunk_id",
        "same schema_version",
        "same point_ids ordering",
        "same grid_shape_nd",
        "same amplitudes_delta shape and dtype",
        "same amplitudes_average shape and dtype",
    ),
    deterministic_serialization_boundary=(
        "persisted scattering state is serialized through the stage artifact manifest plus "
        "existing NPZ/HDF5 artifacts, not through in-memory object identity"
    ),
    duplicate_handling=(
        "duplicate contributing interval ids are forbidden in merge and must be filtered "
        "by the work-unit retry key and _applied_interval_ids before merge"
    ),
    ordering=(
        "contributing interval ids are normalized to sorted order; point_ids ordering must "
        "already match across compatible partial results"
    ),
)


def validate_scattering_partial_result(result: ScatteringPartialResult) -> None:
    if len(set(result.contributing_interval_ids)) != len(result.contributing_interval_ids):
        raise ValueError("contributing_interval_ids must be unique per scattering partial.")
    if result.point_ids.ndim != 1:
        raise ValueError("point_ids must be a 1D array.")
    if result.amplitudes_delta.shape != result.point_ids.shape:
        raise ValueError("amplitudes_delta must align with point_ids.")
    if result.amplitudes_average.shape != result.point_ids.shape:
        raise ValueError("amplitudes_average must align with point_ids.")
    if result.reciprocal_point_count < 0:
        raise ValueError("reciprocal_point_count must be non-negative.")


def validate_scattering_work_unit(work_unit: ScatteringWorkUnit) -> None:
    if work_unit.interval_id < 0:
        raise ValueError("interval_id must be non-negative.")
    if work_unit.chunk_id is not None and work_unit.chunk_id < 0:
        raise ValueError("chunk_id must be non-negative when present.")
    if work_unit.dimension <= 0:
        raise ValueError("dimension must be positive.")
    if work_unit.schema_version != SCATTERING_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unexpected scattering work-unit schema version.")
    if work_unit.chunk_id is None:
        if work_unit.interval_artifact is None or work_unit.chunk_artifact_prefix is not None:
            raise ValueError(
                "Interval-precompute work units must have an interval artifact and no chunk prefix."
            )
    else:
        if work_unit.interval_artifact is None or work_unit.chunk_artifact_prefix is None:
            raise ValueError(
                "Chunk-scoped scattering work units must include interval and chunk artifact identities."
            )
    if work_unit.retry.replay_disposition is not RetryDisposition.NO_OP:
        raise ValueError("Scattering Phase 6 assumes NO_OP replay semantics.")


def scattering_partial_result_identity(
    *,
    chunk_id: int,
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitude_dtype=np.complex128,
) -> ScatteringPartialResult:
    point_ids_arr = np.asarray(point_ids)
    return ScatteringPartialResult(
        chunk_id=chunk_id,
        contributing_interval_ids=(),
        point_ids=point_ids_arr,
        grid_shape_nd=np.asarray(grid_shape_nd),
        amplitudes_delta=np.zeros(point_ids_arr.shape, dtype=amplitude_dtype),
        amplitudes_average=np.zeros(point_ids_arr.shape, dtype=amplitude_dtype),
        reciprocal_point_count=0,
    )


def merge_scattering_partial_results(
    left: ScatteringPartialResult,
    right: ScatteringPartialResult,
) -> ScatteringPartialResult:
    validate_scattering_partial_result(left)
    validate_scattering_partial_result(right)
    if left.chunk_id != right.chunk_id:
        raise ValueError("Cannot merge scattering partials for different chunks.")
    if left.schema_version != right.schema_version:
        raise ValueError("Cannot merge scattering partials with different schema versions.")
    if not np.array_equal(left.point_ids, right.point_ids):
        raise ValueError("Cannot merge scattering partials with different point_ids.")
    if not np.array_equal(left.grid_shape_nd, right.grid_shape_nd):
        raise ValueError("Cannot merge scattering partials with different grid_shape_nd.")
    if left.amplitudes_delta.shape != right.amplitudes_delta.shape:
        raise ValueError("Cannot merge scattering partials with different amplitudes_delta shape.")
    if left.amplitudes_average.shape != right.amplitudes_average.shape:
        raise ValueError(
            "Cannot merge scattering partials with different amplitudes_average shape."
        )
    if left.amplitudes_delta.dtype != right.amplitudes_delta.dtype:
        raise ValueError("Cannot merge scattering partials with different amplitudes_delta dtype.")
    if left.amplitudes_average.dtype != right.amplitudes_average.dtype:
        raise ValueError(
            "Cannot merge scattering partials with different amplitudes_average dtype."
        )
    overlap = set(left.contributing_interval_ids) & set(right.contributing_interval_ids)
    if overlap:
        raise ValueError(
            f"Cannot merge scattering partials with duplicate interval ids: {sorted(overlap)}"
        )
    return ScatteringPartialResult(
        chunk_id=left.chunk_id,
        contributing_interval_ids=tuple(
            sorted(left.contributing_interval_ids + right.contributing_interval_ids)
        ),
        point_ids=left.point_ids.copy(),
        grid_shape_nd=left.grid_shape_nd.copy(),
        amplitudes_delta=np.add(left.amplitudes_delta, right.amplitudes_delta),
        amplitudes_average=np.add(left.amplitudes_average, right.amplitudes_average),
        reciprocal_point_count=left.reciprocal_point_count + right.reciprocal_point_count,
        schema_version=left.schema_version,
    )


@dataclass(frozen=True)
class ScatteringArtifactManifest:
    """Typed manifest for current scattering-stage artifacts and replay semantics."""

    artifact_key: str
    artifacts: tuple[ArtifactRef, ...]
    completion_status: CompletionStatus
    retry: RetryIdempotencySemantics
    interval_id: int | None
    chunk_id: int | None
    producer_stage: str = "scattering"
    consumer_stage: str | None = None
    upstream_artifacts: tuple[ArtifactRef, ...] = ()
    artifact_schema_name: str = SCATTERING_INTERVAL_ARTIFACT_SCHEMA.name
    schema_version: int = SCATTERING_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_work_unit(
        cls,
        work_unit: ScatteringWorkUnit,
        *,
        artifacts: tuple[ArtifactRef, ...],
        completion_status: CompletionStatus,
        consumer_stage: str | None = None,
    ) -> "ScatteringArtifactManifest":
        validate_scattering_work_unit(work_unit)
        upstream = (
            (work_unit.interval_artifact,)
            if work_unit.interval_artifact is not None and work_unit.chunk_id is not None
            else ()
        )
        return cls(
            artifact_key=work_unit.artifact_key,
            artifacts=artifacts,
            completion_status=completion_status,
            retry=work_unit.retry,
            interval_id=work_unit.interval_id,
            chunk_id=work_unit.chunk_id,
            consumer_stage=consumer_stage,
            upstream_artifacts=upstream,
            artifact_schema_name=work_unit.artifact_schema.name,
        )


def validate_scattering_artifact_manifest(manifest: ScatteringArtifactManifest) -> None:
    if manifest.producer_stage != "scattering":
        raise ValueError("Scattering artifact manifests must be produced by the scattering stage.")
    if manifest.schema_version != SCATTERING_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unexpected scattering artifact-manifest schema version.")
    artifact_keys = [artifact.key for artifact in manifest.artifacts]
    if len(set(artifact_keys)) != len(artifact_keys):
        raise ValueError("Scattering artifact manifests must not contain duplicate artifact keys.")
    artifact_schema = (
        SCATTERING_INTERVAL_ARTIFACT_SCHEMA
        if manifest.chunk_id is None
        else SCATTERING_CHUNK_ARTIFACT_SCHEMA
    )
    if manifest.artifact_schema_name != artifact_schema.name:
        raise ValueError("Scattering artifact manifest schema name does not match its scope.")


__all__ = [
    "SCATTERING_CONTRACT_SCHEMA_VERSION",
    "SCATTERING_CHUNK_ARTIFACT_SCHEMA",
    "SCATTERING_INTERVAL_ARTIFACT_SCHEMA",
    "SCATTERING_PARTIAL_RESULT_MERGE_INVARIANTS",
    "ScatteringArtifactManifest",
    "ScatteringPartialResult",
    "ScatteringWorkUnit",
    "build_chunk_artifact_refs",
    "build_interval_artifact_ref",
    "make_scattering_artifact_key",
    "make_scattering_retry_key",
    "merge_scattering_partial_results",
    "scattering_partial_result_identity",
    "validate_scattering_artifact_manifest",
    "validate_scattering_work_unit",
    "validate_scattering_partial_result",
]
