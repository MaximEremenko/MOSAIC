from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.contracts import (
    ArtifactRef,
    CompletionStatus,
    MergeInvariantSpec,
    RetryDisposition,
    RetryIdempotencySemantics,
)


RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION = 1


def make_residual_field_retry_key(chunk_id: int, parameter_digest: str) -> str:
    return f"residual-field:chunk:{chunk_id}:{parameter_digest}"


def make_residual_field_artifact_key(
    kind: str,
    *,
    chunk_id: int,
    parameter_digest: str,
) -> str:
    return f"residual-field:{kind}:chunk-{chunk_id}:params-{parameter_digest}"


def build_residual_field_source_artifacts(
    output_dir: str,
    chunk_id: int,
    interval_id: int | None = None,
) -> tuple[ArtifactRef, ...]:
    chunk_prefix = Path(output_dir) / f"point_data_chunk_{chunk_id}"
    artifacts: list[ArtifactRef] = []
    if interval_id is not None:
        artifacts.append(
            ArtifactRef(
                stage="scattering",
                kind="interval-precompute",
                key=f"scattering:interval-precompute:interval-{interval_id}",
                path=str(Path(output_dir) / "precomputed_intervals" / f"interval_{interval_id}.npz"),
                schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
            )
        )
    artifacts.extend(
        (
        ArtifactRef(
            stage="scattering",
            kind="chunk-amplitudes",
            key=f"scattering:chunk-amplitudes:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-amplitudes-average",
            key=f"scattering:chunk-amplitudes-average:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes_av.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="scattering",
            kind="chunk-grid-shape",
            key=f"scattering:chunk-grid-shape:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_shapeNd.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        )
    )
    return tuple(artifacts)


def build_residual_field_output_artifacts(
    output_dir: str,
    chunk_id: int,
) -> tuple[ArtifactRef, ...]:
    chunk_prefix = Path(output_dir) / f"point_data_chunk_{chunk_id}"
    return (
        ArtifactRef(
            stage="residual_field",
            kind="chunk-residual-values",
            key=f"residual-field:chunk-residual-values:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-residual-average-values",
            key=f"residual-field:chunk-residual-average-values:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_amplitudes_av.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-grid-shape",
            key=f"residual-field:chunk-grid-shape:chunk-{chunk_id}",
            path=str(chunk_prefix.with_name(f"{chunk_prefix.name}_shapeNd.hdf5")),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-reciprocal-point-count",
            key=f"residual-field:chunk-reciprocal-point-count:chunk-{chunk_id}",
            path=str(
                chunk_prefix.with_name(
                    f"{chunk_prefix.name}_amplitudes_nreciprocal_space_points.hdf5"
                )
            ),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-total-reciprocal-point-count",
            key=f"residual-field:chunk-total-reciprocal-point-count:chunk-{chunk_id}",
            path=str(
                chunk_prefix.with_name(
                    f"{chunk_prefix.name}_amplitudes_ntotal_reciprocal_space_points.hdf5"
                )
            ),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="chunk-applied-interval-ids",
            key=f"residual-field:chunk-applied-interval-ids:chunk-{chunk_id}",
            path=str(
                chunk_prefix.with_name(f"{chunk_prefix.name}_applied_interval_ids.hdf5")
            ),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
    )


@dataclass(frozen=True)
class ResidualFieldWorkUnit:
    """
    Pre-extraction seam contract for the future `scattering -> residual_field -> decoding` handoff.
    """

    chunk_id: int
    interval_id: int | None
    parameter_digest: str
    source_artifacts: tuple[ArtifactRef, ...]
    patch_scope: str | None
    window_spec: str | None
    artifact_key: str
    retry: RetryIdempotencySemantics
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION

    @classmethod
    def chunk_scope(
        cls,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        patch_scope: str | None = None,
        window_spec: str | None = None,
    ) -> "ResidualFieldWorkUnit":
        return cls(
            chunk_id=chunk_id,
            interval_id=None,
            parameter_digest=parameter_digest,
            source_artifacts=build_residual_field_source_artifacts(output_dir, chunk_id),
            patch_scope=patch_scope,
            window_spec=window_spec,
            artifact_key=make_residual_field_artifact_key(
                "chunk-scope",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            ),
            retry=RetryIdempotencySemantics(
                failure_unit="residual-field-chunk",
                retry_unit="residual-field-chunk",
                idempotency_key=make_residual_field_retry_key(chunk_id, parameter_digest),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the residual-field chunk as complete only when all referenced "
                    "scattering inputs are readable and the residual output manifest is committed"
                ),
            ),
        )

    @classmethod
    def interval_chunk(
        cls,
        *,
        interval_id: int,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        patch_scope: str | None = None,
        window_spec: str | None = None,
    ) -> "ResidualFieldWorkUnit":
        interval_artifact = ArtifactRef(
            stage="scattering",
            kind="interval-precompute",
            key=f"scattering:interval-precompute:interval-{interval_id}",
            path=str(Path(output_dir) / "precomputed_intervals" / f"interval_{interval_id}.npz"),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        )
        return cls(
            chunk_id=chunk_id,
            interval_id=interval_id,
            parameter_digest=parameter_digest,
            source_artifacts=(interval_artifact,),
            patch_scope=patch_scope,
            window_spec=window_spec,
            artifact_key=make_residual_field_artifact_key(
                "interval-chunk",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            ),
            retry=RetryIdempotencySemantics(
                failure_unit="residual-field-interval-chunk",
                retry_unit="residual-field-interval-chunk",
                idempotency_key=(
                    f"residual-field:interval-chunk:{interval_id}:{chunk_id}:{parameter_digest}"
                ),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the residual-field interval/chunk as committed only when "
                    "SQLite marks the pair as saved and _applied_interval_ids contains "
                    "the interval id"
                ),
            ),
        )

    @classmethod
    def interval_chunk(
        cls,
        *,
        interval_id: int,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        patch_scope: str | None = None,
        window_spec: str | None = None,
    ) -> "ResidualFieldWorkUnit":
        return cls(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            source_artifacts=build_residual_field_source_artifacts(
                output_dir,
                chunk_id,
                interval_id=interval_id,
            ),
            patch_scope=patch_scope,
            window_spec=window_spec,
            interval_id=interval_id,
            artifact_key=make_residual_field_artifact_key(
                "interval-chunk",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            ),
            retry=RetryIdempotencySemantics(
                failure_unit="residual-field-interval-chunk",
                retry_unit="residual-field-interval-chunk",
                idempotency_key=(
                    f"{make_residual_field_retry_key(chunk_id, parameter_digest)}:"
                    f"interval-{interval_id}"
                ),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the residual-field interval-chunk as committed only when the "
                    "chunk artifacts exist, SQLite marks the (interval, chunk) pair as saved, "
                    "and _applied_interval_ids contains the interval id"
                ),
            ),
        )


@dataclass(frozen=True)
class ResidualFieldPartialResult:
    """
    Artifact-oriented residual-field seam result.

    This stays metadata-first in Phase 1B so later extraction can choose the final
    ndarray/reducer shape without changing the seam identity.
    """

    chunk_id: int
    parameter_digest: str
    output_kind: str
    source_artifacts: tuple[ArtifactRef, ...]
    output_artifacts: tuple[ArtifactRef, ...] = ()
    grid_shape: tuple[int, ...] | None = None
    point_ids: tuple[int, ...] = ()
    residual_values: np.ndarray | None = None
    residual_average_values: np.ndarray | None = None
    reciprocal_point_count: int | None = None
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.grid_shape is not None:
            object.__setattr__(self, "grid_shape", tuple(int(v) for v in self.grid_shape))
        object.__setattr__(
            self,
            "point_ids",
            tuple(int(point_id) for point_id in self.point_ids),
        )
        if self.residual_values is not None:
            object.__setattr__(self, "residual_values", np.asarray(self.residual_values).reshape(-1))
        if self.residual_average_values is not None:
            object.__setattr__(
                self,
                "residual_average_values",
                np.asarray(self.residual_average_values).reshape(-1),
            )
        if self.reciprocal_point_count is not None:
            object.__setattr__(
                self,
                "reciprocal_point_count",
                int(self.reciprocal_point_count),
            )


RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS = MergeInvariantSpec(
    identity=(
        "an empty metadata shard for a fixed chunk/output kind with no output artifacts, "
        "no point ids, and no residual payload materialized yet"
    ),
    associative=True,
    compatibility_checks=(
        "same chunk_id",
        "same parameter_digest",
        "same output_kind",
        "same schema_version",
        "same grid_shape when present on both sides",
        "same residual_values shape and dtype when materialized on both sides",
        "same residual_average_values shape and dtype when materialized on both sides",
    ),
    deterministic_serialization_boundary=(
        "the residual-field seam is serialized through artifact manifests and current HDF5 "
        "chunk artifacts; in-memory arrays are Phase 4 accumulation helpers only"
    ),
    duplicate_handling=(
        "duplicate output artifact keys are forbidden in merge; duplicate source artifact "
        "refs are deduplicated by key because they represent shared lineage"
    ),
    ordering=(
        "artifact refs and point ids are normalized to sorted key/id order to keep the "
        "metadata merge deterministic"
    ),
)


def residual_field_partial_result_identity(
    *,
    chunk_id: int,
    parameter_digest: str,
    output_kind: str,
    grid_shape: tuple[int, ...] | None = None,
) -> ResidualFieldPartialResult:
    return ResidualFieldPartialResult(
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        output_kind=output_kind,
        source_artifacts=(),
        output_artifacts=(),
        grid_shape=grid_shape,
        point_ids=(),
        residual_values=None,
        residual_average_values=None,
        reciprocal_point_count=0,
    )


def _merge_artifact_refs(
    left: tuple[ArtifactRef, ...],
    right: tuple[ArtifactRef, ...],
    *,
    allow_duplicates: bool,
) -> tuple[ArtifactRef, ...]:
    merged: dict[str, ArtifactRef] = {artifact.key: artifact for artifact in left}
    for artifact in right:
        existing = merged.get(artifact.key)
        if existing is not None:
            if existing != artifact and not allow_duplicates:
                raise ValueError(f"Conflicting artifact ref for key {artifact.key!r}.")
            continue
        merged[artifact.key] = artifact
    return tuple(merged[key] for key in sorted(merged))


def merge_residual_field_partial_results(
    left: ResidualFieldPartialResult,
    right: ResidualFieldPartialResult,
) -> ResidualFieldPartialResult:
    if left.chunk_id != right.chunk_id:
        raise ValueError("Cannot merge residual-field partials for different chunks.")
    if left.parameter_digest != right.parameter_digest:
        raise ValueError("Cannot merge residual-field partials with different parameter_digest.")
    if left.output_kind != right.output_kind:
        raise ValueError("Cannot merge residual-field partials with different output_kind.")
    if left.schema_version != right.schema_version:
        raise ValueError("Cannot merge residual-field partials with different schema versions.")
    if left.grid_shape is not None and right.grid_shape is not None and left.grid_shape != right.grid_shape:
        raise ValueError("Cannot merge residual-field partials with different grid_shape.")
    if (
        left.residual_values is not None
        and right.residual_values is not None
        and left.residual_values.shape != right.residual_values.shape
    ):
        raise ValueError("Cannot merge residual-field partials with different residual_values shape.")
    if (
        left.residual_average_values is not None
        and right.residual_average_values is not None
        and left.residual_average_values.shape != right.residual_average_values.shape
    ):
        raise ValueError(
            "Cannot merge residual-field partials with different residual_average_values shape."
        )
    if (
        left.residual_values is not None
        and right.residual_values is not None
        and left.point_ids != right.point_ids
    ):
        raise ValueError(
            "Cannot merge materialized residual-field partials with different point_ids."
        )
    merged_point_ids = (
        left.point_ids
        if left.residual_values is not None and right.residual_values is not None
        else tuple(sorted(set(left.point_ids) | set(right.point_ids)))
    )
    return ResidualFieldPartialResult(
        chunk_id=left.chunk_id,
        parameter_digest=left.parameter_digest,
        output_kind=left.output_kind,
        source_artifacts=_merge_artifact_refs(
            left.source_artifacts,
            right.source_artifacts,
            allow_duplicates=True,
        ),
        output_artifacts=_merge_artifact_refs(
            left.output_artifacts,
            right.output_artifacts,
            allow_duplicates=False,
        ),
        grid_shape=left.grid_shape if left.grid_shape is not None else right.grid_shape,
        point_ids=merged_point_ids,
        residual_values=(
            left.residual_values + right.residual_values
            if left.residual_values is not None and right.residual_values is not None
            else left.residual_values
            if left.residual_values is not None
            else right.residual_values
        ),
        residual_average_values=(
            left.residual_average_values + right.residual_average_values
            if left.residual_average_values is not None
            and right.residual_average_values is not None
            else left.residual_average_values
            if left.residual_average_values is not None
            else right.residual_average_values
        ),
        reciprocal_point_count=(
            (left.reciprocal_point_count or 0) + (right.reciprocal_point_count or 0)
        ),
        schema_version=left.schema_version,
    )


@dataclass(frozen=True)
class ResidualFieldArtifactManifest:
    """Typed manifest for the pre-extraction residual-field seam."""

    artifact_key: str
    artifacts: tuple[ArtifactRef, ...]
    completion_status: CompletionStatus
    retry: RetryIdempotencySemantics
    chunk_id: int
    parameter_digest: str
    producer_stage: str = "residual_field"
    consumer_stage: str | None = "decoding"
    upstream_artifacts: tuple[ArtifactRef, ...] = ()
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_work_unit(
        cls,
        work_unit: ResidualFieldWorkUnit,
        *,
        artifacts: tuple[ArtifactRef, ...],
        completion_status: CompletionStatus,
        consumer_stage: str | None = "decoding",
    ) -> "ResidualFieldArtifactManifest":
        return cls(
            artifact_key=work_unit.artifact_key,
            artifacts=artifacts,
            completion_status=completion_status,
            retry=work_unit.retry,
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            consumer_stage=consumer_stage,
            upstream_artifacts=work_unit.source_artifacts,
        )


__all__ = [
    "RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION",
    "RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS",
    "ResidualFieldArtifactManifest",
    "ResidualFieldPartialResult",
    "ResidualFieldWorkUnit",
    "build_residual_field_output_artifacts",
    "build_residual_field_source_artifacts",
    "make_residual_field_artifact_key",
    "make_residual_field_retry_key",
    "merge_residual_field_partial_results",
    "residual_field_partial_result_identity",
]
