from __future__ import annotations

from dataclasses import dataclass, replace
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


RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION = 1
RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA = ArtifactSchemaSpec(
    stage="residual_field",
    name="residual-field-shard-checkpoint",
    schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    required_artifact_kinds=(
        "residual-shard-data",
        "residual-shard-manifest",
    ),
    completeness_rule=(
        "the shard NPZ and shard manifest JSON must both exist and agree on "
        "chunk_id, interval_id, parameter_digest, and schema_version"
    ),
    resume_rule=(
        "resume is allowed while upstream scattering artifacts are readable and the "
        "final reduced chunk has not yet incorporated this shard's interval id"
    ),
)
RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA = ArtifactSchemaSpec(
    stage="residual_field",
    name="residual-field-interval-chunk",
    schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    required_artifact_kinds=(
        "chunk-residual-values",
        "chunk-residual-average-values",
        "chunk-grid-shape",
        "chunk-reciprocal-point-count",
        "chunk-total-reciprocal-point-count",
        "chunk-applied-interval-ids",
    ),
    completeness_rule=(
        "all residual-field chunk artifacts must exist, SQLite must mark the "
        "(interval, chunk) pair as saved, and _applied_interval_ids must contain the interval id"
    ),
    resume_rule=(
        "resume is allowed while the upstream scattering artifacts are readable and the chunk "
        "is not yet committed; duplicate replay becomes a no-op only after the committed-state "
        "predicate above is satisfied for the same parameter digest"
    ),
)
RESIDUAL_FIELD_REDUCER_PROGRESS_SCHEMA = ArtifactSchemaSpec(
    stage="residual_field",
    name="residual-field-reducer-progress",
    schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    required_artifact_kinds=("residual-reducer-progress-manifest",),
    completeness_rule=(
        "the reducer progress manifest must exist and record the incorporated shard keys, "
        "incorporated interval ids, final chunk artifact refs, and cleanup eligibility"
    ),
    resume_rule=(
        "resume is allowed while reducer progress exists but not all discovered shard keys "
        "have been incorporated into the final chunk artifacts"
    ),
)


def make_residual_field_retry_key(chunk_id: int, parameter_digest: str) -> str:
    return f"residual-field:chunk:{chunk_id}:{parameter_digest}"


def make_residual_field_artifact_key(
    kind: str,
    *,
    chunk_id: int,
    parameter_digest: str,
) -> str:
    return f"residual-field:{kind}:chunk-{chunk_id}:params-{parameter_digest}"


def make_residual_field_reducer_key(
    *,
    chunk_id: int,
    parameter_digest: str,
) -> str:
    return make_residual_field_artifact_key(
        "reducer",
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )


def make_residual_field_batch_token(interval_ids: tuple[int, ...]) -> str:
    normalized = tuple(sorted(int(interval_id) for interval_id in interval_ids))
    if not normalized:
        raise ValueError("Residual-field batch token requires at least one interval id.")
    token = f"{normalized[0]}-{normalized[-1]}-n{len(normalized)}"
    return token


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


def build_residual_field_interval_source_artifacts(
    output_dir: str,
    interval_id: int,
) -> tuple[ArtifactRef, ...]:
    return (
        ArtifactRef(
            stage="scattering",
            kind="interval-precompute",
            key=f"scattering:interval-precompute:interval-{interval_id}",
            path=str(Path(output_dir) / "precomputed_intervals" / f"interval_{interval_id}.npz"),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
    )


def build_residual_field_output_artifacts(
    output_dir: str,
    chunk_id: int,
    *,
    legacy_layout: bool = False,
) -> tuple[ArtifactRef, ...]:
    chunk_prefix = Path(output_dir) / (
        f"point_data_chunk_{chunk_id}"
        if legacy_layout
        else f"residual_chunk_{chunk_id}"
    )
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


def build_legacy_residual_field_output_artifacts(
    output_dir: str,
    chunk_id: int,
) -> tuple[ArtifactRef, ...]:
    return build_residual_field_output_artifacts(
        output_dir,
        chunk_id,
        legacy_layout=True,
    )


def build_residual_field_shard_artifacts(
    output_dir: str,
    *,
    chunk_id: int,
    interval_ids: tuple[int, ...],
    parameter_digest: str,
    shard_storage_root: str | None = None,
) -> tuple[ArtifactRef, ...]:
    shard_root = Path(shard_storage_root or output_dir)
    shard_dir = shard_root / "residual_shards" / f"chunk_{chunk_id}"
    batch_token = make_residual_field_batch_token(interval_ids)
    base_name = f"batch_{batch_token}_params_{parameter_digest}"
    return (
        ArtifactRef(
            stage="residual_field",
            kind="residual-shard-data",
            key=make_residual_field_artifact_key(
                "residual-shard-data",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            + f":batch-{batch_token}",
            path=str(shard_dir / f"{base_name}.npz"),
            schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        ),
        ArtifactRef(
            stage="residual_field",
            kind="residual-shard-manifest",
            key=make_residual_field_artifact_key(
                "residual-shard-manifest",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            + f":batch-{batch_token}",
            path=str(shard_dir / f"{base_name}.manifest.json"),
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
    interval_ids: tuple[int, ...]
    parameter_digest: str
    source_artifacts: tuple[ArtifactRef, ...]
    patch_scope: str | None
    window_spec: str | None
    artifact_key: str
    retry: RetryIdempotencySemantics
    partition_id: int | None = None
    point_start: int | None = None
    point_stop: int | None = None
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
            interval_ids=(),
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
        return cls(
            chunk_id=chunk_id,
            interval_id=interval_id,
            interval_ids=(interval_id,),
            parameter_digest=parameter_digest,
            source_artifacts=build_residual_field_interval_source_artifacts(
                output_dir,
                interval_id,
            ),
            patch_scope=patch_scope,
            window_spec=window_spec,
            artifact_key=make_residual_field_artifact_key(
                "interval-chunk",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            + f":interval-{interval_id}",
            retry=RetryIdempotencySemantics(
                failure_unit="residual-field-interval-chunk",
                retry_unit="residual-field-interval-chunk",
                idempotency_key=(
                    f"{make_residual_field_retry_key(chunk_id, parameter_digest)}:"
                    f"interval-{interval_id}"
                ),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the residual-field interval/chunk map task as committed only when "
                    "the residual shard checkpoint NPZ and manifest JSON both exist"
                ),
            ),
        )

    @classmethod
    def interval_chunk_batch(
        cls,
        *,
        interval_ids: tuple[int, ...] | list[int],
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        patch_scope: str | None = None,
        window_spec: str | None = None,
    ) -> "ResidualFieldWorkUnit":
        normalized = tuple(sorted(int(interval_id) for interval_id in interval_ids))
        if not normalized:
            raise ValueError("Residual-field batch work units require at least one interval id.")
        batch_token = make_residual_field_batch_token(normalized)
        return cls(
            chunk_id=chunk_id,
            interval_id=normalized[0],
            interval_ids=normalized,
            parameter_digest=parameter_digest,
            source_artifacts=tuple(
                artifact
                for interval_id in normalized
                for artifact in build_residual_field_interval_source_artifacts(
                    output_dir,
                    interval_id,
                )
            ),
            patch_scope=patch_scope,
            window_spec=window_spec,
            artifact_key=make_residual_field_artifact_key(
                "interval-batch",
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            + f":batch-{batch_token}",
            retry=RetryIdempotencySemantics(
                failure_unit="residual-field-interval-batch",
                retry_unit="residual-field-interval-batch",
                idempotency_key=(
                    f"{make_residual_field_retry_key(chunk_id, parameter_digest)}:"
                    f"batch-{batch_token}"
                ),
                replay_disposition=RetryDisposition.NO_OP,
                crash_recovery_rule=(
                    "treat the residual-field batch map task as committed only when "
                    "the residual shard checkpoint NPZ and manifest JSON both exist "
                    "for the full interval batch"
                ),
            ),
        )

    @property
    def artifact_schema(self) -> ArtifactSchemaSpec:
        return RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA

    def with_partition(
        self,
        *,
        partition_id: int,
        point_start: int,
        point_stop: int,
    ) -> "ResidualFieldWorkUnit":
        token = f"partition-{int(partition_id)}:points-{int(point_start)}-{int(point_stop)}"
        return replace(
            self,
            artifact_key=f"{self.artifact_key}:{token}",
            retry=RetryIdempotencySemantics(
                failure_unit=self.retry.failure_unit,
                retry_unit=self.retry.retry_unit,
                idempotency_key=f"{self.retry.idempotency_key}:{token}",
                replay_disposition=self.retry.replay_disposition,
                crash_recovery_rule=self.retry.crash_recovery_rule,
            ),
            partition_id=int(partition_id),
            point_start=int(point_start),
            point_stop=int(point_stop),
        )


@dataclass(frozen=True)
class ResidualFieldAccumulatorStatus:
    """
    Small task-return contract for worker-owned local reduction.
    """

    artifact_key: str
    chunk_id: int
    parameter_digest: str
    interval_ids: tuple[int, ...]
    contribution_reciprocal_point_count: int
    total_reciprocal_points: int
    partition_id: int | None = None
    reducer_mode: str = "local_owner"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interval_ids",
            tuple(sorted(int(interval_id) for interval_id in self.interval_ids)),
        )
        object.__setattr__(
            self,
            "contribution_reciprocal_point_count",
            int(self.contribution_reciprocal_point_count),
        )
        object.__setattr__(
            self,
            "total_reciprocal_points",
            int(self.total_reciprocal_points),
        )
        if self.partition_id is not None:
            object.__setattr__(self, "partition_id", int(self.partition_id))


@dataclass(frozen=True)
class ResidualFieldPartialResult:
    """
    Artifact-oriented residual-field seam result.

    This stays metadata-first in Phase 1B so later extraction can choose the final
    ndarray/reducer shape without changing the seam identity.
    """

    chunk_id: int
    contributing_interval_ids: tuple[int, ...]
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
        object.__setattr__(
            self,
            "contributing_interval_ids",
            tuple(
                sorted(int(interval_id) for interval_id in self.contributing_interval_ids)
            ),
        )
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
        validate_residual_field_partial_result(self)


RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS = MergeInvariantSpec(
    identity=(
        "an empty metadata shard for a fixed chunk/output kind with no output artifacts, "
        "no point ids, and no residual payload materialized yet"
    ),
    associative=True,
    compatibility_checks=(
        "same chunk_id",
        "same contributing_interval_ids only through duplicate-free merge",
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
        "duplicate contributing interval ids and duplicate output artifact keys are forbidden "
        "in merge; duplicate source artifact refs are deduplicated by key because they "
        "represent shared lineage"
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
        contributing_interval_ids=(),
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


def validate_residual_field_work_unit(work_unit: ResidualFieldWorkUnit) -> None:
    if work_unit.chunk_id < 0:
        raise ValueError("chunk_id must be non-negative.")
    if work_unit.interval_ids:
        if work_unit.interval_id is None or work_unit.interval_id != work_unit.interval_ids[0]:
            raise ValueError("interval_id must match the first interval id in interval_ids.")
    elif work_unit.interval_id is not None:
        raise ValueError("chunk-scope work units must not carry interval_id without interval_ids.")
    if work_unit.interval_ids and min(work_unit.interval_ids) < 0:
        raise ValueError("Residual-field work units require non-negative interval ids.")
    if not work_unit.parameter_digest:
        raise ValueError("parameter_digest must be non-empty.")
    if not work_unit.source_artifacts:
        raise ValueError("Residual-field work units must reference at least one source artifact.")
    source_keys = [artifact.key for artifact in work_unit.source_artifacts]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("Residual-field work units must not contain duplicate source artifact keys.")
    if work_unit.retry.replay_disposition is not RetryDisposition.NO_OP:
        raise ValueError("Residual-field Phase 6 assumes NO_OP replay semantics.")
    if work_unit.partition_id is None:
        if work_unit.point_start is not None or work_unit.point_stop is not None:
            raise ValueError(
                "Unpartitioned residual-field work units must not define point_start/point_stop."
            )
    else:
        if work_unit.point_start is None or work_unit.point_stop is None:
            raise ValueError(
                "Partitioned residual-field work units must define point_start and point_stop."
            )
        if int(work_unit.partition_id) < 0:
            raise ValueError("partition_id must be non-negative.")
        if int(work_unit.point_start) < 0 or int(work_unit.point_stop) <= int(work_unit.point_start):
            raise ValueError("Partition point range must satisfy 0 <= point_start < point_stop.")


def validate_residual_field_partial_result(result: ResidualFieldPartialResult) -> None:
    if len(set(result.contributing_interval_ids)) != len(result.contributing_interval_ids):
        raise ValueError("contributing_interval_ids must be unique per residual-field partial.")
    output_keys = [artifact.key for artifact in result.output_artifacts]
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("output_artifacts must not contain duplicate artifact keys.")
    if result.reciprocal_point_count is not None and result.reciprocal_point_count < 0:
        raise ValueError("reciprocal_point_count must be non-negative.")
    if result.residual_values is not None:
        if result.point_ids and len(result.point_ids) != result.residual_values.shape[0]:
            raise ValueError("residual_values must align with point_ids.")
    if result.residual_average_values is not None:
        if result.point_ids and len(result.point_ids) != result.residual_average_values.shape[0]:
            raise ValueError("residual_average_values must align with point_ids.")
    if (
        result.residual_values is not None
        and result.residual_average_values is not None
        and result.residual_values.shape != result.residual_average_values.shape
    ):
        raise ValueError("residual_values and residual_average_values must have matching shape.")


def merge_residual_field_partial_results(
    left: ResidualFieldPartialResult,
    right: ResidualFieldPartialResult,
) -> ResidualFieldPartialResult:
    validate_residual_field_partial_result(left)
    validate_residual_field_partial_result(right)
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
    overlap = set(left.contributing_interval_ids) & set(right.contributing_interval_ids)
    if overlap:
        raise ValueError(
            "Cannot merge residual-field partials with duplicate interval ids: "
            f"{sorted(overlap)}"
        )
    merged_point_ids = (
        left.point_ids
        if left.residual_values is not None and right.residual_values is not None
        else tuple(sorted(set(left.point_ids) | set(right.point_ids)))
    )
    return ResidualFieldPartialResult(
        chunk_id=left.chunk_id,
        contributing_interval_ids=tuple(
            sorted(left.contributing_interval_ids + right.contributing_interval_ids)
        ),
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
    interval_id: int | None
    chunk_id: int
    parameter_digest: str
    producer_stage: str = "residual_field"
    consumer_stage: str | None = "decoding"
    upstream_artifacts: tuple[ArtifactRef, ...] = ()
    artifact_schema_name: str = RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.name
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_work_unit(
        cls,
        work_unit: ResidualFieldWorkUnit,
        *,
        artifacts: tuple[ArtifactRef, ...],
        completion_status: CompletionStatus,
        consumer_stage: str | None = "decoding",
        artifact_schema_name: str | None = None,
    ) -> "ResidualFieldArtifactManifest":
        validate_residual_field_work_unit(work_unit)
        return cls(
            artifact_key=work_unit.artifact_key,
            artifacts=artifacts,
            completion_status=completion_status,
            retry=work_unit.retry,
            interval_id=work_unit.interval_id,
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            consumer_stage=consumer_stage,
            upstream_artifacts=work_unit.source_artifacts,
            artifact_schema_name=artifact_schema_name or RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.name,
        )


def validate_residual_field_artifact_manifest(
    manifest: ResidualFieldArtifactManifest,
) -> None:
    if manifest.producer_stage != "residual_field":
        raise ValueError(
            "Residual-field artifact manifests must be produced by the residual_field stage."
        )
    if manifest.schema_version != RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unexpected residual-field artifact-manifest schema version.")
    artifact_keys = [artifact.key for artifact in manifest.artifacts]
    if len(set(artifact_keys)) != len(artifact_keys):
        raise ValueError(
            "Residual-field artifact manifests must not contain duplicate artifact keys."
        )
    if manifest.artifact_schema_name != RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.name:
        raise ValueError("Residual-field artifact manifest schema name does not match its scope.")


@dataclass(frozen=True)
class ResidualFieldShardManifest:
    """Immutable map-output checkpoint manifest for one residual-field interval/chunk shard."""

    artifact_key: str
    artifacts: tuple[ArtifactRef, ...]
    completion_status: CompletionStatus
    retry: RetryIdempotencySemantics
    interval_id: int
    contributing_interval_ids: tuple[int, ...]
    chunk_id: int
    parameter_digest: str
    point_count: int
    contribution_reciprocal_point_count: int
    total_reciprocal_point_count: int
    scratch_root: str | None = None
    producer_stage: str = "residual_field"
    consumer_stage: str | None = "residual_field.reducer"
    upstream_artifacts: tuple[ArtifactRef, ...] = ()
    artifact_schema_name: str = RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.name
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_work_unit(
        cls,
        work_unit: ResidualFieldWorkUnit,
        *,
        artifacts: tuple[ArtifactRef, ...],
        completion_status: CompletionStatus,
        point_count: int,
        contribution_reciprocal_point_count: int,
        total_reciprocal_point_count: int,
    ) -> "ResidualFieldShardManifest":
        validate_residual_field_work_unit(work_unit)
        if work_unit.interval_id is None:
            raise ValueError("Shard manifests require interval-scoped residual-field work units.")
        return cls(
            artifact_key=work_unit.artifact_key,
            artifacts=artifacts,
            completion_status=completion_status,
            retry=work_unit.retry,
            interval_id=work_unit.interval_id,
            contributing_interval_ids=tuple(work_unit.interval_ids or (work_unit.interval_id,)),
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            point_count=int(point_count),
            contribution_reciprocal_point_count=int(contribution_reciprocal_point_count),
            total_reciprocal_point_count=int(total_reciprocal_point_count),
            upstream_artifacts=work_unit.source_artifacts,
        )


def validate_residual_field_shard_manifest(
    manifest: ResidualFieldShardManifest,
) -> None:
    if manifest.producer_stage != "residual_field":
        raise ValueError("Residual-field shard manifests must be produced by residual_field.")
    if manifest.schema_version != RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unexpected residual-field shard-manifest schema version.")
    if manifest.interval_id < 0 or manifest.chunk_id < 0:
        raise ValueError("Residual-field shard manifests require non-negative interval/chunk ids.")
    if len(set(manifest.contributing_interval_ids)) != len(manifest.contributing_interval_ids):
        raise ValueError("Residual-field shard manifests require unique contributing interval ids.")
    if not manifest.contributing_interval_ids:
        raise ValueError("Residual-field shard manifests require at least one contributing interval id.")
    if manifest.interval_id != manifest.contributing_interval_ids[0]:
        raise ValueError("Residual-field shard manifest interval_id must match the first contributing interval id.")
    if manifest.point_count < 0:
        raise ValueError("Residual-field shard manifests require non-negative point_count.")
    if manifest.contribution_reciprocal_point_count < 0:
        raise ValueError("Residual-field shard manifests require non-negative reciprocal counts.")
    artifact_keys = [artifact.key for artifact in manifest.artifacts]
    if len(set(artifact_keys)) != len(artifact_keys):
        raise ValueError("Residual-field shard manifests must not contain duplicate artifact keys.")
    if manifest.artifact_schema_name != RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.name:
        raise ValueError("Residual-field shard manifest schema name does not match shard scope.")


@dataclass(frozen=True)
class ResidualFieldReducerProgressManifest:
    artifact: ArtifactRef
    reducer_key: str
    chunk_id: int
    parameter_digest: str
    completion_status: CompletionStatus
    incorporated_shard_keys: tuple[str, ...]
    incorporated_interval_ids: tuple[int, ...]
    reclaimable_shard_keys: tuple[str, ...]
    final_artifacts: tuple[ArtifactRef, ...]
    durable_truth_unit: str = "committed_shard_checkpoint"
    pending_shard_keys: tuple[str, ...] = ()
    pending_interval_ids: tuple[int, ...] = ()
    cleanup_policy: str = "off"
    schema_version: int = RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION


def validate_residual_field_reducer_progress_manifest(
    manifest: ResidualFieldReducerProgressManifest,
) -> None:
    if manifest.artifact.kind != "residual-reducer-progress-manifest":
        raise ValueError("Reducer progress manifest must use the reducer-progress artifact kind.")
    if manifest.durable_truth_unit not in {
        "committed_local_snapshot_generation",
        "committed_shard_checkpoint",
    }:
        raise ValueError(
            "Reducer progress manifest durable_truth_unit must be "
            "'committed_local_snapshot_generation' or 'committed_shard_checkpoint'."
        )
    if len(set(manifest.incorporated_shard_keys)) != len(manifest.incorporated_shard_keys):
        raise ValueError("Reducer progress manifest must not contain duplicate shard keys.")
    if len(set(manifest.incorporated_interval_ids)) != len(manifest.incorporated_interval_ids):
        raise ValueError("Reducer progress manifest must not contain duplicate interval ids.")
    if len(set(manifest.pending_shard_keys)) != len(manifest.pending_shard_keys):
        raise ValueError("Reducer progress manifest must not contain duplicate pending shard keys.")
    if len(set(manifest.pending_interval_ids)) != len(manifest.pending_interval_ids):
        raise ValueError("Reducer progress manifest must not contain duplicate pending interval ids.")
    if len(set(manifest.reclaimable_shard_keys)) != len(manifest.reclaimable_shard_keys):
        raise ValueError("Reducer progress manifest must not contain duplicate reclaimable shard keys.")
    if set(manifest.pending_shard_keys) & set(manifest.reclaimable_shard_keys):
        raise ValueError("Pending shard keys must not already be reclaimable.")
    if manifest.cleanup_policy not in {"off", "delete_reclaimable"}:
        raise ValueError("Reducer progress cleanup_policy must be 'off' or 'delete_reclaimable'.")


__all__ = [
    "RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION",
    "RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA",
    "RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA",
    "RESIDUAL_FIELD_REDUCER_PROGRESS_SCHEMA",
    "RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS",
    "ResidualFieldAccumulatorStatus",
    "ResidualFieldArtifactManifest",
    "ResidualFieldPartialResult",
    "ResidualFieldReducerProgressManifest",
    "ResidualFieldShardManifest",
    "ResidualFieldWorkUnit",
    "build_legacy_residual_field_output_artifacts",
    "build_residual_field_interval_source_artifacts",
    "build_residual_field_output_artifacts",
    "build_residual_field_shard_artifacts",
    "build_residual_field_source_artifacts",
    "make_residual_field_batch_token",
    "make_residual_field_artifact_key",
    "make_residual_field_reducer_key",
    "make_residual_field_retry_key",
    "merge_residual_field_partial_results",
    "residual_field_partial_result_identity",
    "validate_residual_field_artifact_manifest",
    "validate_residual_field_partial_result",
    "validate_residual_field_reducer_progress_manifest",
    "validate_residual_field_shard_manifest",
    "validate_residual_field_work_unit",
]
