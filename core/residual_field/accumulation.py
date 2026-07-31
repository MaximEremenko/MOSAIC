from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.scattering.accumulation import (
    ScatteringPartialResult,
    build_scattering_partial_result,
    build_scattering_partial_result_from_payloads,
    materialize_scattering_payload,
    merge_scattering_partial_results,
)
from core.residual_field.contracts import (
    ResidualFieldPartialResult,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
    point_ids_equal,
    validate_residual_field_partial_result,
)
from core.contracts import ArtifactRef


@dataclass(frozen=True)
class MaterializedResidualFieldState:
    metadata: ResidualFieldPartialResult
    payload: ScatteringPartialResult


def _grid_shape_tuple(grid_shape_nd: np.ndarray | None) -> tuple[int, ...] | None:
    if grid_shape_nd is None:
        return None
    arr = np.asarray(grid_shape_nd)
    if arr.size == 0:
        return None
    return tuple(int(v) for v in arr.ravel())


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
        and not point_ids_equal(left.point_ids, right.point_ids)
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
        else np.union1d(
            np.asarray(left.point_ids, dtype=np.int64),
            np.asarray(right.point_ids, dtype=np.int64),
        )
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


def build_materialized_residual_field_state(
    work_unit: ResidualFieldWorkUnit,
    *,
    output_artifacts: tuple[ArtifactRef, ...],
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    grid_shape_nd: np.ndarray,
    reciprocal_point_count: int,
    point_ids: np.ndarray | None = None,
) -> MaterializedResidualFieldState:
    if work_unit.interval_id is None:
        raise ValueError("Materialized residual-field state requires interval_id.")
    payload = build_scattering_partial_result(
        chunk_id=work_unit.chunk_id,
        interval_id=work_unit.interval_id,
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
        grid_shape_nd=grid_shape_nd,
        reciprocal_point_count=reciprocal_point_count,
        point_ids=point_ids,
    )
    metadata = ResidualFieldPartialResult(
        chunk_id=work_unit.chunk_id,
        contributing_interval_ids=(int(work_unit.interval_id),),
        parameter_digest=work_unit.parameter_digest,
        output_kind="residual-field-chunk",
        source_artifacts=work_unit.source_artifacts,
        output_artifacts=output_artifacts,
        grid_shape=_grid_shape_tuple(grid_shape_nd),
        point_ids=payload.point_ids,
        residual_values=payload.amplitudes_delta.copy(),
        residual_average_values=payload.amplitudes_average.copy(),
        reciprocal_point_count=payload.reciprocal_point_count,
    )
    return validate_materialized_residual_field_state(
        MaterializedResidualFieldState(metadata=metadata, payload=payload)
    )


def build_existing_materialized_residual_field_state(
    *,
    chunk_id: int,
    parameter_digest: str,
    output_artifacts: tuple[ArtifactRef, ...],
    amplitudes_payload: np.ndarray,
    amplitudes_average_payload: np.ndarray,
    grid_shape_nd: np.ndarray | None,
    reciprocal_point_count: int,
    applied_interval_ids: tuple[int, ...],
) -> MaterializedResidualFieldState:
    payload = build_scattering_partial_result_from_payloads(
        chunk_id=chunk_id,
        contributing_interval_ids=applied_interval_ids,
        amplitudes_payload=amplitudes_payload,
        amplitudes_average_payload=amplitudes_average_payload,
        grid_shape_nd=(
            np.asarray(grid_shape_nd)
            if grid_shape_nd is not None
            else np.array([], dtype=int)
        ),
        reciprocal_point_count=reciprocal_point_count,
    )
    metadata = ResidualFieldPartialResult(
        chunk_id=chunk_id,
        contributing_interval_ids=tuple(int(interval_id) for interval_id in applied_interval_ids),
        parameter_digest=parameter_digest,
        output_kind="residual-field-chunk",
        source_artifacts=(),
        output_artifacts=output_artifacts,
        grid_shape=_grid_shape_tuple(grid_shape_nd),
        point_ids=payload.point_ids,
        residual_values=payload.amplitudes_delta.copy(),
        residual_average_values=payload.amplitudes_average.copy(),
        reciprocal_point_count=payload.reciprocal_point_count,
    )
    return validate_materialized_residual_field_state(
        MaterializedResidualFieldState(metadata=metadata, payload=payload)
    )


def build_materialized_residual_field_state_from_shard(
    manifest: ResidualFieldShardManifest,
    *,
    output_artifacts: tuple[ArtifactRef, ...],
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
) -> MaterializedResidualFieldState:
    payload = build_scattering_partial_result(
        chunk_id=manifest.chunk_id,
        interval_id=manifest.interval_id,
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
        grid_shape_nd=grid_shape_nd,
        reciprocal_point_count=manifest.contribution_reciprocal_point_count,
        point_ids=point_ids,
    )
    metadata = ResidualFieldPartialResult(
        chunk_id=manifest.chunk_id,
        contributing_interval_ids=manifest.contributing_interval_ids,
        parameter_digest=manifest.parameter_digest,
        output_kind="residual-field-chunk",
        source_artifacts=manifest.upstream_artifacts,
        output_artifacts=output_artifacts,
        grid_shape=_grid_shape_tuple(grid_shape_nd),
        point_ids=payload.point_ids,
        residual_values=payload.amplitudes_delta.copy(),
        residual_average_values=payload.amplitudes_average.copy(),
        reciprocal_point_count=payload.reciprocal_point_count,
    )
    return validate_materialized_residual_field_state(
        MaterializedResidualFieldState(metadata=metadata, payload=payload)
    )


def validate_materialized_residual_field_state(
    state: MaterializedResidualFieldState,
) -> MaterializedResidualFieldState:
    validate_residual_field_partial_result(state.metadata)
    if state.metadata.chunk_id != state.payload.chunk_id:
        raise ValueError("Residual-field metadata and payload must target the same chunk_id.")
    payload_grid_shape = _grid_shape_tuple(state.payload.grid_shape_nd)
    if state.metadata.grid_shape != payload_grid_shape:
        raise ValueError("Residual-field metadata grid_shape must match the payload grid shape.")
    if not point_ids_equal(state.metadata.point_ids, state.payload.point_ids):
        raise ValueError("Residual-field metadata point_ids must match the payload point_ids.")
    if state.metadata.reciprocal_point_count != state.payload.reciprocal_point_count:
        raise ValueError(
            "Residual-field metadata reciprocal_point_count must match the payload."
        )
    if state.metadata.residual_values is None or state.metadata.residual_average_values is None:
        raise ValueError("Residual-field metadata must materialize residual arrays.")
    if not np.array_equal(state.metadata.residual_values, state.payload.amplitudes_delta):
        raise ValueError("Residual-field metadata residual_values must match payload amplitudes_delta.")
    if not np.array_equal(
        state.metadata.residual_average_values,
        state.payload.amplitudes_average,
    ):
        raise ValueError(
            "Residual-field metadata residual_average_values must match payload amplitudes_average."
        )
    return state


def merge_materialized_residual_field_states(
    left: MaterializedResidualFieldState,
    right: MaterializedResidualFieldState,
) -> MaterializedResidualFieldState:
    merged = MaterializedResidualFieldState(
        metadata=merge_residual_field_partial_results(left.metadata, right.metadata),
        payload=merge_scattering_partial_results(left.payload, right.payload),
    )
    return validate_materialized_residual_field_state(merged)


__all__ = [
    "MaterializedResidualFieldState",
    "build_existing_materialized_residual_field_state",
    "build_materialized_residual_field_state",
    "build_materialized_residual_field_state_from_shard",
    "materialize_scattering_payload",
    "merge_materialized_residual_field_states",
    "merge_residual_field_partial_results",
    "validate_materialized_residual_field_state",
]
