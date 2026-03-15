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
    merge_residual_field_partial_results,
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
        point_ids=tuple(int(point_id) for point_id in payload.point_ids),
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
        point_ids=tuple(int(point_id) for point_id in payload.point_ids),
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
        point_ids=tuple(int(point_id) for point_id in payload.point_ids),
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
    payload_point_ids = tuple(int(point_id) for point_id in state.payload.point_ids)
    if state.metadata.point_ids != payload_point_ids:
        raise ValueError("Residual-field metadata point_ids must match the payload point_ids.")
    if state.metadata.reciprocal_point_count != state.payload.reciprocal_point_count:
        raise ValueError(
            "Residual-field metadata reciprocal_point_count must match the payload."
        )
    if state.metadata.residual_values is None or state.metadata.residual_average_values is None:
        raise ValueError("Residual-field metadata must materialize residual arrays in Phase 6.")
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
    "validate_materialized_residual_field_state",
]
