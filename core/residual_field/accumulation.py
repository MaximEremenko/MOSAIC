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
    ResidualFieldWorkUnit,
    merge_residual_field_partial_results,
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
        parameter_digest=work_unit.parameter_digest,
        output_kind="residual-field-chunk",
        source_artifacts=work_unit.source_artifacts,
        output_artifacts=output_artifacts,
        grid_shape=_grid_shape_tuple(grid_shape_nd),
        point_ids=tuple(int(point_id) for point_id in payload.point_ids),
    )
    return MaterializedResidualFieldState(metadata=metadata, payload=payload)


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
        parameter_digest=parameter_digest,
        output_kind="residual-field-chunk",
        source_artifacts=(),
        output_artifacts=output_artifacts,
        grid_shape=_grid_shape_tuple(grid_shape_nd),
        point_ids=tuple(int(point_id) for point_id in payload.point_ids),
    )
    return MaterializedResidualFieldState(metadata=metadata, payload=payload)


def merge_materialized_residual_field_states(
    left: MaterializedResidualFieldState,
    right: MaterializedResidualFieldState,
) -> MaterializedResidualFieldState:
    return MaterializedResidualFieldState(
        metadata=merge_residual_field_partial_results(left.metadata, right.metadata),
        payload=merge_scattering_partial_results(left.payload, right.payload),
    )


__all__ = [
    "MaterializedResidualFieldState",
    "build_existing_materialized_residual_field_state",
    "build_materialized_residual_field_state",
    "materialize_scattering_payload",
    "merge_materialized_residual_field_states",
]
