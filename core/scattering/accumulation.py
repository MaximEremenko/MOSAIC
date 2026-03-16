from __future__ import annotations

import numpy as np

from core.scattering.contracts import (
    ScatteringPartialResult,
    merge_scattering_partial_results,
    scattering_partial_result_identity,
)


def extract_point_ids_from_payload(amplitude_payload: np.ndarray) -> np.ndarray:
    payload = np.asarray(amplitude_payload)
    if payload.ndim == 2 and payload.shape[1] >= 2:
        return np.asarray(np.rint(np.real(payload[:, 0])), dtype=np.int64)
    return np.arange(payload.shape[0], dtype=np.int64)


def extract_amplitude_values(amplitude_payload: np.ndarray) -> np.ndarray:
    payload = np.asarray(amplitude_payload)
    if payload.ndim == 2 and payload.shape[1] >= 2:
        return np.asarray(payload[:, 1])
    return np.asarray(payload)


def build_scattering_partial_result(
    *,
    chunk_id: int,
    interval_id: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    grid_shape_nd: np.ndarray,
    reciprocal_point_count: int,
    point_ids: np.ndarray | None = None,
) -> ScatteringPartialResult:
    delta_arr = np.asarray(amplitudes_delta).reshape(-1)
    average_arr = np.asarray(amplitudes_average).reshape(-1)
    if point_ids is None:
        point_ids_arr = np.arange(delta_arr.shape[0], dtype=np.int64)
    else:
        point_ids_arr = np.asarray(point_ids, dtype=np.int64)
    return ScatteringPartialResult(
        chunk_id=chunk_id,
        contributing_interval_ids=(int(interval_id),),
        point_ids=point_ids_arr,
        grid_shape_nd=np.asarray(grid_shape_nd),
        amplitudes_delta=delta_arr,
        amplitudes_average=average_arr,
        reciprocal_point_count=int(reciprocal_point_count),
    )


def build_scattering_partial_result_from_payloads(
    *,
    chunk_id: int,
    contributing_interval_ids: tuple[int, ...],
    amplitudes_payload: np.ndarray,
    amplitudes_average_payload: np.ndarray,
    grid_shape_nd: np.ndarray,
    reciprocal_point_count: int,
) -> ScatteringPartialResult:
    point_ids = extract_point_ids_from_payload(amplitudes_payload)
    return ScatteringPartialResult(
        chunk_id=chunk_id,
        contributing_interval_ids=tuple(int(interval_id) for interval_id in contributing_interval_ids),
        point_ids=point_ids,
        grid_shape_nd=np.asarray(grid_shape_nd),
        amplitudes_delta=extract_amplitude_values(amplitudes_payload),
        amplitudes_average=extract_amplitude_values(amplitudes_average_payload),
        reciprocal_point_count=int(reciprocal_point_count),
    )


def materialize_scattering_payload(
    template_payload: np.ndarray | None,
    point_ids: np.ndarray,
    amplitude_values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(amplitude_values)
    if template_payload is None:
        return np.column_stack(
            [
                np.asarray(point_ids, dtype=np.complex128),
                values.astype(np.complex128, copy=False),
            ]
        )

    template = np.asarray(template_payload)
    if template.ndim == 2 and template.shape[1] >= 2:
        out = np.array(template, copy=True)
        out[:, 0] = np.asarray(point_ids, dtype=out.dtype)
        out[:, 1] = values.astype(out.dtype, copy=False)
        return out
    return values.astype(template.dtype if template.dtype else np.complex128, copy=False)


def apply_scattering_partial_result(
    current_rows: np.ndarray,
    current_average_rows: np.ndarray,
    current_reciprocal_point_count: int,
    partial_result: ScatteringPartialResult,
    *,
    mirror_conjugate_symmetry: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.array(current_rows, copy=True)
    average_rows = np.array(current_average_rows, copy=True)
    point_ids = extract_point_ids_from_payload(rows)
    if not np.array_equal(point_ids, partial_result.point_ids):
        raise ValueError("current_rows point ids must match partial_result.point_ids.")

    delta = partial_result.amplitudes_delta
    average_delta = partial_result.amplitudes_average
    if mirror_conjugate_symmetry:
        delta = delta + np.conj(delta)
        average_delta = average_delta + np.conj(average_delta)
        reciprocal_count = current_reciprocal_point_count + (partial_result.reciprocal_point_count * 2)
    else:
        reciprocal_count = current_reciprocal_point_count + partial_result.reciprocal_point_count

    if rows.ndim == 2 and rows.shape[1] >= 2:
        rows[:, 1] = np.asarray(rows[:, 1]) + delta
    else:
        rows = np.asarray(rows) + delta

    if average_rows.ndim == 2 and average_rows.shape[1] >= 2:
        average_rows[:, 1] = np.asarray(average_rows[:, 1]) + average_delta
    else:
        average_rows = np.asarray(average_rows) + average_delta

    return rows, average_rows, int(reciprocal_count)


__all__ = [
    "ScatteringPartialResult",
    "apply_scattering_partial_result",
    "build_scattering_partial_result",
    "build_scattering_partial_result_from_payloads",
    "extract_amplitude_values",
    "extract_point_ids_from_payload",
    "materialize_scattering_payload",
    "merge_scattering_partial_results",
    "scattering_partial_result_identity",
]
