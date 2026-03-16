from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import replace
from dataclasses import asdict, is_dataclass

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit

_RESIDUAL_GRID_VALUE_BYTES_PER_POINT = 40


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in sorted(asdict(value).items())}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items())}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, int, float, bool)):
        return {
            str(key): _to_jsonable(val)
            for key, val in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return value


def _parameter_value(parameters: object, key: str, default=None):
    if isinstance(parameters, dict):
        return parameters.get(key, default)
    return getattr(parameters, key, default)


def build_residual_field_parameter_digest(parameters: object) -> str:
    payload = {
        "postprocessing_mode": _parameter_value(parameters, "postprocessing_mode"),
        "supercell": _to_jsonable(_parameter_value(parameters, "supercell")),
        "rspace_info": _to_jsonable(_parameter_value(parameters, "rspace_info", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _batch_interval_chunks(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    max_intervals_per_shard: int,
) -> list[tuple[int, tuple[int, ...]]]:
    if max_intervals_per_shard <= 0:
        raise ValueError("max_intervals_per_shard must be positive.")
    grouped: dict[int, list[int]] = {}
    for interval_id, chunk_id in sorted(
        {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
    ):
        grouped.setdefault(int(chunk_id), []).append(int(interval_id))
    batches: list[tuple[int, tuple[int, ...]]] = []
    for chunk_id in sorted(grouped):
        interval_ids = grouped[chunk_id]
        for start in range(0, len(interval_ids), max_intervals_per_shard):
            batches.append((chunk_id, tuple(interval_ids[start : start + max_intervals_per_shard])))
    return batches


def build_residual_field_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    parameters: object,
    output_dir: str,
    max_intervals_per_shard: int = 1,
) -> list[ResidualFieldWorkUnit]:
    digest = build_residual_field_parameter_digest(parameters)
    patch_scope = "chunk"
    window_spec = str(_parameter_value(parameters, "postprocessing_mode", "displacement"))
    work_units: list[ResidualFieldWorkUnit] = []
    for chunk_id, interval_ids in _batch_interval_chunks(
        unsaved_interval_chunks,
        max_intervals_per_shard=max_intervals_per_shard,
    ):
        if len(interval_ids) == 1:
            work_units.append(
                ResidualFieldWorkUnit.interval_chunk(
                    interval_id=int(interval_ids[0]),
                    chunk_id=int(chunk_id),
                    parameter_digest=digest,
                    output_dir=output_dir,
                    patch_scope=patch_scope,
                    window_spec=window_spec,
                )
            )
            continue
        work_units.append(
            ResidualFieldWorkUnit.interval_chunk_batch(
                interval_ids=interval_ids,
                chunk_id=int(chunk_id),
                parameter_digest=digest,
                output_dir=output_dir,
                patch_scope=patch_scope,
                window_spec=window_spec,
            )
        )
    return work_units


@dataclass(frozen=True)
class ResidualFieldAdaptivePartitionPlan:
    chunk_id: int
    dimensionality: int
    point_count: int
    estimated_rifft_points: int
    estimated_bytes: int
    target_partition_bytes: int
    target_partitions: int
    reason: str
    rifft_points_per_atom: tuple[int, ...] = ()
    partition_rifft_points: tuple[int, ...] = ()
    partition_imbalance_ratio: float = 1.0


def _axis_sample_count(dist: float, step: float) -> int:
    if step <= 0 or dist <= step:
        return 1
    eps = 1e-8
    return int(np.floor(((2.0 * float(dist)) + float(step) - eps) / float(step))) + 1


def estimate_rifft_points_for_point(point_row: dict[str, object]) -> int:
    dist = np.asarray(point_row["dist_from_atom_center"], dtype=float).reshape(-1)
    step = np.asarray(point_row["step_in_frac"], dtype=float).reshape(-1)
    if dist.shape != step.shape:
        raise ValueError("dist_from_atom_center and step_in_frac must have matching shape.")
    point_count = 1
    for axis_dist, axis_step in zip(dist, step):
        point_count *= _axis_sample_count(float(axis_dist), float(axis_step))
    return int(point_count)


def estimate_chunk_partition_bytes(
    point_rows: list[dict[str, object]],
) -> tuple[int, int, int]:
    if not point_rows:
        return 1, 0, 0
    sample_row = point_rows[0]
    coordinate_like = sample_row.get("coordinates")
    if coordinate_like is None:
        coordinate_like = sample_row.get("dist_from_atom_center")
    if coordinate_like is None:
        coordinate_like = sample_row.get("step_in_frac")
    dimensionality = (
        int(len(np.asarray(coordinate_like).reshape(-1)))
        if coordinate_like is not None
        else 1
    )
    rifft_points_per_atom = _rifft_points_per_atom(point_rows)
    estimated_rifft_points = int(np.sum(rifft_points_per_atom, dtype=np.int64))
    estimated_bytes = int(estimated_rifft_points) * _RESIDUAL_GRID_VALUE_BYTES_PER_POINT
    estimated_bytes += int(len(point_rows)) * dimensionality * 8
    return dimensionality, int(estimated_rifft_points), int(estimated_bytes)


def _rifft_points_per_atom(point_rows: list[dict[str, object]]) -> np.ndarray:
    if not point_rows:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(
        [
            estimate_rifft_points_for_point(point_row)
            if "dist_from_atom_center" in point_row and "step_in_frac" in point_row
            else 1
            for point_row in point_rows
        ],
        dtype=np.int64,
    )


def _partition_rifft_points(
    *,
    rifft_points_per_atom: np.ndarray,
    target_partitions: int,
) -> tuple[int, ...]:
    if rifft_points_per_atom.size == 0:
        return ()
    point_indices = np.arange(rifft_points_per_atom.shape[0], dtype=np.int64)
    return tuple(
        int(np.sum(rifft_points_per_atom[selection], dtype=np.int64))
        for selection in _weighted_partition_split(
            point_indices,
            rifft_points_per_atom,
            int(target_partitions),
        )
        if selection.size > 0
    )


def build_adaptive_partition_plan(
    point_rows_by_chunk: dict[int, list[dict[str, object]]],
    *,
    effective_nufft_workers: int,
    target_partition_bytes: int,
    target_partition_bytes_3d: int | None = None,
    max_partitions_per_chunk: int | None = None,
    min_points_per_partition: int = 1,
) -> dict[int, ResidualFieldAdaptivePartitionPlan]:
    if effective_nufft_workers <= 0:
        raise ValueError("effective_nufft_workers must be positive.")
    if target_partition_bytes <= 0:
        raise ValueError("target_partition_bytes must be positive.")
    if min_points_per_partition <= 0:
        raise ValueError("min_points_per_partition must be positive.")
    if target_partition_bytes_3d is None:
        target_partition_bytes_3d = max(1, int(target_partition_bytes) // 2)
    if target_partition_bytes_3d <= 0:
        raise ValueError("target_partition_bytes_3d must be positive.")
    if max_partitions_per_chunk is None:
        max_partitions_per_chunk = max(1, int(effective_nufft_workers) * 2)
    if max_partitions_per_chunk <= 0:
        raise ValueError("max_partitions_per_chunk must be positive.")

    plans: dict[int, ResidualFieldAdaptivePartitionPlan] = {}
    chunk_count = max(1, len(point_rows_by_chunk))
    for chunk_id, point_rows in sorted(point_rows_by_chunk.items()):
        rifft_points_per_atom = _rifft_points_per_atom(point_rows)
        dimensionality, estimated_rifft_points, estimated_bytes = estimate_chunk_partition_bytes(point_rows)
        point_count = int(len(point_rows))
        target_bytes = (
            int(target_partition_bytes_3d)
            if dimensionality >= 3
            else int(target_partition_bytes)
        )
        worker_floor = (
            max(1, int(effective_nufft_workers) // chunk_count)
            if chunk_count < int(effective_nufft_workers)
            else 1
        )
        byte_budget_count = max(1, int(math.ceil(float(estimated_bytes) / float(target_bytes))))
        raw_target = max(worker_floor, byte_budget_count)
        max_by_points = max(1, point_count // int(min_points_per_partition))
        target = min(int(max_partitions_per_chunk), int(max_by_points), int(point_count), int(raw_target))
        if target <= 1:
            reason = "whole-chunk"
        elif byte_budget_count > worker_floor and dimensionality >= 3:
            reason = "3d-byte-budget"
        elif byte_budget_count > worker_floor:
            reason = "byte-budget"
        elif worker_floor > 1:
            reason = "worker-capacity-floor"
        else:
            reason = "partition-cap"
        partition_rifft_points = _partition_rifft_points(
            rifft_points_per_atom=rifft_points_per_atom,
            target_partitions=int(max(1, target)),
        )
        if partition_rifft_points:
            min_partition_weight = min(partition_rifft_points)
            max_partition_weight = max(partition_rifft_points)
            partition_imbalance_ratio = (
                float(max_partition_weight) / float(min_partition_weight)
                if min_partition_weight > 0
                else float("inf")
            )
        else:
            partition_imbalance_ratio = 1.0
        plans[int(chunk_id)] = ResidualFieldAdaptivePartitionPlan(
            chunk_id=int(chunk_id),
            dimensionality=int(dimensionality),
            point_count=int(point_count),
            estimated_rifft_points=int(estimated_rifft_points),
            estimated_bytes=int(estimated_bytes),
            target_partition_bytes=int(target_bytes),
            target_partitions=int(max(1, target)),
            reason=reason,
            rifft_points_per_atom=tuple(int(value) for value in rifft_points_per_atom.tolist()),
            partition_rifft_points=partition_rifft_points,
            partition_imbalance_ratio=float(partition_imbalance_ratio),
        )
    return plans


def _weighted_partition_split(
    point_indices: np.ndarray,
    rifft_points_per_atom: np.ndarray,
    target_partitions: int,
) -> list[np.ndarray]:
    indices = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    weights = np.asarray(rifft_points_per_atom, dtype=np.int64).reshape(-1)
    if indices.shape[0] != weights.shape[0]:
        raise ValueError("point_indices and rifft_points_per_atom must have matching length.")
    point_count = int(indices.shape[0])
    if point_count == 0:
        return []
    target_partitions = max(1, min(int(target_partitions), point_count))
    if target_partitions == 1:
        return [indices]
    weights = np.maximum(weights, 1)
    total_weight = int(np.sum(weights, dtype=np.int64))
    if total_weight <= 0:
        return [
            selection
            for selection in np.array_split(indices, target_partitions)
            if selection.size > 0
        ]
    cumulative = np.cumsum(weights, dtype=np.int64)
    partitions: list[np.ndarray] = []
    start = 0
    for part_index in range(1, target_partitions):
        threshold = float(total_weight) * (float(part_index) / float(target_partitions))
        stop = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        min_stop = start + 1
        max_stop = point_count - (target_partitions - part_index)
        stop = max(min_stop, min(stop, max_stop))
        partitions.append(indices[start:stop])
        start = stop
    partitions.append(indices[start:point_count])
    return [selection for selection in partitions if selection.size > 0]


def partition_residual_field_work_units(
    work_units: list[ResidualFieldWorkUnit],
    *,
    point_counts_by_chunk: dict[int, int],
    target_partitions_by_chunk: dict[int, int],
    rifft_points_by_chunk: dict[int, np.ndarray] | None = None,
) -> list[ResidualFieldWorkUnit]:
    partitioned: list[ResidualFieldWorkUnit] = []
    for work_unit in work_units:
        chunk_id = int(work_unit.chunk_id)
        target_partitions = int(target_partitions_by_chunk.get(chunk_id, 1))
        if target_partitions <= 1:
            partitioned.append(work_unit)
            continue
        point_count = int(point_counts_by_chunk.get(chunk_id, 0))
        if point_count <= 1:
            partitioned.append(work_unit)
            continue
        target_partitions = min(target_partitions, point_count)
        point_indices = np.arange(point_count, dtype=np.int64)
        rifft_weights = None
        if rifft_points_by_chunk is not None:
            rifft_weights = rifft_points_by_chunk.get(chunk_id)
        selections = (
            _weighted_partition_split(point_indices, rifft_weights, target_partitions)
            if rifft_weights is not None
            else [
                selection
                for selection in np.array_split(point_indices, target_partitions)
                if selection.size > 0
            ]
        )
        for partition_id, selection in enumerate(selections):
            if selection.size == 0:
                continue
            partitioned.append(
                work_unit.with_partition(
                    partition_id=int(partition_id),
                    point_start=int(selection[0]),
                    point_stop=int(selection[-1]) + 1,
                )
            )
    return partitioned


def chunk_ids_for_work_units(work_units: list[ResidualFieldWorkUnit]) -> list[int]:
    return sorted({int(work_unit.chunk_id) for work_unit in work_units})


__all__ = [
    "_batch_interval_chunks",
    "build_residual_field_parameter_digest",
    "build_adaptive_partition_plan",
    "build_residual_field_work_units",
    "chunk_ids_for_work_units",
    "estimate_chunk_partition_bytes",
    "estimate_rifft_points_for_point",
    "_weighted_partition_split",
    "partition_residual_field_work_units",
    "ResidualFieldAdaptivePartitionPlan",
]
