from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import reciprocal_space_points_counter, to_interval_dict


@dataclass(frozen=True)
class ScatteringExecutionPlan:
    interval_work_units: tuple[ScatteringWorkUnit, ...]
    chunk_work_units: tuple[ScatteringWorkUnit, ...]
    chunk_ids: tuple[int, ...]
    total_reciprocal_points: int


def build_scattering_interval_lookup(
    reciprocal_space_intervals: list[dict],
) -> dict[int, dict]:
    return {
        int(interval["id"]): interval
        for interval in reciprocal_space_intervals
    }


def build_scattering_precompute_work_units(
    reciprocal_space_intervals: list[dict],
    *,
    dimension: int,
    output_dir: str,
) -> list[ScatteringWorkUnit]:
    return [
        ScatteringWorkUnit.precompute_interval(
            interval_id=int(interval["id"]),
            dimension=dimension,
            output_dir=output_dir,
        )
        for interval in sorted(reciprocal_space_intervals, key=lambda item: int(item["id"]))
    ]


def build_scattering_interval_chunk_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    dimension: int,
    output_dir: str,
) -> list[ScatteringWorkUnit]:
    return [
        ScatteringWorkUnit.interval_chunk(
            interval_id=int(interval_id),
            chunk_id=int(chunk_id),
            dimension=dimension,
            output_dir=output_dir,
        )
        for interval_id, chunk_id in sorted(
            {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
        )
    ]


def chunk_ids_for_work_units(work_units: list[ScatteringWorkUnit]) -> list[int]:
    return sorted(
        {
            int(work_unit.chunk_id)
            for work_unit in work_units
            if work_unit.chunk_id is not None
        }
    )


def interval_ids_for_work_units(work_units: list[ScatteringWorkUnit]) -> list[int]:
    return sorted({int(work_unit.interval_id) for work_unit in work_units})


def interval_paths_for_work_units(work_units: list[ScatteringWorkUnit]) -> dict[int, Path]:
    interval_paths: dict[int, Path] = {}
    for work_unit in work_units:
        if work_unit.interval_artifact is None or work_unit.interval_artifact.path is None:
            continue
        interval_paths[int(work_unit.interval_id)] = Path(work_unit.interval_artifact.path)
    return interval_paths


def build_scattering_execution_plan(
    *,
    parameters: dict,
    db_manager,
    output_dir: str,
) -> ScatteringExecutionPlan:
    supercell = np.asarray(parameters["supercell"])
    dimension = int(len(supercell))
    interval_work_units = tuple(
        build_scattering_precompute_work_units(
            list(parameters["reciprocal_space_intervals"]),
            dimension=dimension,
            output_dir=output_dir,
        )
    )
    chunk_work_units = tuple(
        build_scattering_interval_chunk_work_units(
            list(db_manager.get_unsaved_interval_chunks()),
            dimension=dimension,
            output_dir=output_dir,
        )
    )
    total_reciprocal_points = sum(
        reciprocal_space_points_counter(to_interval_dict(interval), supercell)
        for interval in parameters["reciprocal_space_intervals_all"]
    )
    return ScatteringExecutionPlan(
        interval_work_units=interval_work_units,
        chunk_work_units=chunk_work_units,
        chunk_ids=tuple(chunk_ids_for_work_units(list(chunk_work_units))),
        total_reciprocal_points=int(total_reciprocal_points),
    )


__all__ = [
    "ScatteringExecutionPlan",
    "build_scattering_interval_chunk_work_units",
    "build_scattering_execution_plan",
    "build_scattering_interval_lookup",
    "build_scattering_precompute_work_units",
    "chunk_ids_for_work_units",
    "interval_ids_for_work_units",
    "interval_paths_for_work_units",
]
