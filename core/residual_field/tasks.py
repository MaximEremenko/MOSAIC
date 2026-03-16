from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from core.residual_field.backend import (
    ResidualFieldReducerBackend,
    ResidualFieldLocalAccumulatorPartial,
    build_residual_field_reducer_backend,
)
from core.scattering.kernels import build_rifft_grid_for_chunk
from core.scattering.kernels import IntervalTask
from core.scattering.tasks import load_interval_task_payload, scattering_contribution_point_count
from core.residual_field.contracts import ResidualFieldShardManifest, ResidualFieldWorkUnit
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_super_batch,
)
from core.runtime import handle_worker_gpu_failure, task_progress_enabled


logger = logging.getLogger(__name__)


def _task_progress_enabled(quiet_logs: bool) -> bool:
    if not quiet_logs:
        return True
    return task_progress_enabled(False)


def _normalize_interval_inputs(
    interval_inputs: Path | str | IntervalTask | Sequence[Path | str | IntervalTask],
) -> tuple[Path | IntervalTask, ...]:
    if isinstance(interval_inputs, IntervalTask):
        return (interval_inputs,)
    if isinstance(interval_inputs, (str, Path)):
        return (Path(interval_inputs),)
    normalized: list[Path | IntervalTask] = []
    for item in interval_inputs:
        if isinstance(item, IntervalTask):
            normalized.append(item)
        else:
            normalized.append(Path(item))
    return tuple(normalized)


def _q_grid_signature(q_grid: np.ndarray) -> tuple:
    arr = np.asarray(q_grid)
    return (tuple(arr.shape), str(arr.dtype), arr.tobytes())


def run_residual_field_interval_chunk_task(
    work_unit: ResidualFieldWorkUnit,
    interval_paths: Path | str | IntervalTask | Sequence[Path | str | IntervalTask],
    atoms: np.recarray,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    scratch_root: str | None = None,
    reducer_backend: ResidualFieldReducerBackend | None = None,
    quiet_logs: bool = False,
) -> ResidualFieldShardManifest | ResidualFieldLocalAccumulatorPartial | None:
    interval_ids = work_unit.interval_ids or ((work_unit.interval_id,) if work_unit.interval_id is not None else ())
    try:
        show_progress = _task_progress_enabled(quiet_logs)
        resolved_backend = reducer_backend or build_residual_field_reducer_backend(
            "local_restartable"
        )
        loaded_interval_inputs = _normalize_interval_inputs(interval_paths)
        if not loaded_interval_inputs:
            raise ValueError("Residual-field batch task requires at least one interval artifact path.")
        chunk_data = [
            {
                "coordinates": atoms["coordinates"][index],
                "dist_from_atom_center": atoms["dist_from_atom_center"][index],
                "step_in_frac": atoms["step_in_frac"][index],
            }
            for index in range(atoms.shape[0])
        ]
        rifft_grid, grid_shape_nd = build_rifft_grid_for_chunk(chunk_data)
        if show_progress:
            logger.info(
                "Residual batch start | chunk=%d | intervals=%s | rifft_points=%d",
                int(work_unit.chunk_id),
                ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
                int(rifft_grid.shape[0]),
            )
        contribution_reciprocal_points = 0
        interval_tasks = [
            interval_input
            if isinstance(interval_input, IntervalTask)
            else load_interval_task_payload(interval_input)
            for interval_input in loaded_interval_inputs
        ]
        grouped_interval_tasks: dict[tuple, list] = {}
        for interval_task in interval_tasks:
            grouped_interval_tasks.setdefault(
                _q_grid_signature(interval_task.q_grid),
                [],
            ).append(interval_task)
            contribution_reciprocal_points += scattering_contribution_point_count(interval_task)
        if not grouped_interval_tasks:
            raise ValueError("Residual-field batch task produced no interval contributions.")
        amplitudes_delta = None
        amplitudes_average = None
        total_groups = len(grouped_interval_tasks)
        for group_index, grouped_tasks in enumerate(grouped_interval_tasks.values(), start=1):
            reference_q_grid = grouped_tasks[0].q_grid
            if show_progress:
                logger.info(
                    "Residual batch group | chunk=%d | group=%d/%d | intervals=%d | q_points=%d",
                    int(work_unit.chunk_id),
                    int(group_index),
                    int(total_groups),
                    int(len(grouped_tasks)),
                    int(reference_q_grid.shape[0]),
                )
            stacked_weights = []
            for interval_task in grouped_tasks:
                stacked_weights.extend(
                    [
                        interval_task.q_amp - interval_task.q_amp_av,
                        interval_task.q_amp_av,
                    ]
                )
            inverse_outputs = execute_inverse_cunufft_super_batch(
                q_coords=reference_q_grid,
                weights=np.stack(stacked_weights, axis=0),
                real_coords=rifft_grid,
                eps=1e-12,
            )
            grouped_delta = np.asarray(
                np.sum(inverse_outputs[0::2], axis=0),
                dtype=np.complex128,
            )
            grouped_average = np.asarray(
                np.sum(inverse_outputs[1::2], axis=0),
                dtype=np.complex128,
            )
            amplitudes_delta = (
                grouped_delta
                if amplitudes_delta is None
                else amplitudes_delta + grouped_delta
            )
            amplitudes_average = (
                grouped_average
                if amplitudes_average is None
                else amplitudes_average + grouped_average
            )
        point_ids = np.arange(amplitudes_delta.shape[0], dtype=np.int64)
        if resolved_backend.uses_local_chunk_accumulator():
            if show_progress:
                logger.info(
                    "Residual batch ready for local accumulator | chunk=%d | intervals=%s",
                    int(work_unit.chunk_id),
                    ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
                )
            return resolved_backend.build_local_partial(
                work_unit,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                contribution_reciprocal_points=contribution_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
                point_ids=point_ids,
            )
        if show_progress:
            logger.info(
                "Residual batch persisting durable shard | chunk=%d | intervals=%s",
                int(work_unit.chunk_id),
                ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
            )
        return resolved_backend.persist_shard_checkpoint(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_ids=point_ids,
            output_dir=output_dir,
            scratch_root=scratch_root,
            quiet_logs=quiet_logs,
        )
    except Exception as err:
        logger.error(
            "chunk %d | batch %s FAILED: %s",
            work_unit.chunk_id,
            ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
            err,
            exc_info=True,
        )
        handle_worker_gpu_failure(err, logger=logger)
        return None


__all__ = ["run_residual_field_interval_chunk_task"]
