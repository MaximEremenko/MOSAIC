from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from core.scattering.kernels import build_rifft_grid_for_chunk
from core.scattering.tasks import load_interval_task_payload, scattering_contribution_point_count
from core.residual_field.artifacts import persist_residual_field_shard_checkpoint
from core.residual_field.contracts import ResidualFieldShardManifest, ResidualFieldWorkUnit
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_super_batch,
)
from core.runtime import handle_worker_gpu_failure


logger = logging.getLogger(__name__)


def _normalize_interval_paths(interval_paths: Path | str | Sequence[Path | str]) -> tuple[Path, ...]:
    if isinstance(interval_paths, (str, Path)):
        return (Path(interval_paths),)
    return tuple(Path(path) for path in interval_paths)


def _q_grid_signature(q_grid: np.ndarray) -> tuple:
    arr = np.asarray(q_grid)
    return (tuple(arr.shape), str(arr.dtype), arr.tobytes())


def run_residual_field_interval_chunk_task(
    work_unit: ResidualFieldWorkUnit,
    interval_paths: Path | str | Sequence[Path | str],
    atoms: np.recarray,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    scratch_root: str | None = None,
    quiet_logs: bool = False,
) -> ResidualFieldShardManifest | None:
    interval_ids = work_unit.interval_ids or ((work_unit.interval_id,) if work_unit.interval_id is not None else ())
    try:
        loaded_interval_paths = _normalize_interval_paths(interval_paths)
        if not loaded_interval_paths:
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
        contribution_reciprocal_points = 0
        interval_tasks = [load_interval_task_payload(interval_path) for interval_path in loaded_interval_paths]
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
        for grouped_tasks in grouped_interval_tasks.values():
            reference_q_grid = grouped_tasks[0].q_grid
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
        return persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_ids=np.arange(amplitudes_delta.shape[0], dtype=np.int64),
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
