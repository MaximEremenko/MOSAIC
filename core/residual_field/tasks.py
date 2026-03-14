from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.scattering.kernels import build_rifft_grid_for_chunk
from core.scattering.tasks import load_interval_task_payload, scattering_contribution_point_count
from core.residual_field.artifacts import persist_residual_field_interval_chunk_result
from core.residual_field.contracts import ResidualFieldArtifactManifest, ResidualFieldWorkUnit
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft,
)
from core.runtime import handle_worker_gpu_failure


logger = logging.getLogger(__name__)


def run_residual_field_interval_chunk_task(
    work_unit: ResidualFieldWorkUnit,
    interval_path: Path,
    atoms: np.recarray,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    db_path: str,
    quiet_logs: bool = False,
) -> ResidualFieldArtifactManifest | None:
    interval_id = work_unit.interval_id
    try:
        interval_task = load_interval_task_payload(interval_path)
        chunk_data = [
            {
                "coordinates": atoms["coordinates"][index],
                "dist_from_atom_center": atoms["dist_from_atom_center"][index],
                "step_in_frac": atoms["step_in_frac"][index],
            }
            for index in range(atoms.shape[0])
        ]
        rifft_grid, grid_shape_nd = build_rifft_grid_for_chunk(chunk_data)
        amplitudes_delta = execute_inverse_cunufft(
            q_coords=interval_task.q_grid,
            c=interval_task.q_amp - interval_task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        amplitudes_average = execute_inverse_cunufft(
            q_coords=interval_task.q_grid,
            c=interval_task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        return persist_residual_field_interval_chunk_result(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=scattering_contribution_point_count(interval_task),
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_ids=np.arange(amplitudes_delta.shape[0], dtype=np.int64),
            output_dir=output_dir,
            db_path=db_path,
            quiet_logs=quiet_logs,
        )
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            work_unit.chunk_id,
            interval_id if interval_id is not None else "n/a",
            err,
            exc_info=True,
        )
        handle_worker_gpu_failure(err, logger=logger)
        return None


__all__ = ["run_residual_field_interval_chunk_task"]
