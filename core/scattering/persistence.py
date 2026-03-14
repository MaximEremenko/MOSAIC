from __future__ import annotations

from core.scattering.artifacts import persist_scattering_interval_chunk_result
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import IntervalTask


def save_amplitudes_and_meta(
    *,
    chunk_id: int,
    task: IntervalTask,
    grid_shape_nd,
    total_reciprocal_points: int,
    amplitudes_delta,
    amplitudes_average,
    point_data_processor,
    db_path: str,
    quiet_logs: bool = False,
) -> None:
    output_dir = point_data_processor.data_saver.output_dir
    work_unit = ScatteringWorkUnit.interval_chunk(
        interval_id=int(task.irecip_id),
        chunk_id=int(chunk_id),
        dimension=int(task.q_grid.shape[1]),
        output_dir=output_dir,
    )
    persist_scattering_interval_chunk_result(
        work_unit,
        grid_shape_nd=grid_shape_nd,
        total_reciprocal_points=total_reciprocal_points,
        contribution_reciprocal_points=(
            int(task.q_grid.shape[0]) * 2
            if task.q_grid.shape[1] > 2 and abs(task.q_grid[:, 2]).max() > 1e-7
            else int(task.q_grid.shape[0])
        ),
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
        output_dir=output_dir,
        db_path=db_path,
        quiet_logs=quiet_logs,
    )
