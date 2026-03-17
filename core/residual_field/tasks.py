from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from core.residual_field.backend import (
    ResidualFieldReducerBackend,
    build_residual_field_reducer_backend,
    get_process_local_residual_field_backend,
)
from core.scattering.kernels import build_rifft_grid_for_chunk
from core.scattering.kernels import IntervalTask
from core.scattering.tasks import load_interval_task_payload, scattering_contribution_point_count
from core.residual_field.contracts import (
    ResidualFieldAccumulatorStatus,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_super_batch,
)
from core.runtime import handle_worker_gpu_failure, task_progress_enabled


logger = logging.getLogger(__name__)

_worker_logging_configured = False


def _ensure_worker_logging() -> None:
    """Ensure root logger has a handler in Dask worker processes."""
    global _worker_logging_configured
    if _worker_logging_configured:
        return
    _worker_logging_configured = True
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"
        ))
        root.addHandler(handler)
        root.setLevel(logging.INFO)


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
    db_path: str | None = None,
    scratch_root: str | None = None,
    reducer_backend: ResidualFieldReducerBackend | None = None,
    total_expected_partials: int | None = None,
    owner_local_reducer: bool = False,
    quiet_logs: bool = False,
) -> ResidualFieldShardManifest | ResidualFieldAccumulatorStatus | None:
    _ensure_worker_logging()
    interval_ids = work_unit.interval_ids or ((work_unit.interval_id,) if work_unit.interval_id is not None else ())
    try:
        show_progress = _task_progress_enabled(quiet_logs)
        resolved_backend = reducer_backend or build_residual_field_reducer_backend(
            "local_restartable"
        )
        loaded_interval_inputs = _normalize_interval_inputs(interval_paths)
        if not loaded_interval_inputs:
            raise ValueError("Residual-field batch task requires at least one interval artifact path.")
        partition_atoms = atoms
        if work_unit.point_start is not None or work_unit.point_stop is not None:
            start = int(work_unit.point_start or 0)
            stop = int(work_unit.point_stop or len(atoms))
            partition_atoms = atoms[start:stop]
        chunk_data = [
            {
                "coordinates": partition_atoms["coordinates"][index],
                "dist_from_atom_center": partition_atoms["dist_from_atom_center"][index],
                "step_in_frac": partition_atoms["step_in_frac"][index],
            }
            for index in range(partition_atoms.shape[0])
        ]
        if owner_local_reducer:
            if scratch_root is None or db_path is None or total_expected_partials is None:
                raise ValueError(
                    "Owner-local reducer tasks require scratch_root, db_path, and total_expected_partials."
                )
            if not callable(getattr(resolved_backend, "accept_local_contribution", None)):
                raise ValueError(
                    "Residual-field owner-local reduction requires accept_local_contribution support."
                )
            worker_backend = get_process_local_residual_field_backend(
                resolved_backend
            )
            if not callable(getattr(worker_backend, "accept_local_contribution", None)):
                raise ValueError(
                    "Residual-field owner-local reduction requires process-local accept_local_contribution support."
                )
            already_durable = getattr(
                worker_backend,
                "local_intervals_already_durable",
                lambda *_args, **_kwargs: False,
            )
            if already_durable(
                work_unit,
                output_dir=output_dir,
            ):
                return ResidualFieldAccumulatorStatus(
                    artifact_key=work_unit.artifact_key,
                    chunk_id=work_unit.chunk_id,
                    parameter_digest=work_unit.parameter_digest,
                    interval_ids=work_unit.interval_ids,
                    partition_id=work_unit.partition_id,
                    contribution_reciprocal_point_count=0,
                    total_reciprocal_points=total_reciprocal_points,
                )
        rifft_grid, grid_shape_nd = build_rifft_grid_for_chunk(chunk_data)
        if show_progress:
            logger.debug(
                "Residual batch start | chunk=%d | partition=%s | intervals=%s | rifft_points=%d",
                int(work_unit.chunk_id),
                work_unit.partition_id,
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
                logger.debug(
                    "Residual batch group | chunk=%d | partition=%s | group=%d/%d | intervals=%d | q_points=%d",
                    int(work_unit.chunk_id),
                    work_unit.partition_id,
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
            stacked_arr = np.stack(stacked_weights, axis=0)
            del stacked_weights
            inverse_outputs = execute_inverse_cunufft_super_batch(
                q_coords=reference_q_grid,
                weights=stacked_arr,
                real_coords=rifft_grid,
                eps=1e-12,
            )
            del stacked_arr
            grouped_delta = np.sum(inverse_outputs[0::2], axis=0, dtype=np.complex128)
            grouped_average = np.sum(inverse_outputs[1::2], axis=0, dtype=np.complex128)
            del inverse_outputs
            if amplitudes_delta is None:
                amplitudes_delta = grouped_delta
            else:
                amplitudes_delta += grouped_delta
                del grouped_delta
            if amplitudes_average is None:
                amplitudes_average = grouped_average
            else:
                amplitudes_average += grouped_average
                del grouped_average
        point_ids = np.arange(amplitudes_delta.shape[0], dtype=np.int64)
        if owner_local_reducer:
            worker_backend.accept_local_contribution(
                work_unit,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                contribution_reciprocal_points=contribution_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
                point_ids=point_ids,
                output_dir=output_dir,
                scratch_root=scratch_root,
                db_path=db_path,
                total_expected_partials=total_expected_partials,
                cleanup_policy="off",
            )
            return ResidualFieldAccumulatorStatus(
                artifact_key=work_unit.artifact_key,
                chunk_id=work_unit.chunk_id,
                parameter_digest=work_unit.parameter_digest,
                interval_ids=work_unit.interval_ids,
                partition_id=work_unit.partition_id,
                contribution_reciprocal_point_count=contribution_reciprocal_points,
                total_reciprocal_points=total_reciprocal_points,
            )
        if show_progress:
            logger.debug(
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
