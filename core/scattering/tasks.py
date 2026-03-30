from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.scattering.artifacts import (
    build_scattering_interval_manifest,
    is_interval_artifact_committed,
    mark_empty_interval_precomputed,
    persist_precomputed_interval_artifact,
    persist_scattering_interval_chunk_result,
)
from core.scattering.contracts import ScatteringArtifactManifest, ScatteringWorkUnit
from core.scattering.kernels import (
    IntervalTask,
    aggregate_interval_contributions,
    build_rifft_grid_for_chunk,
    compute_interval_coeff_contribution,
    compute_interval_element_contribution,
    generate_q_space_grid_sync,
)
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_batch_materialize_once,
)
from core.contracts import CompletionStatus
from core.runtime import handle_worker_gpu_failure


logger = logging.getLogger(__name__)


def _lower_worker_log_levels() -> None:
    for name in (
        __name__,
        "core.storage.database_manager",
        "DatabaseManager",
        "core.patch_centers.point_data",
        "PointDataProcessor",
        "RIFFTInDataSaver",
    ):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


def load_interval_task_payload(interval_path: Path) -> IntervalTask:
    with np.load(interval_path, mmap_mode="r") as data:
        return IntervalTask(
            int(data["irecip_id"].item()),
            str(data["element"].item()),
            data["q_grid"],
            data["q_amp"],
            data["q_amp_av"],
        )


def scattering_contribution_point_count(interval_task: IntervalTask) -> int:
    q_grid = interval_task.q_grid
    if q_grid.shape[1] > 2 and np.max(np.abs(q_grid[:, 2])) > 1e-7:
        return int(q_grid.shape[0]) * 2
    return int(q_grid.shape[0])


def compute_scattering_interval_payload(
    interval: dict,
    *,
    B_: np.ndarray,
    mask_params: dict,
    MaskStrategy,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    use_coeff: bool,
    coeff_val: np.ndarray | None,
    unique_elements: list[str],
    ff_factory,
) -> IntervalTask | None:
    q_grid = generate_q_space_grid_sync(interval, B_, mask_params, MaskStrategy, supercell)
    if q_grid.size == 0:
        return None

    contributions: list[tuple] = []
    if use_coeff:
        contributions.append(
            compute_interval_coeff_contribution(
                interval,
                q_grid,
                coeff_val,
                original_coords,
                cells_origin,
            )
        )
    else:
        for element in unique_elements:
            contribution = compute_interval_element_contribution(
                interval,
                q_grid,
                element,
                original_coords,
                cells_origin,
                elements_arr,
                charge,
                ff_factory,
            )
            if contribution is not None:
                contributions.append(contribution)

    if not contributions:
        return None

    return aggregate_interval_contributions(contributions, use_coeff=use_coeff)


def run_scattering_interval_task(
    work_unit: ScatteringWorkUnit,
    interval: dict,
    *,
    B_: np.ndarray,
    mask_params: dict,
    MaskStrategy,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    use_coeff: bool,
    coeff_val: np.ndarray | None,
    unique_elements: list[str],
    ff_factory,
    output_dir: str,
    db_path: str,
) -> ScatteringArtifactManifest | None:
    if is_interval_artifact_committed(work_unit, db_path=db_path):
        return build_scattering_interval_manifest(
            work_unit,
            completion_status=CompletionStatus.COMMITTED,
        )

    interval_task = compute_scattering_interval_payload(
        interval,
        B_=B_,
        mask_params=mask_params,
        MaskStrategy=MaskStrategy,
        supercell=supercell,
        original_coords=original_coords,
        cells_origin=cells_origin,
        elements_arr=elements_arr,
        charge=charge,
        use_coeff=use_coeff,
        coeff_val=coeff_val,
        unique_elements=unique_elements,
        ff_factory=ff_factory,
    )
    if interval_task is None:
        # Mask eliminated all Q-points in this interval.  Mark it as
        # precomputed so that downstream stages (Stage-2 and
        # residual-field) do not attempt to load a non-existent .npz.
        mark_empty_interval_precomputed(
            work_unit.interval_id, db_path=db_path
        )
        return None
    return persist_precomputed_interval_artifact(work_unit, interval_task, db_path=db_path)


def run_scattering_interval_chunk_task(
    work_unit: ScatteringWorkUnit,
    interval_path: Path,
    atoms: np.recarray,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    db_path: str,
    quiet_logs: bool = False,
) -> ScatteringArtifactManifest | None:
    if quiet_logs:
        _lower_worker_log_levels()

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
        inverse_pair = execute_inverse_cunufft_batch_materialize_once(
            q_coords=interval_task.q_grid,
            weights=np.stack(
                [
                    interval_task.q_amp - interval_task.q_amp_av,
                    interval_task.q_amp_av,
                ],
                axis=0,
            ),
            real_coords=rifft_grid,
            eps=1e-12,
        )
        amplitudes_delta = inverse_pair[0]
        amplitudes_average = inverse_pair[1]
        return persist_scattering_interval_chunk_result(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=scattering_contribution_point_count(interval_task),
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            output_dir=output_dir,
            db_path=db_path,
            quiet_logs=quiet_logs,
        )
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            int(work_unit.chunk_id) if work_unit.chunk_id is not None else -1,
            interval_id,
            err,
            exc_info=True,
        )
        handle_worker_gpu_failure(err, logger=logger)
        return None


__all__ = [
    "compute_scattering_interval_payload",
    "load_interval_task_payload",
    "run_scattering_interval_chunk_task",
    "run_scattering_interval_task",
    "scattering_contribution_point_count",
]
