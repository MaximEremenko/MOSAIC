from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

from core.scattering.artifacts import is_interval_artifact_committed
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import (
    point_list_to_recarray,
    reciprocal_space_points_counter,
    to_interval_dict,
)
from core.scattering.planning import (
    build_scattering_interval_chunk_work_units,
    build_scattering_interval_lookup,
    build_scattering_precompute_work_units,
    chunk_ids_for_work_units,
    interval_paths_for_work_units,
)
from core.runtime import (
    DEFAULT_TASK_RETRIES,
    is_sync_client,
    logging_redirect_tqdm,
    progress_bar,
    quiet_loggers,
    register_cleanup_plugin,
    yield_futures_with_results,
)
from core.scattering.tasks import (
    run_scattering_interval_chunk_task,
    run_scattering_interval_task,
)
from core.storage.database_manager import DatabaseManager

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)


def _chunk_task_key(work_unit: ScatteringWorkUnit) -> str:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk task key requires a chunk-scoped work unit.")
    return f"proc-{work_unit.interval_id}-{work_unit.chunk_id}"


def run_interval_precompute(
    work_units: list[ScatteringWorkUnit],
    *,
    interval_lookup: dict[int, dict],
    B_: np.ndarray,
    parameters: Dict[str, Any],
    unique_elements: Iterable[str],
    mask_params: Dict[str, Any],
    MaskStrategy,
    supercell: np.ndarray,
    output_dir: str,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
    db: DatabaseManager,
    client: "Client | None",
) -> list[Path]:
    pending = [
        work_unit
        for work_unit in work_units
        if not is_interval_artifact_committed(work_unit, db_path=db.db_path)
    ]
    cached = [
        Path(work_unit.interval_artifact.path)
        for work_unit in work_units
        if work_unit.interval_artifact is not None
        and work_unit.interval_artifact.path is not None
        and is_interval_artifact_committed(work_unit, db_path=db.db_path)
    ]

    if not pending:
        logger.info(
            "Stage-1 complete: %d written, %d cached, %d skipped",
            0,
            len(cached),
            0,
        )
        return cached

    written_files: list[Path] = []
    if (client is not None) and (not is_sync_client(client)):
        futures = [
            client.submit(
                run_scattering_interval_task,
                work_unit,
                interval_lookup[work_unit.interval_id],
                B_=B_,
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=supercell,
                original_coords=original_coords,
                cells_origin=cells_origin,
                elements_arr=elements_arr,
                charge=charge,
                use_coeff=("coeff" in parameters),
                coeff_val=parameters.get("coeff"),
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                output_dir=output_dir,
                db_path=db.db_path,
                pure=False,
                resources={"nufft": 1},
            )
            for work_unit in pending
        ]
        with logging_redirect_tqdm():
            with progress_bar(len(futures), desc="Precompute intervals", unit="intervals") as pbar:
                for future, _ in yield_futures_with_results(futures, client):
                    try:
                        manifest = future.result()
                    except Exception:
                        manifest = None
                    if manifest is not None and manifest.artifacts:
                        artifact_path = manifest.artifacts[0].path
                        if artifact_path is not None:
                            written_files.append(Path(artifact_path))
                    pbar.update(1)
                    pbar.refresh()
        logger.info(
            "Stage-1 complete: %d written, %d cached, %d skipped",
            len(written_files),
            len(cached),
            len(pending) - len(written_files),
        )
        return cached + written_files

    with progress_bar(len(pending), desc="Precompute intervals", unit="intervals") as pbar:
        for work_unit in pending:
            manifest = run_scattering_interval_task(
                work_unit,
                interval_lookup[work_unit.interval_id],
                B_=B_,
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=supercell,
                original_coords=original_coords,
                cells_origin=cells_origin,
                elements_arr=elements_arr,
                charge=charge,
                use_coeff=("coeff" in parameters),
                coeff_val=parameters.get("coeff"),
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                output_dir=output_dir,
                db_path=db.db_path,
            )
            if manifest is not None and manifest.artifacts:
                artifact_path = manifest.artifacts[0].path
                if artifact_path is not None:
                    written_files.append(Path(artifact_path))
            pbar.update(1)
            pbar.refresh()

    logger.info(
        "Stage-1 complete: %d written, %d cached, %d skipped",
        len(written_files),
        len(cached),
        len(pending) - len(written_files),
    )
    return cached + written_files


def run_interval_chunk_execution(
    work_units: list[ScatteringWorkUnit],
    *,
    total_reciprocal_points: int,
    point_data_list: list[dict],
    db_manager: DatabaseManager,
    client: "Client | None",
    output_dir: str,
    max_inflight: int = 5_000,
) -> None:
    total_tasks = len(work_units)
    if total_tasks == 0:
        logger.info("Stage-2 skipped – no unsaved (interval, chunk) pairs.")
        return

    interval_paths = interval_paths_for_work_units(work_units)
    if client is None:
        rec = point_list_to_recarray(point_data_list)
        with progress_bar(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                manifest = run_scattering_interval_chunk_task(
                    work_unit,
                    interval_paths[work_unit.interval_id],
                    atoms,
                    total_reciprocal_points=total_reciprocal_points,
                    output_dir=output_dir,
                    db_path=db_manager.db_path,
                    quiet_logs=False,
                )
                pbar.update(1)
                pbar.refresh()
                if manifest is None:
                    logger.error(
                        "GAVE UP after retries | iv %d | chunk %d (sync)",
                        work_unit.interval_id,
                        work_unit.chunk_id,
                    )
        logger.info("Stage-2 finished (sync).")
        return

    fail_streak, fail_threshold = 0, 3
    gpu_tripped = False

    def _trip_to_cpu_only() -> None:
        nonlocal gpu_tripped, max_inflight
        if gpu_tripped:
            return
        if hasattr(client, "run"):
            try:
                from core.adapters.cunufft_wrapper import set_cpu_only

                client.run(set_cpu_only, True)
            except Exception:
                pass
        max_inflight = min(max_inflight, 256)
        gpu_tripped = True
        logger.warning("Circuit-breaker: switching Stage-2 to CPU-only & throttling.")

    rec = point_list_to_recarray(point_data_list)
    chunk_futures = {
        chunk_id: client.scatter(rec[rec.chunk_id == chunk_id], broadcast=False, hash=False)
        for chunk_id in chunk_ids_for_work_units(work_units)
    }
    interval_path_futures = {
        interval_id: client.scatter(path, broadcast=False)
        for interval_id, path in interval_paths.items()
    }

    retries_left = {
        (work_unit.interval_id, int(work_unit.chunk_id)): DEFAULT_TASK_RETRIES
        for work_unit in work_units
        if work_unit.chunk_id is not None
    }
    flying: set = set()
    future_meta: dict = {}
    submitted = 0

    def _submit(work_unit: ScatteringWorkUnit) -> None:
        nonlocal submitted
        future = client.submit(
            run_scattering_interval_chunk_task,
            work_unit,
            interval_path_futures[work_unit.interval_id],
            chunk_futures[int(work_unit.chunk_id)],
            total_reciprocal_points=total_reciprocal_points,
            output_dir=output_dir,
            db_path=db_manager.db_path,
            quiet_logs=True,
            key=_chunk_task_key(work_unit),
            pure=False,
            resources={"nufft": 1},
            retries=DEFAULT_TASK_RETRIES,
        )
        flying.add(future)
        future_meta[future] = work_unit
        submitted += 1

    def _harvest_finished_nonblocking(bump) -> None:
        nonlocal fail_streak
        done_now = [future for future in list(flying) if future.done()]
        for future in done_now:
            try:
                ok = future.result() is not None
            except Exception:
                ok = False

            flying.discard(future)
            work_unit = future_meta.pop(future, None)
            bump()

            if not ok and work_unit is not None:
                fail_streak += 1
                if fail_streak >= fail_threshold:
                    _trip_to_cpu_only()
                key = (work_unit.interval_id, int(work_unit.chunk_id))
                if retries_left.get(key, 0) > 0:
                    retries_left[key] -= 1
                    _submit(work_unit)
            else:
                fail_streak = 0

    with logging_redirect_tqdm():
        with progress_bar(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:

            def bump() -> None:
                pbar.update(1)
                pbar.refresh()

            for work_unit in work_units:
                _submit(work_unit)
                _harvest_finished_nonblocking(bump)
                while len(flying) >= max_inflight:
                    for future, result in yield_futures_with_results(list(flying), client):
                        ok = bool(result)
                        flying.discard(future)
                        completed_work_unit = future_meta.pop(future, None)
                        bump()
                        if not ok and completed_work_unit is not None:
                            fail_streak += 1
                            if fail_streak >= fail_threshold:
                                _trip_to_cpu_only()
                            key = (
                                completed_work_unit.interval_id,
                                int(completed_work_unit.chunk_id),
                            )
                            if retries_left.get(key, 0) > 0:
                                retries_left[key] -= 1
                                _submit(completed_work_unit)
                        else:
                            fail_streak = 0

            for future, result in yield_futures_with_results(list(flying), client):
                completed_work_unit = future_meta.pop(future, None)
                bump()
                if not bool(result) and completed_work_unit is not None:
                    logger.error(
                        "GAVE UP after retries | iv %d | chunk %d",
                        completed_work_unit.interval_id,
                        completed_work_unit.chunk_id,
                    )

    logger.info("Stage-2 finished – %d tasks submitted", submitted)


def run_scattering_stage(
    parameters: Dict[str, Any],
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters: Dict[str, Any],
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor,
    client: "Client | None",
) -> None:
    register_cleanup_plugin(client, is_sync_client=is_sync_client)

    reciprocal_space_intervals_all = parameters["reciprocal_space_intervals_all"]
    reciprocal_space_intervals = parameters["reciprocal_space_intervals"]
    original_coords = parameters["original_coords"]
    cells_origin = parameters["cells_origin"]
    elements_arr = parameters["elements"]
    vectors = parameters["vectors"]
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0.0)
    dimension = int(len(supercell))

    B_ = np.linalg.inv(vectors / supercell)
    unique_elements = np.unique(elements_arr)

    precompute_work_units = build_scattering_precompute_work_units(
        reciprocal_space_intervals,
        dimension=dimension,
        output_dir=output_dir,
    )
    interval_lookup = build_scattering_interval_lookup(reciprocal_space_intervals)
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        run_interval_precompute(
            precompute_work_units,
            interval_lookup=interval_lookup,
            B_=B_,
            parameters=parameters,
            unique_elements=unique_elements,
            mask_params=MaskStrategyParameters,
            MaskStrategy=MaskStrategy,
            supercell=supercell,
            output_dir=output_dir,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            charge=charge,
            ff_factory=FormFactorFactoryProducer,
            db=db_manager,
            client=client,
        )
    logger.info("Completed scattering interval precompute stage")


__all__ = [
    "run_interval_chunk_execution",
    "run_interval_precompute",
    "run_scattering_stage",
]
