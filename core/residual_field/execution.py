from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from core.scattering.kernels import (
    point_list_to_recarray,
    reciprocal_space_points_counter,
    to_interval_dict,
)
from core.runtime import (
    DEFAULT_TASK_RETRIES,
    is_sync_client,
    logging_redirect_tqdm,
    progress_bar,
    yield_futures_with_results,
)
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.residual_field.contracts import ResidualFieldWorkUnit

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)


def _residual_field_parameter_digest(workflow_parameters) -> str:
    payload = {
        "mode": workflow_parameters.rspace_info.get("mode")
        or workflow_parameters.rspace_info.get("postprocess_mode")
        or workflow_parameters.rspace_info.get("postprocessing_mode")
        or "displacement",
        "q_window_kind": workflow_parameters.rspace_info.get("q_window_kind", "cheb"),
        "q_window_at_db": workflow_parameters.rspace_info.get("q_window_at_db", 100.0),
        "edge_guard_frac": workflow_parameters.rspace_info.get("edge_guard_frac", 0.10),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_residual_field_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    workflow_parameters,
    output_dir: str,
) -> list[ResidualFieldWorkUnit]:
    digest = _residual_field_parameter_digest(workflow_parameters)
    return [
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=int(interval_id),
            chunk_id=int(chunk_id),
            parameter_digest=digest,
            output_dir=output_dir,
            patch_scope="chunk",
            window_spec=str(
                workflow_parameters.rspace_info.get("q_window_kind", "cheb")
            ),
        )
        for interval_id, chunk_id in sorted(
            {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
        )
    ]


def _interval_path_for_work_unit(work_unit: ResidualFieldWorkUnit):
    for artifact in work_unit.source_artifacts:
        if artifact.kind == "interval-precompute" and artifact.path is not None:
            return artifact.path
    raise ValueError("ResidualFieldWorkUnit is missing a source interval artifact path.")


def run_residual_field_stage(
    *,
    workflow_parameters,
    structure,
    artifacts,
    client: "Client | None",
    max_inflight: int = 5_000,
) -> None:
    work_units = build_residual_field_work_units(
        artifacts.db_manager.get_unsaved_interval_chunks(),
        workflow_parameters=workflow_parameters,
        output_dir=artifacts.output_dir,
    )
    total_tasks = len(work_units)
    if total_tasks == 0:
        logger.info("Residual-field skipped – no unsaved (interval, chunk) pairs.")
        return

    total_reciprocal_points = sum(
        reciprocal_space_points_counter(to_interval_dict(interval), structure.supercell)
        for interval in artifacts.padded_intervals
    )
    chunk_ids = sorted({work_unit.chunk_id for work_unit in work_units})
    point_data_list: list[dict] = []
    for chunk_id in chunk_ids:
        point_data_list.extend(artifacts.db_manager.get_point_data_for_chunk(int(chunk_id)))

    if client is None:
        rec = point_list_to_recarray(point_data_list)
        with progress_bar(total_tasks, desc="Residual field (chunks × intervals)", unit="pairs") as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                manifest = run_residual_field_interval_chunk_task(
                    work_unit,
                    _interval_path_for_work_unit(work_unit),
                    atoms,
                    total_reciprocal_points=total_reciprocal_points,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path,
                    quiet_logs=False,
                )
                pbar.update(1)
                pbar.refresh()
                if manifest is None:
                    logger.error(
                        "GAVE UP after retries | residual iv %s | chunk %d (sync)",
                        work_unit.interval_id,
                        work_unit.chunk_id,
                    )
        logger.info("Residual-field finished (sync).")
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
        logger.warning("Circuit-breaker: switching residual-field to CPU-only & throttling.")

    rec = point_list_to_recarray(point_data_list)
    chunk_futures = {
        chunk_id: client.scatter(rec[rec.chunk_id == chunk_id], broadcast=False, hash=False)
        for chunk_id in chunk_ids
    }
    interval_path_futures = {
        int(work_unit.interval_id): client.scatter(
            _interval_path_for_work_unit(work_unit),
            broadcast=False,
        )
        for work_unit in work_units
        if work_unit.interval_id is not None
    }
    retries_left = {
        (int(work_unit.interval_id), int(work_unit.chunk_id)): DEFAULT_TASK_RETRIES
        for work_unit in work_units
        if work_unit.interval_id is not None
    }
    flying: set = set()
    future_meta: dict = {}
    submitted = 0

    def _submit(work_unit: ResidualFieldWorkUnit) -> None:
        nonlocal submitted
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            interval_path_futures[int(work_unit.interval_id)],
            chunk_futures[int(work_unit.chunk_id)],
            total_reciprocal_points=total_reciprocal_points,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path,
            quiet_logs=True,
            key=f"residual-{work_unit.interval_id}-{work_unit.chunk_id}",
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
                key = (int(work_unit.interval_id), int(work_unit.chunk_id))
                if retries_left.get(key, 0) > 0:
                    retries_left[key] -= 1
                    _submit(work_unit)
            else:
                fail_streak = 0

    with logging_redirect_tqdm():
        with progress_bar(total_tasks, desc="Residual field (chunks × intervals)", unit="pairs") as pbar:

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
                                int(completed_work_unit.interval_id),
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
                        "GAVE UP after retries | residual iv %s | chunk %d",
                        completed_work_unit.interval_id,
                        completed_work_unit.chunk_id,
                    )

    logger.info("Residual-field finished – %d tasks submitted", submitted)


__all__ = ["build_residual_field_work_units", "run_residual_field_stage"]
