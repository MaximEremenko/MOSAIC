from __future__ import annotations

import logging
import os
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
    resolve_worker_scratch_root,
    yield_futures_with_results,
)
from core.residual_field.artifacts import (
    delete_reclaimable_residual_field_shards,
    reduce_residual_field_shards_for_chunk,
)
from core.residual_field.planning import build_residual_field_work_units
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.residual_field.contracts import ResidualFieldShardManifest, ResidualFieldWorkUnit

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_INTERVALS_PER_SHARD = 4


def _interval_paths_for_work_unit(work_unit: ResidualFieldWorkUnit) -> tuple[str, ...]:
    interval_paths = tuple(
        artifact.path
        for artifact in work_unit.source_artifacts
        if artifact.kind == "interval-precompute" and artifact.path is not None
    )
    if not interval_paths:
        raise ValueError("ResidualFieldWorkUnit is missing source interval artifact paths.")
    return interval_paths


def _cleanup_residual_shards_enabled(workflow_parameters) -> bool:
    return _residual_shard_cleanup_policy(workflow_parameters) == "delete_reclaimable"


def _residual_shard_cleanup_policy(workflow_parameters) -> str:
    runtime_policy = workflow_parameters.runtime_info.get(
        "residual_shard_cleanup_policy"
    )
    if runtime_policy is not None:
        value = str(runtime_policy).strip().lower()
        if value in {"off", "keep"}:
            return "off"
        if value in {"delete_reclaimable", "cleanup"}:
            return "delete_reclaimable"
    runtime_value = workflow_parameters.runtime_info.get("cleanup_residual_shards")
    if runtime_value is not None:
        return "delete_reclaimable" if bool(runtime_value) else "off"
    return (
        "delete_reclaimable"
        if os.getenv("MOSAIC_CLEANUP_RESIDUAL_SHARDS", "0") == "1"
        else "off"
    )


def _finalize_residual_field_chunks(
    *,
    chunk_ids: list[int],
    parameter_digest: str,
    manifests_by_chunk: dict[int, list[ResidualFieldShardManifest]],
    output_dir: str,
    db_path: str,
    cleanup_policy: str,
) -> None:
    for chunk_id in sorted(set(int(chunk_id) for chunk_id in chunk_ids)):
        manifest = reduce_residual_field_shards_for_chunk(
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=manifests_by_chunk.get(int(chunk_id)),
            cleanup_policy=cleanup_policy,
            quiet_logs=False,
        )
        if cleanup_policy == "delete_reclaimable" and manifest is not None:
            delete_reclaimable_residual_field_shards(
                output_dir=output_dir,
                chunk_id=int(chunk_id),
                parameter_digest=parameter_digest,
                db_path=db_path,
            )


def run_residual_field_stage(
    *,
    workflow_parameters,
    structure,
    artifacts,
    client: "Client | None",
    max_inflight: int = 5_000,
) -> None:
    max_intervals_per_shard = int(
        workflow_parameters.runtime_info.get(
            "residual_shard_batch_size",
            DEFAULT_RESIDUAL_INTERVALS_PER_SHARD,
        )
    )
    scratch_root = resolve_worker_scratch_root(
        preferred=workflow_parameters.runtime_info.get(
            "residual_shard_scratch_root",
            os.getenv("MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT"),
        ),
        stage="residual_field",
    )
    cleanup_policy = _residual_shard_cleanup_policy(workflow_parameters)
    work_units = build_residual_field_work_units(
        artifacts.db_manager.get_unsaved_interval_chunks(),
        parameters=workflow_parameters,
        output_dir=artifacts.output_dir,
        max_intervals_per_shard=max_intervals_per_shard,
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

    manifests_by_chunk: dict[int, list[ResidualFieldShardManifest]] = {}

    if client is None or is_sync_client(client):
        rec = point_list_to_recarray(point_data_list)
        with progress_bar(total_tasks, desc="Residual field (chunks × shard-batches)", unit="batches") as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                manifest = run_residual_field_interval_chunk_task(
                    work_unit,
                    _interval_paths_for_work_unit(work_unit),
                    atoms,
                    total_reciprocal_points=total_reciprocal_points,
                    output_dir=artifacts.output_dir,
                    scratch_root=scratch_root,
                    quiet_logs=False,
                )
                pbar.update(1)
                pbar.refresh()
                if manifest is None:
                    logger.error(
                        "GAVE UP after retries | residual batch %s | chunk %d (sync)",
                        ",".join(str(interval_id) for interval_id in work_unit.interval_ids),
                        work_unit.chunk_id,
                    )
                else:
                    manifests_by_chunk.setdefault(int(work_unit.chunk_id), []).append(manifest)
        _finalize_residual_field_chunks(
            chunk_ids=chunk_ids,
            parameter_digest=work_units[0].parameter_digest,
            manifests_by_chunk=manifests_by_chunk,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path,
            cleanup_policy=cleanup_policy,
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
    retries_left = {
        (str(work_unit.artifact_key), int(work_unit.chunk_id)): DEFAULT_TASK_RETRIES
        for work_unit in work_units
    }
    flying: set = set()
    future_meta: dict = {}
    submitted = 0

    def _submit(work_unit: ResidualFieldWorkUnit) -> None:
        nonlocal submitted
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            _interval_paths_for_work_unit(work_unit),
            chunk_futures[int(work_unit.chunk_id)],
            total_reciprocal_points=total_reciprocal_points,
            output_dir=artifacts.output_dir,
            scratch_root=scratch_root,
            quiet_logs=True,
            key=f"residual-{work_unit.artifact_key}",
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
                key = (str(work_unit.artifact_key), int(work_unit.chunk_id))
                if retries_left.get(key, 0) > 0:
                    retries_left[key] -= 1
                    _submit(work_unit)
            else:
                fail_streak = 0

    with logging_redirect_tqdm():
        with progress_bar(total_tasks, desc="Residual field (chunks × shard-batches)", unit="batches") as pbar:

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
                                str(completed_work_unit.artifact_key),
                                int(completed_work_unit.chunk_id),
                            )
                            if retries_left.get(key, 0) > 0:
                                retries_left[key] -= 1
                                _submit(completed_work_unit)
                        else:
                            fail_streak = 0
                            if result and completed_work_unit is not None:
                                try:
                                    manifests_by_chunk.setdefault(
                                        int(completed_work_unit.chunk_id),
                                        [],
                                    ).append(future.result())
                                except Exception:
                                    pass

            for future, result in yield_futures_with_results(list(flying), client):
                completed_work_unit = future_meta.pop(future, None)
                bump()
                if not bool(result) and completed_work_unit is not None:
                    logger.error(
                        "GAVE UP after retries | residual batch %s | chunk %d",
                        ",".join(str(interval_id) for interval_id in completed_work_unit.interval_ids),
                        completed_work_unit.chunk_id,
                    )
                elif completed_work_unit is not None:
                    try:
                        manifests_by_chunk.setdefault(
                            int(completed_work_unit.chunk_id),
                            [],
                        ).append(future.result())
                    except Exception:
                        pass

    _finalize_residual_field_chunks(
        chunk_ids=chunk_ids,
        parameter_digest=work_units[0].parameter_digest,
        manifests_by_chunk=manifests_by_chunk,
        output_dir=artifacts.output_dir,
        db_path=artifacts.db_manager.db_path,
        cleanup_policy=cleanup_policy,
    )
    logger.info("Residual-field finished – %d tasks submitted", submitted)


__all__ = ["build_residual_field_work_units", "run_residual_field_stage"]
