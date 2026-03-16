from __future__ import annotations

import logging
import math
import os
from pathlib import Path
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
    task_progress_enabled,
    yield_futures_with_results,
)
from core.residual_field.backend import (
    ResidualFieldReducerBackend,
    ResidualFieldLocalAccumulatorPartial,
    build_residual_field_reducer_backend,
    is_same_node_local_client,
    resolve_residual_field_reducer_backend,
)
from core.residual_field.planning import build_residual_field_work_units
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.residual_field.contracts import ResidualFieldShardManifest, ResidualFieldWorkUnit

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_INTERVALS_PER_SHARD = 4


def _format_progress_bar(count: int, total: int, *, width: int = 20) -> str:
    total = max(int(total), 1)
    count = max(0, min(int(count), total))
    filled = int(round((count / float(total)) * width))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _work_unit_interval_label(work_unit: ResidualFieldWorkUnit) -> str:
    interval_ids = work_unit.interval_ids or (
        (work_unit.interval_id,) if work_unit.interval_id is not None else ()
    )
    return ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a"


def _log_async_residual_progress(
    *,
    enabled: bool,
    event: str,
    work_unit: ResidualFieldWorkUnit,
    completed: int,
    total: int,
    submitted: int,
    running: int,
    detail: str | None = None,
) -> None:
    if not enabled:
        return
    current = int(submitted if event == "queue" else completed)
    progress_bar_text = _format_progress_bar(current, total)
    percent = (100.0 * current / float(max(int(total), 1)))
    suffix = f" | {detail}" if detail else ""
    logger.info(
        "Residual-field %s %s %d/%d (%.0f%%) | running=%d | chunk=%d | intervals=%s%s",
        event,
        progress_bar_text,
        current,
        int(total),
        percent,
        int(running),
        int(work_unit.chunk_id),
        _work_unit_interval_label(work_unit),
        suffix,
    )


def _should_log_async_progress(
    *,
    phase: str,
    count: int,
    total: int,
    force: bool = False,
) -> bool:
    if force or total <= 0:
        return True
    if count <= 1 or count >= total:
        return True
    target_updates = 4 if phase == "queue" else 8
    stride = max(1, int(math.ceil(total / float(target_updates))))
    return count % stride == 0


def _cap_async_max_inflight(
    *,
    client,
    requested: int,
) -> int:
    requested = max(1, int(requested))
    if client is None or is_sync_client(client):
        return requested
    if not is_same_node_local_client(client):
        return requested
    try:
        scheduler_info = client.scheduler_info()
        workers = scheduler_info.get("workers", {})
    except Exception:
        workers = {}
    if not workers:
        return min(requested, 1)
    nufft_slots = sum(
        int(worker.get("resources", {}).get("nufft", 0))
        for worker in workers.values()
    )
    capacity = max(1, int(nufft_slots) if int(nufft_slots) > 0 else len(workers))
    capped = min(requested, capacity)
    if capped < requested:
        logger.info(
            "Residual-field local inflight cap | requested=%d | effective=%d | workers=%d | nufft_slots=%d",
            requested,
            capped,
            len(workers),
            int(nufft_slots),
        )
    return capped


def _build_task_reducer_backend(
    reducer_backend: ResidualFieldReducerBackend,
) -> ResidualFieldReducerBackend:
    return build_residual_field_reducer_backend(
        reducer_backend.layout.kind,
        shard_storage_root_override=getattr(
            reducer_backend,
            "shard_storage_root_override",
            None,
        ),
        local_accumulator_max_ram_bytes=int(
            getattr(
                reducer_backend,
                "local_accumulator_max_ram_bytes",
                256 * 1024 * 1024,
            )
        ),
    )


def _interval_paths_for_work_unit(work_unit: ResidualFieldWorkUnit) -> tuple[str, ...]:
    interval_paths = tuple(
        artifact.path
        for artifact in work_unit.source_artifacts
        if artifact.kind == "interval-precompute" and artifact.path is not None
    )
    if not interval_paths:
        raise ValueError("ResidualFieldWorkUnit is missing source interval artifact paths.")
    return interval_paths


def _interval_inputs_for_work_unit(
    work_unit: ResidualFieldWorkUnit,
    *,
    transient_interval_payloads: dict[int, object] | None,
):
    interval_ids = tuple(int(interval_id) for interval_id in (work_unit.interval_ids or ()))
    if work_unit.interval_id is not None and not interval_ids:
        interval_ids = (int(work_unit.interval_id),)
    if transient_interval_payloads:
        payload_source = transient_interval_payloads
        if all(int(interval_id) in payload_source for interval_id in interval_ids):
            values = tuple(payload_source[int(interval_id)] for interval_id in interval_ids)
            return values[0] if len(values) == 1 else values
    return _interval_paths_for_work_unit(work_unit)


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
    reducer_backend: ResidualFieldReducerBackend,
    scratch_root: str | None,
) -> None:
    for chunk_id in sorted(set(int(chunk_id) for chunk_id in chunk_ids)):
        manifest = reducer_backend.finalize_chunk(
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=manifests_by_chunk.get(int(chunk_id)),
            cleanup_policy=cleanup_policy,
            scratch_root=scratch_root,
            quiet_logs=False,
        )
        if cleanup_policy == "delete_reclaimable" and manifest is not None:
            reducer_backend.cleanup_reclaimable_shards(
                output_dir=output_dir,
                chunk_id=int(chunk_id),
                parameter_digest=parameter_digest,
                db_path=db_path,
                manifests=manifests_by_chunk.get(int(chunk_id)),
                scratch_root=scratch_root,
            )


def run_residual_field_stage(
    *,
    workflow_parameters,
    structure,
    artifacts,
    client: "Client | None",
    max_inflight: int = 5_000,
) -> None:
    explicit_scratch_root = workflow_parameters.runtime_info.get(
        "residual_shard_scratch_root",
        os.getenv("MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT"),
    )
    preliminary_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=workflow_parameters,
        client=client,
    )
    max_intervals_per_shard = int(
        workflow_parameters.runtime_info.get(
            "residual_shard_batch_size",
            DEFAULT_RESIDUAL_INTERVALS_PER_SHARD,
        )
    )
    scratch_root = resolve_worker_scratch_root(
        preferred=(
            explicit_scratch_root
            if explicit_scratch_root is not None
            else str(Path(artifacts.output_dir) / ".local_restartable")
            if preliminary_backend.layout.kind == "local_restartable"
            else None
        ),
        stage="residual_field",
    )
    reducer_backend = preliminary_backend
    task_reducer_backend = _build_task_reducer_backend(reducer_backend)
    reducer_runtime_state = reducer_backend.describe_runtime_state(
        output_dir=artifacts.output_dir,
        scratch_root=scratch_root,
    )
    logger.info(
        "Residual-field reducer backend %s | scratch=%s | durable=%s",
        reducer_runtime_state.kind,
        reducer_runtime_state.local_scratch_root or "<none>",
        reducer_runtime_state.durable_root,
    )
    logger.debug(
        "Residual-field reducer state | ram=%s | scratch=%s | durable=%s | transport=%s | restart=%s",
        ", ".join(reducer_runtime_state.ram_state),
        ", ".join(reducer_runtime_state.local_scratch_state),
        ", ".join(reducer_runtime_state.durable_state),
        reducer_runtime_state.scattering_interval_transport,
        reducer_runtime_state.uncommitted_restart_rule,
    )
    logger.debug(
        "Residual-field reducer committed shards | root=%s | storage=%s | compression=%s | direct_handoff=%s",
        reducer_runtime_state.committed_shard_root,
        reducer_runtime_state.committed_shard_storage,
        reducer_runtime_state.shard_compression,
        reducer_runtime_state.direct_interval_handoff_supported,
    )
    logger.debug(
        "Residual-field checkpoint policy | interval=%s | shards=%s | progress=%s | final=%s | scratch_role=%s",
        reducer_runtime_state.checkpoint_policy.interval_artifacts,
        reducer_runtime_state.checkpoint_policy.shard_checkpoints,
        reducer_runtime_state.checkpoint_policy.reducer_progress_manifest,
        reducer_runtime_state.checkpoint_policy.final_chunk_artifacts,
        reducer_runtime_state.checkpoint_policy.worker_local_scratch_role,
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
    total_partials_by_chunk = {
        int(chunk_id): sum(1 for work_unit in work_units if int(work_unit.chunk_id) == int(chunk_id))
        for chunk_id in chunk_ids
    }
    transient_interval_payloads = getattr(artifacts, "transient_interval_payloads", {}) or {}
    stage_task_logs = task_progress_enabled(False)
    if client is None or is_sync_client(client):
        rec = point_list_to_recarray(point_data_list)
        with progress_bar(total_tasks, desc="Residual field (chunks × shard-batches)", unit="batches") as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                manifest = run_residual_field_interval_chunk_task(
                    work_unit,
                    _interval_inputs_for_work_unit(
                        work_unit,
                        transient_interval_payloads=transient_interval_payloads,
                    ),
                    atoms,
                    total_reciprocal_points=total_reciprocal_points,
                    output_dir=artifacts.output_dir,
                    scratch_root=scratch_root,
                    reducer_backend=task_reducer_backend,
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
                elif isinstance(manifest, ResidualFieldLocalAccumulatorPartial):
                    reducer_backend.accept_partial(
                        manifest,
                        output_dir=artifacts.output_dir,
                        scratch_root=scratch_root,
                        db_path=artifacts.db_manager.db_path,
                        total_expected_partials=total_partials_by_chunk[int(work_unit.chunk_id)],
                        cleanup_policy=cleanup_policy,
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
            reducer_backend=reducer_backend,
            scratch_root=scratch_root,
        )
        if transient_interval_payloads:
            transient_interval_payloads.clear()
        logger.info("Residual-field finished (sync).")
        return

    fail_streak, fail_threshold = 0, 3
    gpu_tripped = False
    max_inflight = _cap_async_max_inflight(
        client=client,
        requested=max_inflight,
    )

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
    completed = 0

    if stage_task_logs:
        logger.info(
            "Residual-field start | batches=%d | chunks=%d | batch_size=%d | max_inflight=%d | cleanup=%s",
            int(total_tasks),
            int(len(chunk_ids)),
            int(max_intervals_per_shard),
            int(max_inflight),
            cleanup_policy,
        )

    def _submit(work_unit: ResidualFieldWorkUnit) -> None:
        nonlocal submitted
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            _interval_inputs_for_work_unit(
                work_unit,
                transient_interval_payloads=transient_interval_payloads,
            ),
            chunk_futures[int(work_unit.chunk_id)],
            total_reciprocal_points=total_reciprocal_points,
            output_dir=artifacts.output_dir,
            scratch_root=scratch_root,
            reducer_backend=task_reducer_backend,
            quiet_logs=True,
            key=f"residual-{work_unit.artifact_key}",
            pure=False,
            resources={"nufft": 1},
            retries=DEFAULT_TASK_RETRIES,
        )
        flying.add(future)
        future_meta[future] = work_unit
        submitted += 1
        if _should_log_async_progress(
            phase="queue",
            count=submitted,
            total=total_tasks,
        ):
            _log_async_residual_progress(
                enabled=stage_task_logs,
                event="queue",
                work_unit=work_unit,
                completed=completed,
                total=total_tasks,
                submitted=submitted,
                running=len(flying),
            )

    def _incorporate_completed_result(
        future,
        completed_work_unit: ResidualFieldWorkUnit | None,
    ) -> None:
        if completed_work_unit is None:
            return
        try:
            payload = future.result()
            if isinstance(payload, ResidualFieldLocalAccumulatorPartial):
                reducer_backend.accept_partial(
                    payload,
                    output_dir=artifacts.output_dir,
                    scratch_root=scratch_root,
                    db_path=artifacts.db_manager.db_path,
                    total_expected_partials=total_partials_by_chunk[
                        int(completed_work_unit.chunk_id)
                    ],
                    cleanup_policy=cleanup_policy,
                )
            else:
                manifests_by_chunk.setdefault(
                    int(completed_work_unit.chunk_id),
                    [],
                ).append(payload)
        except Exception:
            pass

    def _harvest_finished_nonblocking(bump) -> None:
        nonlocal completed, fail_streak
        done_now = [future for future in list(flying) if future.done()]
        for future in done_now:
            try:
                ok = future.result() is not None
            except Exception:
                ok = False
            flying.discard(future)
            work_unit = future_meta.pop(future, None)
            bump()
            completed += 1
            if work_unit is not None:
                if _should_log_async_progress(
                    phase="progress",
                    count=completed,
                    total=total_tasks,
                    force=not ok,
                ):
                    _log_async_residual_progress(
                        enabled=stage_task_logs,
                        event="progress",
                        work_unit=work_unit,
                        completed=completed,
                        total=total_tasks,
                        submitted=submitted,
                        running=len(flying),
                        detail=("failed" if not ok else "result-ready"),
                    )
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
                _incorporate_completed_result(future, work_unit)

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
                        completed += 1
                        if completed_work_unit is not None:
                            if _should_log_async_progress(
                                phase="progress",
                                count=completed,
                                total=total_tasks,
                                force=not ok,
                            ):
                                _log_async_residual_progress(
                                    enabled=stage_task_logs,
                                    event="progress",
                                    work_unit=completed_work_unit,
                                    completed=completed,
                                    total=total_tasks,
                                    submitted=submitted,
                                    running=len(flying),
                                    detail=("failed" if not ok else "result-ready"),
                                )
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
                            if result:
                                _incorporate_completed_result(future, completed_work_unit)

            for future, result in yield_futures_with_results(list(flying), client):
                flying.discard(future)
                completed_work_unit = future_meta.pop(future, None)
                bump()
                completed += 1
                if completed_work_unit is not None:
                    if _should_log_async_progress(
                        phase="progress",
                        count=completed,
                        total=total_tasks,
                        force=not bool(result),
                    ):
                        _log_async_residual_progress(
                            enabled=stage_task_logs,
                            event="progress",
                            work_unit=completed_work_unit,
                            completed=completed,
                            total=total_tasks,
                            submitted=submitted,
                            running=len(flying),
                            detail=("failed" if not bool(result) else "result-ready"),
                        )
                if not bool(result) and completed_work_unit is not None:
                    logger.error(
                        "GAVE UP after retries | residual batch %s | chunk %d",
                        ",".join(str(interval_id) for interval_id in completed_work_unit.interval_ids),
                        completed_work_unit.chunk_id,
                    )
                elif completed_work_unit is not None:
                    _incorporate_completed_result(future, completed_work_unit)

    _finalize_residual_field_chunks(
        chunk_ids=chunk_ids,
        parameter_digest=work_units[0].parameter_digest,
        manifests_by_chunk=manifests_by_chunk,
        output_dir=artifacts.output_dir,
        db_path=artifacts.db_manager.db_path,
        cleanup_policy=cleanup_policy,
        reducer_backend=reducer_backend,
        scratch_root=scratch_root,
    )
    if transient_interval_payloads:
        transient_interval_payloads.clear()
    logger.info("Residual-field finished – %d tasks submitted", submitted)


__all__ = ["build_residual_field_work_units", "run_residual_field_stage"]
