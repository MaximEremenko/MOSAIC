from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import time
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
    register_cleanup_plugin,
    resolve_worker_scratch_root,
    task_progress_enabled,
    yield_futures_with_results,
)
from core.residual_field.backend import (
    finalize_process_local_residual_chunk,
    flush_process_local_residual_reducer_target,
    inspect_process_local_residual_reducer_target,
    ResidualFieldReducerBackend,
    ResidualFieldLocalAccumulatorPartial,
    build_residual_field_reducer_backend,
    is_same_node_local_client,
    resolve_residual_field_reducer_backend,
)
from core.residual_field.contracts import (
    ResidualFieldAccumulatorStatus,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)
from core.residual_field.artifacts import (
    summarize_residual_field_output_artifacts,
    summarize_residual_field_shards,
)
from core.residual_field.planning import (
    _RESIDUAL_GRID_VALUE_BYTES_PER_POINT,
    _weighted_partition_split,
    build_adaptive_partition_plan,
    build_residual_field_work_units,
    partition_residual_field_work_units,
)
from core.residual_field.tasks import run_residual_field_interval_chunk_task

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_INTERVALS_PER_SHARD = 4


def _worker_owned_local_reducer_enabled(workflow_parameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    override = None
    if hasattr(runtime_info, "get"):
        override = runtime_info.get("residual_local_owner_reducer")
    if override is None:
        override = os.getenv("MOSAIC_RESIDUAL_LOCAL_OWNER_REDUCER")
    if override is None:
        return True
    if isinstance(override, str):
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(override)


def _distributed_owner_affinity_enabled(workflow_parameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    override = None
    if hasattr(runtime_info, "get"):
        override = runtime_info.get("residual_distributed_owner_affinity")
    if override is None:
        override = os.getenv("MOSAIC_RESIDUAL_DISTRIBUTED_OWNER_AFFINITY")
    if override is None:
        return True
    if isinstance(override, str):
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(override)


def _distributed_owner_local_reducer_supported(
    reducer_backend: ResidualFieldReducerBackend,
    *,
    reducer_runtime_state,
) -> bool:
    support_override = getattr(
        reducer_backend,
        "distributed_owner_local_reducer_supported",
        None,
    )
    if support_override is None:
        support_override = getattr(
            reducer_backend,
            "supports_distributed_owner_local_reducer",
            None,
        )
    if support_override is not None:
        return bool(support_override)
    return (
        callable(getattr(reducer_backend, "accept_local_contribution", None))
        and callable(getattr(reducer_backend, "inspect_local_reducer_target", None))
        and callable(getattr(reducer_backend, "flush_local_reducer_target", None))
        and getattr(reducer_runtime_state, "durable_truth_unit", None)
        == "committed_local_snapshot_generation"
        and getattr(reducer_runtime_state, "durable_checkpoint_storage_role", None)
        in {"durable-local-snapshot-generation", "durable-shared-generation"}
    )


def _owner_local_reducer_enabled(
    *,
    reducer_backend: ResidualFieldReducerBackend,
    worker_owned_local_reducer: bool,
    distributed_owner_local_reducer: bool,
) -> bool:
    return bool(worker_owned_local_reducer or distributed_owner_local_reducer)


def _residual_partition_runtime_policy(
    workflow_parameters,
    *,
    default_target_bytes: int,
    effective_nufft_workers: int,
) -> dict[str, int | float]:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}

    def _get_int(name: str, default: int) -> int:
        value = None
        if hasattr(runtime_info, "get"):
            value = runtime_info.get(name)
        if value is None:
            env_name = f"MOSAIC_{name.upper()}"
            value = os.getenv(env_name)
        return int(value) if value is not None else int(default)

    def _get_float(name: str, default: float) -> float:
        value = None
        if hasattr(runtime_info, "get"):
            value = runtime_info.get(name)
        if value is None:
            env_name = f"MOSAIC_{name.upper()}"
            value = os.getenv(env_name)
        return float(value) if value is not None else float(default)

    return {
        "target_partition_bytes": max(1, _get_int("residual_partition_target_bytes", int(default_target_bytes))),
        "target_partition_bytes_3d": max(
            1,
            _get_int(
                "residual_partition_target_bytes_3d",
                max(1, int(default_target_bytes) // 2),
            ),
        ),
        "max_partitions_per_chunk": max(
            1,
            _get_int(
                "residual_max_partitions_per_chunk",
                max(1, int(effective_nufft_workers) * 2),
            ),
        ),
        "min_points_per_partition": max(
            1,
            _get_int("residual_min_points_per_partition", 1),
        ),
        "hysteresis_low_factor": _get_float(
            "residual_partition_hysteresis_low",
            0.8,
        ),
        "hysteresis_high_factor": _get_float(
            "residual_partition_hysteresis_high",
            1.2,
        ),
    }


def _format_progress_bar(count: int, total: int, *, width: int = 20) -> str:
    total = max(int(total), 1)
    count = max(0, min(int(count), total))
    filled = int(round((count / float(total)) * width))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _planned_partition_imbalance_ratio(
    *,
    rifft_points_per_atom: tuple[int, ...],
    target_partitions: int,
) -> float:
    weights = np.asarray(rifft_points_per_atom, dtype=np.int64)
    if weights.size == 0 or target_partitions <= 1:
        return 1.0
    selections = _weighted_partition_split(
        np.arange(weights.shape[0], dtype=np.int64),
        weights,
        int(target_partitions),
    )
    partition_weights = [
        int(np.sum(weights[selection], dtype=np.int64))
        for selection in selections
        if selection.size > 0
    ]
    if not partition_weights:
        return 1.0
    min_weight = min(partition_weights)
    max_weight = max(partition_weights)
    if min_weight <= 0:
        return float("inf") if max_weight > 0 else 1.0
    return float(max_weight) / float(min_weight)


def _build_planned_target_metrics(
    partition_plans: dict[int, object],
) -> dict[tuple[int, int | None], dict[str, object]]:
    planned_metrics: dict[tuple[int, int | None], dict[str, object]] = {}
    for chunk_id, plan in partition_plans.items():
        weights = np.asarray(
            getattr(
                plan,
                "rifft_points_per_atom",
                np.ones(int(plan.point_count), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        target_partitions = int(getattr(plan, "target_partitions", 1))
        if target_partitions <= 1:
            planned_metrics[(int(chunk_id), None)] = {
                "planned_partition_count": 1,
                "planned_rifft_points": int(np.sum(weights, dtype=np.int64)),
                "planned_estimated_bytes": int(getattr(plan, "estimated_bytes", 0)),
                "target_partition_bytes": int(getattr(plan, "target_partition_bytes", 0)),
                "planned_imbalance_ratio": 1.0,
            }
            continue
        selections = _weighted_partition_split(
            np.arange(weights.shape[0], dtype=np.int64),
            weights,
            target_partitions,
        )
        imbalance_ratio = _planned_partition_imbalance_ratio(
            rifft_points_per_atom=tuple(int(value) for value in weights.tolist()),
            target_partitions=target_partitions,
        )
        for partition_id, selection in enumerate(selections):
            planned_rifft_points = int(np.sum(weights[selection], dtype=np.int64))
            planned_estimated_bytes = (
                planned_rifft_points * int(_RESIDUAL_GRID_VALUE_BYTES_PER_POINT)
            ) + (int(selection.size) * int(getattr(plan, "dimensionality", 1)) * 8)
            planned_metrics[(int(chunk_id), int(partition_id))] = {
                "planned_partition_count": int(target_partitions),
                "planned_rifft_points": int(planned_rifft_points),
                "planned_estimated_bytes": int(planned_estimated_bytes),
                "target_partition_bytes": int(getattr(plan, "target_partition_bytes", 0)),
                "planned_imbalance_ratio": float(imbalance_ratio),
            }
    return planned_metrics


def _log_partition_effectiveness_report(
    *,
    planned_target_metrics: dict[tuple[int, int | None], dict[str, object]],
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None],
) -> None:
    if not planned_target_metrics or not inspected_target_states:
        return
    for target_key in sorted(planned_target_metrics):
        planned = planned_target_metrics.get(target_key) or {}
        actual = inspected_target_states.get(target_key) or {}
        checkpoint_metrics = actual.get("checkpoint_metrics") if isinstance(actual, dict) else {}
        if not isinstance(checkpoint_metrics, dict):
            checkpoint_metrics = {}
        actual_checkpoint_bytes = int(
            checkpoint_metrics.get(
                "latest_checkpoint_bytes_written",
                checkpoint_metrics.get("total_checkpoint_bytes_written", 0),
            )
        )
        actual_checkpoint_writes = int(checkpoint_metrics.get("total_checkpoint_writes", 0))
        actual_checkpoint_wall = float(checkpoint_metrics.get("total_checkpoint_wall_seconds", 0.0))
        target_partition_bytes = int(planned.get("target_partition_bytes", 0))
        logger.info(
            "Residual-field partition report | target=%s | planned_rifft_points=%d | planned_bytes=%d | target_bytes=%d | actual_checkpoint_bytes=%d | actual_checkpoint_writes=%d | actual_checkpoint_wall=%.3fs | imbalance=%.3f | over_budget=%s",
            target_key,
            int(planned.get("planned_rifft_points", 0)),
            int(planned.get("planned_estimated_bytes", 0)),
            target_partition_bytes,
            actual_checkpoint_bytes,
            actual_checkpoint_writes,
            actual_checkpoint_wall,
            float(planned.get("planned_imbalance_ratio", 1.0)),
            str(bool(target_partition_bytes > 0 and actual_checkpoint_bytes > target_partition_bytes)).lower(),
        )


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


def _current_worker_addresses(client) -> list[str]:
    if client is None or is_sync_client(client):
        return []
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        workers = {}
    return sorted(workers)


def _resolve_owner_address(
    *,
    target_key: tuple[int, int | None],
    target_owners: dict[tuple[int, int | None], str],
    worker_addresses: list[str],
) -> str | None:
    current_owner = target_owners.get(target_key)
    if current_owner in worker_addresses:
        return current_owner
    if not worker_addresses:
        return current_owner
    live_owner_loads = {address: 0 for address in worker_addresses}
    for other_target_key, owner_address in target_owners.items():
        if other_target_key == target_key:
            continue
        if owner_address in live_owner_loads:
            live_owner_loads[owner_address] += 1
    replacement_owner = min(
        worker_addresses,
        key=lambda address: (live_owner_loads[address], address),
    )
    target_owners[target_key] = replacement_owner
    if current_owner is not None and current_owner != replacement_owner:
        logger.warning(
            "Residual-field owner remap | target=%s | previous=%s | replacement=%s",
            target_key,
            current_owner,
            replacement_owner,
        )
    return replacement_owner


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


def _reducer_target_key(work_unit: ResidualFieldWorkUnit) -> tuple[int, int | None]:
    return int(work_unit.chunk_id), (
        int(work_unit.partition_id) if work_unit.partition_id is not None else None
    )


def _unique_reducer_target_keys(
    work_units: list[ResidualFieldWorkUnit],
) -> list[tuple[int, int | None]]:
    return list(dict.fromkeys(_reducer_target_key(work_unit) for work_unit in work_units))


def _work_unit_expected_interval_ids(work_unit: ResidualFieldWorkUnit) -> tuple[int, ...]:
    if work_unit.interval_ids:
        return tuple(int(interval_id) for interval_id in work_unit.interval_ids)
    if work_unit.interval_id is None:
        return ()
    return (int(work_unit.interval_id),)


def _reconcile_and_filter_local_durable_work_units(
    *,
    work_units: list[ResidualFieldWorkUnit],
    reducer_backend: ResidualFieldReducerBackend,
    output_dir: str,
) -> list[ResidualFieldWorkUnit]:
    if not work_units:
        return work_units
    already_durable = getattr(reducer_backend, "local_intervals_already_durable", None)
    if not callable(already_durable):
        return work_units
    filtered = [
        work_unit
        for work_unit in work_units
        if not already_durable(
            work_unit,
            output_dir=output_dir,
        )
    ]
    if len(filtered) != len(work_units):
        logger.info(
            "Residual-field owner-local recovery filter | durable=%d | pending=%d",
            int(len(work_units) - len(filtered)),
            int(len(filtered)),
        )
    return filtered


def _validate_local_durable_coverage_or_raise(
    *,
    work_units: list[ResidualFieldWorkUnit],
    reducer_backend: ResidualFieldReducerBackend,
    output_dir: str,
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None] | None = None,
) -> dict[tuple[int, int | None], dict[str, object] | None]:
    if not work_units:
        if inspected_target_states is not None:
            return dict(inspected_target_states)
        return {}
    missing_by_target: dict[tuple[int, int | None], tuple[int, ...]] = {}
    resolved_states: dict[tuple[int, int | None], dict[str, object] | None] = {}
    for target_key in _unique_reducer_target_keys(work_units):
        representative = next(
            work_unit
            for work_unit in work_units
            if _reducer_target_key(work_unit) == target_key
        )
        if inspected_target_states is None:
            inspect_target = getattr(reducer_backend, "inspect_local_reducer_target", None)
            if inspect_target is None:
                raise RuntimeError(
                    "Residual-field owner-local finalize requires target inspection support "
                    "before publishing chunk artifacts."
                )
            target_state = inspect_target(
                chunk_id=int(representative.chunk_id),
                parameter_digest=str(representative.parameter_digest),
                output_dir=output_dir,
                partition_id=representative.partition_id,
            ) or {}
        else:
            target_state = inspected_target_states.get(target_key) or {}
        resolved_states[target_key] = target_state
        durable_interval_ids = set(
            int(interval_id) for interval_id in target_state.get("durable_interval_ids", ())
        )
        expected_interval_ids = {
            int(interval_id)
            for candidate in work_units
            if _reducer_target_key(candidate) == target_key
            for interval_id in _work_unit_expected_interval_ids(candidate)
        }
        missing_interval_ids = tuple(sorted(expected_interval_ids - durable_interval_ids))
        if missing_interval_ids:
            missing_by_target[target_key] = missing_interval_ids
    if missing_by_target:
        formatted = ", ".join(
            f"{target_key}:{list(interval_ids)}"
            for target_key, interval_ids in sorted(missing_by_target.items())
        )
        raise RuntimeError(
            "Residual-field owner-local finalize missing durable reducer-target coverage: "
            f"{formatted}"
        )
    return resolved_states


def _inspect_owner_local_reducer_targets_or_raise(
    *,
    client,
    template_backend: ResidualFieldReducerBackend,
    target_keys: list[tuple[int, int | None]],
    parameter_digest: str,
    output_dir: str,
    target_owners: dict[tuple[int, int | None], str],
) -> dict[tuple[int, int | None], dict[str, object] | None]:
    if not target_keys:
        return {}
    inspect_helper = inspect_process_local_residual_reducer_target
    if inspect_helper is None:
        raise RuntimeError(
            "Residual-field owner-local finalize requires per-target inspection support "
            "before publishing chunk artifacts."
        )
    inspect_futures = []
    target_keys_by_future = {}
    worker_addresses = _current_worker_addresses(client)
    for chunk_id, partition_id in target_keys:
        owner_address = _resolve_owner_address(
            target_key=(int(chunk_id), partition_id),
            target_owners=target_owners,
            worker_addresses=worker_addresses,
        )
        if owner_address is None:
            raise RuntimeError(
                "Residual-field owner-local finalize requires target ownership for "
                f"reducer target {(int(chunk_id), partition_id)}."
            )
        future = client.submit(
            inspect_helper,
            template_backend,
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            partition_id=partition_id,
            pure=False,
            workers=[owner_address],
            allow_other_workers=False,
        )
        inspect_futures.append(future)
        target_keys_by_future[future] = (int(chunk_id), partition_id)
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None] = {}
    for future, result in yield_futures_with_results(inspect_futures, client):
        if future is None:
            continue
        try:
            inspected_target_states[target_keys_by_future[future]] = future.result()
        except Exception:
            inspected_target_states[target_keys_by_future[future]] = None
    return inspected_target_states


def _log_owner_local_finalize_metrics(
    *,
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None],
    backend_kind: str,
) -> None:
    if not inspected_target_states:
        return
    total_bytes = 0
    total_writes = 0
    total_wall_seconds = 0.0
    saw_metrics = False
    for target_key in sorted(inspected_target_states):
        target_state = inspected_target_states.get(target_key) or {}
        checkpoint_metrics = target_state.get("checkpoint_metrics")
        if not isinstance(checkpoint_metrics, dict):
            checkpoint_metrics = target_state
        checkpoint_bytes = target_state.get(
            "total_checkpoint_bytes_written",
            checkpoint_metrics.get("total_checkpoint_bytes_written")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_bytes_written"),
        )
        checkpoint_writes = target_state.get(
            "total_checkpoint_writes",
            checkpoint_metrics.get("total_checkpoint_writes")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_writes"),
        )
        checkpoint_wall_seconds = target_state.get(
            "total_checkpoint_wall_seconds",
            checkpoint_metrics.get("total_checkpoint_wall_seconds")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_wall_seconds"),
        )
        if (
            checkpoint_bytes is None
            and checkpoint_writes is None
            and checkpoint_wall_seconds is None
        ):
            continue
        saw_metrics = True
        target_bytes = int(checkpoint_bytes or 0)
        target_writes = int(checkpoint_writes or 0)
        target_wall_seconds = float(checkpoint_wall_seconds or 0.0)
        total_bytes += target_bytes
        total_writes += target_writes
        total_wall_seconds += target_wall_seconds
        logger.info(
            "Residual-field finalize checkpoints | backend=%s | target=%s | writes=%d | bytes=%d | wall=%.3fs",
            backend_kind,
            target_key,
            target_writes,
            target_bytes,
            target_wall_seconds,
        )
    if saw_metrics:
        logger.info(
            "Residual-field finalize checkpoints total | backend=%s | targets=%d | writes=%d | bytes=%d | wall=%.3fs",
            backend_kind,
            int(len(inspected_target_states)),
            total_writes,
            total_bytes,
            total_wall_seconds,
        )


def _record_residual_task_result(
    *,
    payload,
    work_unit: ResidualFieldWorkUnit,
    manifests_by_chunk: dict[int, list[ResidualFieldShardManifest]],
) -> None:
    if payload is None or isinstance(payload, ResidualFieldAccumulatorStatus):
        return
    if isinstance(payload, ResidualFieldLocalAccumulatorPartial):
        raise RuntimeError(
            "Residual-field local tasks must return status-only results; "
            f"got driver-side partial for target {_reducer_target_key(work_unit)}."
        )
    manifests_by_chunk.setdefault(int(work_unit.chunk_id), []).append(payload)


def _flush_local_reducer_targets_or_raise(
    *,
    client,
    template_backend: ResidualFieldReducerBackend,
    target_keys: list[tuple[int, int | None]],
    parameter_digest: str,
    output_dir: str,
    db_path: str,
    target_owners: dict[tuple[int, int | None], str],
) -> None:
    if not target_keys:
        return
    flush_helper = flush_process_local_residual_reducer_target
    if flush_helper is None:
        chunk_target_counts: dict[int, int] = {}
        for chunk_id, _partition_id in target_keys:
            chunk_target_counts[int(chunk_id)] = chunk_target_counts.get(int(chunk_id), 0) + 1
        multi_owner_chunks = sorted(
            chunk_id for chunk_id, count in chunk_target_counts.items() if int(count) > 1
        )
        if multi_owner_chunks:
            raise RuntimeError(
                "Residual-field local finalize requires per-target flush support "
                f"for partitioned chunks: {multi_owner_chunks}"
            )
        return
    flush_futures = []
    worker_addresses = _current_worker_addresses(client)
    for chunk_id, partition_id in target_keys:
        owner_address = _resolve_owner_address(
            target_key=(int(chunk_id), partition_id),
            target_owners=target_owners,
            worker_addresses=worker_addresses,
        )
        if owner_address is None:
            continue
        flush_futures.append(
            client.submit(
                flush_helper,
                template_backend,
                chunk_id=int(chunk_id),
                parameter_digest=parameter_digest,
                output_dir=output_dir,
                db_path=db_path,
                partition_id=partition_id,
                pure=False,
                workers=[owner_address],
                allow_other_workers=False,
            )
        )
    for future, result in yield_futures_with_results(flush_futures, client):
        if future is None:
            continue


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
    expected_interval_ids_by_chunk: dict[int, tuple[int, ...]],
    output_dir: str,
    db_path: str,
    cleanup_policy: str,
    reducer_backend: ResidualFieldReducerBackend,
    scratch_root: str | None,
) -> None:
    for chunk_id in sorted(set(int(chunk_id) for chunk_id in chunk_ids)):
        shard_manifests = manifests_by_chunk.get(int(chunk_id))
        if not reducer_backend.uses_local_chunk_accumulator():
            reconciled_progress = reducer_backend.reconcile_progress(
                chunk_id=int(chunk_id),
                parameter_digest=parameter_digest,
                output_dir=output_dir,
                db_path=db_path,
                manifests=shard_manifests,
                scratch_root=scratch_root,
            )
            expected_interval_ids = set(
                int(interval_id)
                for interval_id in expected_interval_ids_by_chunk.get(int(chunk_id), ())
            )
            if reconciled_progress is not None:
                durable_interval_ids = set(
                    int(interval_id)
                    for interval_id in reconciled_progress.incorporated_interval_ids
                )
                missing_interval_ids = tuple(sorted(expected_interval_ids - durable_interval_ids))
                if missing_interval_ids:
                    raise RuntimeError(
                        "Residual-field distributed finalize missing durable coverage for "
                        f"chunk {int(chunk_id)}: {missing_interval_ids}"
                    )
                logger.info(
                    "Residual-field reconcile | chunk=%d | truth=%s | committed_shards=%d | pending_shards=%d | pending_intervals=%d",
                    int(chunk_id),
                    reconciled_progress.durable_truth_unit,
                    int(len(reconciled_progress.incorporated_shard_keys)),
                    int(len(reconciled_progress.pending_shard_keys)),
                    int(len(reconciled_progress.pending_interval_ids)),
                )
        shard_summary = summarize_residual_field_shards(shard_manifests or [])
        finalize_start = time.perf_counter()
        manifest = reducer_backend.finalize_chunk(
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=shard_manifests,
            cleanup_policy=cleanup_policy,
            scratch_root=scratch_root,
            quiet_logs=False,
        )
        if manifest is not None:
            output_summary = summarize_residual_field_output_artifacts(manifest.artifacts)
            logger.info(
                "Residual-field finalize | chunk=%d | shard_bytes=%d | final_bytes=%d | duration=%.3fs",
                int(chunk_id),
                int(shard_summary["committed_shard_bytes"]),
                int(output_summary["final_artifact_bytes"]),
                time.perf_counter() - finalize_start,
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
    register_cleanup_plugin(client, is_sync_client=is_sync_client)
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
    worker_owned_local_reducer = (
        reducer_backend.layout.kind == "local_restartable"
        and _worker_owned_local_reducer_enabled(workflow_parameters)
    )
    distributed_owner_local_reducer = (
        reducer_backend.layout.kind == "durable_shared_restartable"
        and _distributed_owner_affinity_enabled(workflow_parameters)
    )
    if reducer_backend.layout.kind == "local_restartable" and not worker_owned_local_reducer:
        raise ValueError(
            "Residual-field local execution requires worker-owned local reduction. "
            "The driver-owned local reducer path has been removed."
        )
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
        "Residual-field storage roles | truth=%s | live=%s | checkpoint=%s | final=%s",
        reducer_runtime_state.durable_truth_unit,
        reducer_runtime_state.live_state_storage_role,
        reducer_runtime_state.durable_checkpoint_storage_role,
        reducer_runtime_state.final_artifact_storage_role,
    )
    logger.debug(
        "Residual-field checkpoint policy | interval=%s | shards=%s | progress=%s | final=%s | scratch_role=%s",
        reducer_runtime_state.checkpoint_policy.interval_artifacts,
        reducer_runtime_state.checkpoint_policy.shard_checkpoints,
        reducer_runtime_state.checkpoint_policy.reducer_progress_manifest,
        reducer_runtime_state.checkpoint_policy.final_chunk_artifacts,
        reducer_runtime_state.checkpoint_policy.worker_local_scratch_role,
    )
    if reducer_backend.layout.kind == "durable_shared_restartable":
        if not distributed_owner_local_reducer:
            raise ValueError(
                "Residual-field distributed durable execution requires owner affinity. "
                "The shard-per-partial distributed path has been removed."
            )
        if not _distributed_owner_local_reducer_supported(
            reducer_backend,
            reducer_runtime_state=reducer_runtime_state,
        ):
            raise RuntimeError(
                "Residual-field distributed durable execution requires backend support "
                "for owner-local accumulation via accept_local_contribution, "
                "inspect_local_reducer_target, and flush_local_reducer_target."
            )
    owner_local_reducer = _owner_local_reducer_enabled(
        reducer_backend=reducer_backend,
        worker_owned_local_reducer=worker_owned_local_reducer,
        distributed_owner_local_reducer=distributed_owner_local_reducer,
    )
    cleanup_policy = _residual_shard_cleanup_policy(workflow_parameters)
    work_units = build_residual_field_work_units(
        artifacts.db_manager.get_unsaved_interval_chunks(),
        parameters=workflow_parameters,
        output_dir=artifacts.output_dir,
        max_intervals_per_shard=max_intervals_per_shard,
    )
    planned_target_metrics: dict[tuple[int, int | None], dict[str, object]] = {}
    chunk_ids = sorted({work_unit.chunk_id for work_unit in work_units})
    point_data_list: list[dict] = []
    for chunk_id in chunk_ids:
        point_data_list.extend(artifacts.db_manager.get_point_data_for_chunk(int(chunk_id)))

    if owner_local_reducer and client is not None and not is_sync_client(client) and work_units:
        point_rows_by_chunk = {
            int(chunk_id): [
                point_data
                for point_data in point_data_list
                if int(point_data["chunk_id"]) == int(chunk_id)
            ]
            for chunk_id in chunk_ids
        }
        local_partition_capacity = _cap_async_max_inflight(
            client=client,
            requested=max_inflight,
        )
        partition_policy = _residual_partition_runtime_policy(
            workflow_parameters,
            default_target_bytes=getattr(
                task_reducer_backend,
                "local_accumulator_max_ram_bytes",
                256 * 1024 * 1024,
            ),
            effective_nufft_workers=int(local_partition_capacity),
        )
        partition_plans = build_adaptive_partition_plan(
            point_rows_by_chunk,
            effective_nufft_workers=int(local_partition_capacity),
            target_partition_bytes=int(partition_policy["target_partition_bytes"]),
            target_partition_bytes_3d=int(partition_policy["target_partition_bytes_3d"]),
            max_partitions_per_chunk=int(partition_policy["max_partitions_per_chunk"]),
            min_points_per_partition=int(partition_policy["min_points_per_partition"]),
            hysteresis_low_factor=float(partition_policy["hysteresis_low_factor"]),
            hysteresis_high_factor=float(partition_policy["hysteresis_high_factor"]),
        )
        target_partitions_by_chunk = {
            int(chunk_id): int(plan.target_partitions)
            for chunk_id, plan in partition_plans.items()
        }
        planned_target_metrics = _build_planned_target_metrics(partition_plans)
        rifft_points_by_chunk = {
            int(chunk_id): np.asarray(
                getattr(
                    plan,
                    "rifft_points_per_atom",
                    np.ones(int(plan.point_count), dtype=np.int64),
                ),
                dtype=np.int64,
            )
            for chunk_id, plan in partition_plans.items()
        }
        for chunk_id, plan in partition_plans.items():
            if plan.target_partitions > 1:
                low_threshold_bytes = int(
                    round(
                        float(plan.target_partition_bytes)
                        * float(partition_policy["hysteresis_low_factor"])
                    )
                )
                high_threshold_bytes = int(
                    round(
                        float(plan.target_partition_bytes)
                        * float(partition_policy["hysteresis_high_factor"])
                    )
                )
                partition_rifft_points = tuple(
                    int(value)
                    for value in getattr(plan, "partition_rifft_points", ())
                )
                planned_imbalance_ratio = float(
                    getattr(plan, "partition_imbalance_ratio", 0.0) or 0.0
                )
                if planned_imbalance_ratio <= 0.0:
                    planned_imbalance_ratio = _planned_partition_imbalance_ratio(
                        rifft_points_per_atom=getattr(plan, "rifft_points_per_atom", ()),
                        target_partitions=int(plan.target_partitions),
                    )
                logger.info(
                    "Residual-field partition plan | chunk=%d | dim=%d | points=%d | rifft_points=%d | est_bytes=%d | target_bytes=%d | hysteresis_band=%d-%d | partitions=%d | partition_rifft_points=%s | imbalance=%.3f | reason=%s",
                    int(chunk_id),
                    int(plan.dimensionality),
                    int(plan.point_count),
                    int(plan.estimated_rifft_points),
                    int(plan.estimated_bytes),
                    int(plan.target_partition_bytes),
                    int(low_threshold_bytes),
                    int(high_threshold_bytes),
                    int(plan.target_partitions),
                    partition_rifft_points,
                    float(planned_imbalance_ratio),
                    plan.reason,
                )
        if any(int(value) > 1 for value in target_partitions_by_chunk.values()):
            work_units = partition_residual_field_work_units(
                work_units,
                point_counts_by_chunk={
                    int(chunk_id): len(point_rows)
                    for chunk_id, point_rows in point_rows_by_chunk.items()
                },
                target_partitions_by_chunk=target_partitions_by_chunk,
                rifft_points_by_chunk=rifft_points_by_chunk,
            )

    planned_work_units = list(work_units)
    if owner_local_reducer:
        work_units = _reconcile_and_filter_local_durable_work_units(
            work_units=work_units,
            reducer_backend=task_reducer_backend,
            output_dir=artifacts.output_dir,
        )

    total_tasks = len(work_units)
    if total_tasks == 0 and not (owner_local_reducer and planned_work_units):
        logger.info("Residual-field skipped – no unsaved (interval, chunk) pairs.")
        return

    total_reciprocal_points = sum(
        reciprocal_space_points_counter(to_interval_dict(interval), structure.supercell)
        for interval in artifacts.padded_intervals
    )

    manifests_by_chunk: dict[int, list[ResidualFieldShardManifest]] = {}
    expected_interval_ids_by_chunk = {
        int(chunk_id): tuple(
            sorted(
                {
                    int(interval_id)
                    for work_unit in planned_work_units
                    if int(work_unit.chunk_id) == int(chunk_id)
                    for interval_id in _work_unit_expected_interval_ids(work_unit)
                }
            )
        )
        for chunk_id in chunk_ids
    }
    total_partials_by_target = {
        _reducer_target_key(work_unit): sum(
            1
            for candidate in work_units
            if _reducer_target_key(candidate) == _reducer_target_key(work_unit)
        )
        for work_unit in work_units
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
                    db_path=artifacts.db_manager.db_path,
                    scratch_root=scratch_root,
                    reducer_backend=task_reducer_backend,
                    total_expected_partials=total_partials_by_target[_reducer_target_key(work_unit)],
                    owner_local_reducer=owner_local_reducer,
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
                    _record_residual_task_result(
                        payload=manifest,
                        work_unit=work_unit,
                        manifests_by_chunk=manifests_by_chunk,
                    )
        if owner_local_reducer:
            local_flush = getattr(task_reducer_backend, "flush_local_reducer_target", None)
            for target_key in _unique_reducer_target_keys(work_units):
                representative = next(
                    work_unit
                    for work_unit in work_units
                    if _reducer_target_key(work_unit) == target_key
                )
                if local_flush is None:
                    raise RuntimeError(
                        "Residual-field local finalize requires target flush support "
                        "before publishing chunk artifacts."
                    )
                local_flush(
                    chunk_id=int(representative.chunk_id),
                    parameter_digest=str(representative.parameter_digest),
                    partition_id=representative.partition_id,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path,
                    cleanup_policy=cleanup_policy,
                )
            inspected_target_states = _validate_local_durable_coverage_or_raise(
                work_units=planned_work_units,
                reducer_backend=task_reducer_backend,
                output_dir=artifacts.output_dir,
            )
            _log_owner_local_finalize_metrics(
                inspected_target_states=inspected_target_states,
                backend_kind=reducer_backend.layout.kind,
            )
            _log_partition_effectiveness_report(
                planned_target_metrics=planned_target_metrics,
                inspected_target_states=inspected_target_states,
            )
            for chunk_id in chunk_ids:
                finalize_process_local_residual_chunk(
                    task_reducer_backend,
                    chunk_id=int(chunk_id),
                    parameter_digest=planned_work_units[0].parameter_digest,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path,
                    cleanup_policy=cleanup_policy,
                    scratch_root=scratch_root,
                    quiet_logs=False,
                )
        else:
            _finalize_residual_field_chunks(
                chunk_ids=chunk_ids,
                parameter_digest=work_units[0].parameter_digest,
                manifests_by_chunk=manifests_by_chunk,
                expected_interval_ids_by_chunk=expected_interval_ids_by_chunk,
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
    worker_addresses = _current_worker_addresses(client)
    owner_local_target_units = planned_work_units if owner_local_reducer else work_units
    target_owners = (
        {
            target_key: worker_addresses[index % len(worker_addresses)]
            for index, target_key in enumerate(_unique_reducer_target_keys(owner_local_target_units))
        }
        if owner_local_reducer and worker_addresses
        else {}
    )
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
        submit_kwargs = dict(
            total_reciprocal_points=total_reciprocal_points,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path,
            scratch_root=scratch_root,
            reducer_backend=task_reducer_backend,
            total_expected_partials=total_partials_by_target[_reducer_target_key(work_unit)],
            owner_local_reducer=owner_local_reducer,
            quiet_logs=True,
            key=f"residual-{work_unit.artifact_key}",
            pure=False,
            resources={"nufft": 1},
            retries=DEFAULT_TASK_RETRIES,
        )
        target_key = _reducer_target_key(work_unit)
        if target_key in target_owners:
            owner_address = _resolve_owner_address(
                target_key=target_key,
                target_owners=target_owners,
                worker_addresses=_current_worker_addresses(client),
            )
            if owner_address is not None:
                submit_kwargs["workers"] = [owner_address]
                submit_kwargs["allow_other_workers"] = False
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            _interval_inputs_for_work_unit(
                work_unit,
                transient_interval_payloads=transient_interval_payloads,
            ),
            chunk_futures[int(work_unit.chunk_id)],
            **submit_kwargs,
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
        payload = future.result()
        _record_residual_task_result(
            payload=payload,
            work_unit=completed_work_unit,
            manifests_by_chunk=manifests_by_chunk,
        )

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

    if owner_local_reducer and worker_addresses:
        _flush_local_reducer_targets_or_raise(
            client=client,
            template_backend=task_reducer_backend,
            target_keys=_unique_reducer_target_keys(work_units),
            parameter_digest=planned_work_units[0].parameter_digest,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path,
            target_owners=target_owners,
        )
        inspected_target_states = _inspect_owner_local_reducer_targets_or_raise(
            client=client,
            template_backend=task_reducer_backend,
            target_keys=_unique_reducer_target_keys(planned_work_units),
            parameter_digest=planned_work_units[0].parameter_digest,
            output_dir=artifacts.output_dir,
            target_owners=target_owners,
        )
        _validate_local_durable_coverage_or_raise(
            work_units=planned_work_units,
            reducer_backend=task_reducer_backend,
            output_dir=artifacts.output_dir,
            inspected_target_states=inspected_target_states,
        )
        _log_owner_local_finalize_metrics(
            inspected_target_states=inspected_target_states,
            backend_kind=reducer_backend.layout.kind,
        )
        _log_partition_effectiveness_report(
            planned_target_metrics=planned_target_metrics,
            inspected_target_states=inspected_target_states,
        )
        finalize_futures = []
        for chunk_id in chunk_ids:
            finalizer_owner = None
            if any(int(work_unit.chunk_id) == int(chunk_id) for work_unit in owner_local_target_units):
                representative = next(
                    work_unit
                    for work_unit in owner_local_target_units
                    if int(work_unit.chunk_id) == int(chunk_id)
                )
                finalizer_owner = _resolve_owner_address(
                    target_key=_reducer_target_key(representative),
                    target_owners=target_owners,
                    worker_addresses=_current_worker_addresses(client),
                )
            elif _current_worker_addresses(client):
                live_workers = _current_worker_addresses(client)
                finalizer_owner = live_workers[int(chunk_id) % len(live_workers)]
            if finalizer_owner is None:
                raise RuntimeError(
                    f"Owner-local residual finalization requires an available worker for chunk {int(chunk_id)}."
                )
            finalize_futures.append(
                client.submit(
                    finalize_process_local_residual_chunk,
                    task_reducer_backend,
                    chunk_id=int(chunk_id),
                    parameter_digest=planned_work_units[0].parameter_digest,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path,
                    cleanup_policy=cleanup_policy,
                    scratch_root=scratch_root,
                    quiet_logs=False,
                    pure=False,
                    workers=[finalizer_owner],
                    allow_other_workers=False,
                )
            )
        for future, result in yield_futures_with_results(finalize_futures, client):
            if not bool(result):
                raise RuntimeError("Owner-local residual finalization failed.")
    else:
        _finalize_residual_field_chunks(
            chunk_ids=chunk_ids,
            parameter_digest=planned_work_units[0].parameter_digest,
            manifests_by_chunk=manifests_by_chunk,
            expected_interval_ids_by_chunk=expected_interval_ids_by_chunk,
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
