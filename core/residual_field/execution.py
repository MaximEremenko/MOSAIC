from __future__ import annotations

import logging
import math
from collections import OrderedDict
import os
from pathlib import Path
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from core.scattering.kernels import (
    point_list_to_recarray,
    reciprocal_space_points_counter,
    to_interval_dict,
)
from core.scattering.contracts import build_interval_artifact_ref
from core.runtime import (
    DEFAULT_TASK_RETRIES,
    is_sync_client,
    logging_redirect_tqdm,
    nufft_task_resources,
    path_is_tmpfs,
    profile_output_filesystem,
    progress_bar,
    require_gpu_admission,
    register_cleanup_plugin,
    resolve_nufft_execution_settings,
    resolve_worker_scratch_root,
    runtime_provenance_for_attempt,
    short_path,
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
from core.residual_field.commit import build_residual_work_unit_digest
from core.residual_field.artifacts import (
    discover_residual_field_reducer_progress_manifest,
    summarize_residual_field_output_artifacts,
    summarize_residual_field_shards,
)
from core.residual_field.planning import (
    _RESIDUAL_GRID_VALUE_BYTES_PER_POINT,
    _weighted_partition_split,
    build_adaptive_partition_plan,
    build_residual_field_parameter_digest,
    build_residual_field_work_units,
    partition_residual_field_work_units,
    resolve_residual_shard_grid_budget_bytes,
    resolve_residual_shard_source_budget,
)
from core.residual_field.tasks import (
    _residual_lattice_fft_enabled,
    build_residual_rifft_payload,
    clear_residual_rifft_payload_cache,
    run_residual_field_interval_chunk_task,
)
from core.storage.digests import digest_dict
from core.storage.run_state_cache import (
    pending_residual_interval_chunks,
    rebuild_sqlite_cache_from_manifests,
)
from core.residual_field.runtime_policy import (
    DEFAULT_RESIDUAL_PARTITION_TARGET_BYTES,
    _cleanup_residual_attempts_enabled,
    _distributed_owner_affinity_enabled,
    _distributed_owner_local_reducer_supported,
    _memory_backpressure_poll_seconds,
    _memory_backpressure_threshold,
    _owner_local_reducer_enabled,
    _residual_attempt_cleanup_policy,
    _residual_nufft_policy,
    _residual_nufft_prefetch_factor,
    _residual_nufft_resources,
    _residual_nufft_settings,
    _residual_partition_runtime_policy,
    _residual_rifft_payload_reuse_enabled,
    _worker_owned_local_reducer_enabled,
)
from core.residual_field.progress_logging import (
    _build_planned_target_metrics,
    _format_elapsed_eta,
    _format_progress_bar,
    _log_async_residual_progress,
    _log_owner_local_finalize_metrics,
    _log_partition_effectiveness_report,
    _planned_partition_imbalance_ratio,
    _should_log_async_progress,
    _work_unit_interval_label,
)
from core.residual_field.cluster_helpers import (
    _cap_async_max_inflight,
    _clear_worker_rifft_payload_caches,
    _cluster_host_memory_pressure,
    _current_worker_addresses,
    _resolve_owner_address,
    _scheduler_nufft_capacity,
    _trim_workers_for_memory_pressure,
)
from core.residual_field.work_unit_utils import (
    _hex64_or_digest,
    _interval_inputs_for_work_unit,
    _interval_paths_for_work_unit,
    _reducer_target_key,
    _sort_work_units_by_target,
    _unique_reducer_target_keys,
    _work_unit_expected_interval_ids,
    _work_unit_sort_key,
)

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_INTERVALS_PER_SHARD = 4


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


def _residual_current_identity(
    *,
    workflow_parameters,
    work_units: list[ResidualFieldWorkUnit],
    point_counts_by_chunk: dict[int, int],
    backend_kind: str,
    max_intervals_per_batch: int,
) -> dict[str, str | None]:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    if not hasattr(runtime_info, "get"):
        runtime_info = {}
    run_digest = (
        runtime_info.get("residual_run_digest")
        or runtime_info.get("residual_field_run_digest")
        or runtime_info.get("scattering_run_digest")
    )
    if not run_digest:
        run_digest = digest_dict(
            {
                "parameter_digest": work_units[0].parameter_digest if work_units else "",
                "chunks": sorted(point_counts_by_chunk),
            },
            domain="mosaic.residual_field.run_digest.v1",
        )
    source_scattering_commit_digest = _hex64_or_digest(
        runtime_info.get("source_scattering_commit_digest")
        or runtime_info.get("scattering_stage_digest")
        or runtime_info.get("scattering_run_digest")
        or run_digest,
        domain="mosaic.residual_field.source_scattering_commit.v1",
    )
    source_replacement = runtime_info.get("source_replacement_digest")
    return {
        "run_digest": str(run_digest),
        "partition_plan_digest": digest_dict(
            {
                "point_counts_by_chunk": point_counts_by_chunk,
                "work_units": [
                    {
                        "chunk_id": int(work_unit.chunk_id),
                        "interval_ids": list(_work_unit_expected_interval_ids(work_unit)),
                    }
                    for work_unit in work_units
                ],
            },
            domain="mosaic.residual_field.partition_plan.v1",
        ),
        "source_scattering_commit_digest": source_scattering_commit_digest,
        "source_replacement_digest": (
            None
            if source_replacement in {None, ""}
            else _hex64_or_digest(
                source_replacement,
                domain="mosaic.residual_field.source_replacement.v1",
            )
        ),
        "backend_policy_digest": digest_dict(
            {
                "backend_kind": str(backend_kind),
                "max_intervals_per_batch": int(max_intervals_per_batch),
                "protocol": "attempt-commit",
            },
            domain="mosaic.residual_field.backend_policy.v1",
        ),
        "expected_output_digest": digest_dict(
            {
                "point_counts_by_chunk": point_counts_by_chunk,
                "work_units": [
                    {
                        "chunk_id": int(work_unit.chunk_id),
                        "interval_ids": list(_work_unit_expected_interval_ids(work_unit)),
                        "parameter_digest": work_unit.parameter_digest,
                    }
                    for work_unit in work_units
                ],
            },
            domain="mosaic.residual_field.expected_output.v1",
        ),
    }


def _identity_complete_residual_work_units(
    *,
    work_units: list[ResidualFieldWorkUnit],
    point_data_list: list[dict],
    workflow_parameters,
    backend_kind: str,
    max_intervals_per_batch: int,
    force_partition: bool = True,
) -> list[ResidualFieldWorkUnit]:
    point_counts_by_chunk: dict[int, int] = {}
    for row in point_data_list:
        chunk_id = int(row["chunk_id"])
        point_counts_by_chunk[chunk_id] = point_counts_by_chunk.get(chunk_id, 0) + 1
    identity = _residual_current_identity(
        workflow_parameters=workflow_parameters,
        work_units=work_units,
        point_counts_by_chunk=point_counts_by_chunk,
        backend_kind=backend_kind,
        max_intervals_per_batch=max_intervals_per_batch,
    )
    completed: list[ResidualFieldWorkUnit] = []
    for work_unit in work_units:
        with_identity = replace(work_unit, **identity)
        if force_partition and with_identity.partition_id is None:
            point_count = int(point_counts_by_chunk.get(int(with_identity.chunk_id), 0))
            if point_count <= 0:
                point_count = 1
            with_identity = with_identity.with_partition(
                partition_id=0,
                point_start=0,
                point_stop=point_count,
            )
        completed.append(with_identity)
    return completed


def _scheduler_kind(client) -> str:
    if client is None or is_sync_client(client):
        return "sync"
    return "dask"


def _runtime_provenance_for_residual(
    *,
    workflow_parameters,
    fs_capability_digest: str | None,
    client,
) -> dict[str, object]:
    settings = _residual_nufft_settings(workflow_parameters)
    provenance = runtime_provenance_for_attempt(
        fs_capability_digest=fs_capability_digest,
        scheduler_kind=_scheduler_kind(client),
        nufft_policy=settings.execution_policy,
        resource_requirements=_residual_nufft_resources(workflow_parameters),
        cuda_probe=settings.gpu_only,
    )
    provenance["nufft_execution_settings"] = settings.identity_payload()
    return provenance


def _residual_work_unit_digest(work_unit: ResidualFieldWorkUnit) -> str:
    # Device-independent checkpoint address (backend policy is metadata,
    # not identity) so CPU and GPU residual work units share one address.
    return build_residual_work_unit_digest(
        run_digest=str(work_unit.run_digest),
        chunk_id=int(work_unit.chunk_id),
        partition_id=int(work_unit.partition_id),
        point_start=int(work_unit.point_start),
        point_stop=int(work_unit.point_stop),
        interval_ids=tuple(int(item) for item in work_unit.interval_ids),
        parameter_digest=str(work_unit.parameter_digest),
        partition_plan_digest=str(work_unit.partition_plan_digest),
        source_scattering_commit_digest=str(work_unit.source_scattering_commit_digest),
        source_replacement_digest=work_unit.source_replacement_digest,
        expected_output_digest=str(work_unit.expected_output_digest),
    )


def _expected_partition_family_for_chunk(
    planned_work_units: list[ResidualFieldWorkUnit],
    *,
    chunk_id: int,
) -> tuple[tuple[int | None, int | None, int | None], ...]:
    """The plan's (partition_id, point_start, point_stop) family for a chunk.

    Passed to finalize so the snapshot family is checked against the plan --
    the only source of truth that can see a missing tail partition.
    """
    family: dict[int | None, tuple[int | None, int | None]] = {}
    for work_unit in planned_work_units:
        if int(work_unit.chunk_id) != int(chunk_id):
            continue
        family[work_unit.partition_id] = (
            None if work_unit.point_start is None else int(work_unit.point_start),
            None if work_unit.point_stop is None else int(work_unit.point_stop),
        )
    return tuple(
        (partition_id, start, stop)
        for partition_id, (start, stop) in sorted(
            family.items(),
            key=lambda item: (-1 if item[0] is None else int(item[0])),
        )
    )


def _streaming_subchunk_slot_count(workflow_parameters, client) -> int:
    """Number of subchunk slots per chunk in streaming mode.

    The slot count is part of work-unit/checkpoint identity, so it must
    NOT depend on how many workers happen to be alive: a run started on a
    1-GPU node has to resume on an 8-GPU node with its checkpoints intact.
    Fixed default of 8 keeps every realistic worker count folding
    concurrently; ``runtime_info.residual_streaming_subchunks`` overrides.
    Sync execution gets one slot (a single accumulator per chunk)."""
    raw = workflow_parameters.runtime_info.get("residual_streaming_subchunks")
    if raw is not None:
        return max(1, int(raw))
    if client is None or is_sync_client(client):
        return 1
    return 8


def _streaming_slot_owner_map(
    target_keys,
    worker_addresses: list[str],
) -> dict[tuple[int, int | None], str]:
    """Owner assignment round-robin over the sorted (chunk, slot) targets.

    Slot-keyed placement (slot % workers) balanced unit COUNTS but not
    WORK: slots are content-addressed from batch interval ids, and a
    sparse mask concentrates nearly all real scattering volume in one
    slot's batches — observed on the hkl40 'rod' case, whose second half
    ran entirely on ONE GPU while three folded mask-empty no-ops. Spreading
    each slot's per-chunk accumulators across workers quarters that skew.
    The old co-location rationale (one worker computes a batch's payloads
    once for all chunks) is obsolete with the durable stage-1 store: any
    worker reads the payloads at disk speed, and the lattice entry a batch
    needs is rebuilt at most once per owning worker, overlapped with other
    units' transforms. Runtime-only placement — never part of work-unit
    or checkpoint identity."""
    # SLOT-major ordering: a slot's per-chunk accumulators take consecutive
    # ranks, so one heavy slot spreads across workers even in the degenerate
    # case where targets-per-chunk divides the worker count (chunk-major
    # ordering collapses back to slot-keyed placement exactly then).
    ordered = sorted(
        {
            (int(key[1]) if key[1] is not None else -1, int(key[0]))
            for key in target_keys
        }
    )
    rank = {target: index for index, target in enumerate(ordered)}
    return {
        target_key: worker_addresses[
            rank[
                (
                    int(target_key[1]) if target_key[1] is not None else -1,
                    int(target_key[0]),
                )
            ]
            % len(worker_addresses)
        ]
        for target_key in target_keys
    }


def _mark_finalized_chunk_intervals_saved(*, db_path, chunk_id, interval_ids):
    """Driver-side SQLite marking for streaming finalizes (single writer).

    Marks the PLAN's interval set: _validate_local_durable_coverage_or_raise
    proved the durable union covers the plan, and finalize_chunk's family
    validation raises unless the snapshot union equals it exactly."""
    from core.residual_field.reducer_helpers import _mark_residual_intervals_saved

    _mark_residual_intervals_saved(
        db_path=db_path, chunk_id=int(chunk_id), interval_ids=tuple(interval_ids)
    )


def _streaming_finalize_owner_map(
    chunk_ids,
    worker_addresses: list[str],
) -> dict[int, str]:
    """Finalize ownership keyed by CHUNK, round-robin across workers.

    Finalize reads the durable slot snapshots from the shared output
    directory, so unlike fold tasks it has no slot-owner locality tie —
    but the slot-keyed map sends EVERY chunk's finalize to the same
    worker (every chunk shares the identical content-addressed slot set),
    serializing ~34 GB reads + 13.5 GB writes per chunk on one worker
    while the rest idle. Deterministic and independent of the fold-owner
    map (which _resolve_owner_address may mutate on remap)."""
    return {
        int(chunk_id): worker_addresses[index % len(worker_addresses)]
        for index, chunk_id in enumerate(sorted(int(c) for c in chunk_ids))
    }


def _prewarm_stage1_store_if_enabled(
    *,
    client,
    streaming_context,
    streaming_context_future,
    work_units,
    worker_addresses,
    workflow_parameters,
) -> None:
    """Fill the durable stage-1 payload store in parallel across ALL workers.

    Without this, each interval's first compute happens inside its owner's
    first fold of the shard: a serial, mostly-CPU prologue on one worker while
    every other GPU idles (measured on hkl40: multi-minute all-idle windows,
    25+ min for one sparse-mask shard). Prewarm batches are NOT slot-pinned,
    so stage-1 spreads over every GPU up front; the store then makes every
    later load — including restarts on any cluster size — a memmap read.
    Best-effort: any failure degrades to the in-fold compute path.
    ``MOSAIC_STREAMING_STAGE1_PREWARM=0`` disables."""
    if streaming_context is None or streaming_context_future is None:
        return
    if not worker_addresses:
        return
    store_dir = getattr(streaming_context, "payload_store_dir", None)
    if not store_dir:
        return
    raw = os.getenv("MOSAIC_STREAMING_STAGE1_PREWARM", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return
    try:
        from core.scattering.streaming import (
            prewarm_stage1_payload_store,
            stage1_store_has,
        )

        interval_ids = sorted(
            {
                int(interval_id)
                for work_unit in work_units
                for interval_id in _work_unit_expected_interval_ids(work_unit)
            }
        )
        missing = [
            interval_id
            for interval_id in interval_ids
            if not stage1_store_has(store_dir, interval_id)
        ]
        if not missing:
            if interval_ids:
                logger.info(
                    "Stage-1 payload store already complete (%d interval(s)).",
                    len(interval_ids),
                )
            return
        # ~4 batches per worker: enough tasks to balance uneven interval
        # costs without paying scheduler overhead per interval.
        batch_count = max(1, min(len(missing), 4 * len(worker_addresses)))
        batch_size = -(-len(missing) // batch_count)
        batches = [
            missing[start : start + batch_size]
            for start in range(0, len(missing), batch_size)
        ]
        settings = _residual_nufft_settings(workflow_parameters)
        start_time = time.perf_counter()
        logger.info(
            "Prewarming stage-1 payload store: %d interval(s) in %d batch(es) "
            "across %d worker(s) -> %s",
            len(missing),
            len(batches),
            len(worker_addresses),
            store_dir,
        )
        futures = [
            client.submit(
                prewarm_stage1_payload_store,
                batch,
                streaming_context_future,
                nufft_eps=settings.eps,
                nufft_prefer_cpu=settings.prefer_cpu,
                nufft_gpu_only=settings.gpu_only,
                resources={"nufft": 1},
                retries=1,
                pure=False,
            )
            for batch in batches
        ]
        computed = sum(int(count or 0) for count in client.gather(futures))
        logger.info(
            "Stage-1 payload store prewarmed: %d interval(s) in %.1fs.",
            computed,
            time.perf_counter() - start_time,
        )
    except Exception:
        logger.warning(
            "Stage-1 store prewarm failed; falling back to in-fold compute.",
            exc_info=True,
        )


def _sort_streaming_work_units_batch_major(
    work_units: list[ResidualFieldWorkUnit],
    target_owners: dict[tuple[int, int | None], str] | None = None,
) -> list[ResidualFieldWorkUnit]:
    """Batch-major submission order for streaming work units.

    Sorts by (interval batch, chunk) so the SAME batch's units for different
    chunks are adjacent: the batch's stage-1 payloads are computed once, all
    chunks fold them, and only then does the next batch start. Combined with
    the capped per-worker payload memo this is the all-in-RAM constraint —
    the 1 GiB memo only ever needs the CURRENT batch, never more than one
    batch's payloads at a time. Chunk-major (plan) order would instead touch
    every batch once per chunk and thrash the memo.

    Across queues the round-robin is keyed by the RESOLVED OWNER, not the
    slot: slot-keyed interleave degenerates whenever worker count divides
    chunk count (hkl40's 4 chunks on 4 GPUs), because (slot*C + c) mod W
    collapses to chunk-only placement and every slot queue's head is the
    lowest chunk — the first S in-flight units all land on ONE worker unless
    prefetch covers ~3S units. Owner-keyed queues make the first W
    submissions cover W distinct workers at any prefetch, while each owner's
    OWN queue stays batch-major (the memo/lattice-cache locality is per
    worker). Falls back to slot keying when no owner map exists (sync
    clients)."""
    batch_major = sorted(
        work_units,
        key=lambda work_unit: (
            _work_unit_expected_interval_ids(work_unit),
            int(work_unit.chunk_id),
        ),
    )
    if target_owners:
        def _queue_key(work_unit):
            return target_owners.get(
                (int(work_unit.chunk_id), work_unit.partition_id),
                work_unit.partition_id,
            )
    else:
        def _queue_key(work_unit):
            return work_unit.partition_id
    slot_queues: "OrderedDict[object, list[ResidualFieldWorkUnit]]" = OrderedDict()
    for work_unit in batch_major:
        slot_queues.setdefault(_queue_key(work_unit), []).append(work_unit)
    if len(slot_queues) <= 1:
        return batch_major
    interleaved: list[ResidualFieldWorkUnit] = []
    queues = [iter(queue) for queue in slot_queues.values()]
    while queues:
        remaining = []
        for queue in queues:
            unit = next(queue, None)
            if unit is not None:
                interleaved.append(unit)
                remaining.append(queue)
        queues = remaining
    return interleaved


def _invalidate_incompatible_local_checkpoints(
    *,
    planned_work_units: list[ResidualFieldWorkUnit],
    reducer_backend: ResidualFieldReducerBackend,
    output_dir: str,
) -> None:
    """Drop on-disk checkpoints whose partition layout no longer matches the plan.

    Without this, snapshots surviving a crash across a code/configuration change
    (different partition count or atom ranges) stay referenced by the progress
    manifest and are concatenated at finalize, silently corrupting the chunk.
    """
    invalidate = getattr(
        reducer_backend, "invalidate_incompatible_local_checkpoints", None
    )
    if not callable(invalidate):
        return
    expected_by_chunk: dict[tuple[int, str], dict[int | None, dict[str, object]]] = {}
    for work_unit in planned_work_units:
        key = (int(work_unit.chunk_id), str(work_unit.parameter_digest))
        target = expected_by_chunk.setdefault(key, {}).setdefault(
            work_unit.partition_id,
            {
                "point_start": (
                    None if work_unit.point_start is None else int(work_unit.point_start)
                ),
                "point_stop": (
                    None if work_unit.point_stop is None else int(work_unit.point_stop)
                ),
                # Snapshots from the OTHER partition axis must be dropped, not
                # reconciled: a points-axis checkpoint surviving into a
                # streaming (intervals-axis) plan, or vice versa, describes a
                # different decomposition of the same chunk.
                "partition_axis": getattr(work_unit, "partition_axis", "points"),
                "interval_batches": [],
            },
        )
        target["interval_batches"].append(
            frozenset(
                int(interval_id)
                for interval_id in _work_unit_expected_interval_ids(work_unit)
            )
        )
    for (chunk_id, parameter_digest), expected_targets in expected_by_chunk.items():
        for target in expected_targets.values():
            target["interval_batches"] = tuple(target["interval_batches"])
        invalidate(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            expected_targets=expected_targets,
        )


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


def _dead_cluster_horizon_seconds() -> float:
    try:
        return max(
            60.0,
            float(os.getenv("MOSAIC_RESIDUAL_DEAD_CLUSTER_HORIZON_SECONDS", "900")),
        )
    except ValueError:
        return 900.0


def _barrier_future_ok(future) -> bool:
    status = getattr(future, "status", None)
    if status is not None and status != "finished":
        return False
    try:
        result = future.result()
    except Exception:
        return False
    return result is not None and result is not False


def _drain_owner_pinned_barrier(
    *,
    client,
    futures_by_key: dict,
    owner_by_key: dict,
    resubmit,
    barrier_name: str,
    timeout_seconds: float = 45.0,
):
    """Yield (key, future, ok) for owner-pinned barrier futures, rescuing any
    whose pinned worker left the cluster.

    A future pinned with workers=[owner], allow_other_workers=False whose only
    allowed worker died parks in no-worker state as 'pending' FOREVER (nanny
    restarts come back on NEW addresses), so a bare as_completed here hangs
    the driver — the same failure the fold drain loop already rescues; the
    flush/inspect/finalize barriers after it did not. Waits in bounded slices,
    cancels futures pinned to dead owners, and resubmits via
    ``resubmit(key, new_owner)`` on a live worker. Raises if the cluster has
    no live workers for longer than the dead-cluster horizon."""
    if client is None or is_sync_client(client):
        reverse = {future: key for key, future in futures_by_key.items()}
        for future, _ok in yield_futures_with_results(
            list(futures_by_key.values()), client
        ):
            key = reverse.get(future)
            if key is not None:
                yield key, future, _barrier_future_ok(future)
        return
    from distributed import wait as _distributed_wait

    pending = dict(futures_by_key)
    owners = dict(owner_by_key)
    dead_since: float | None = None
    rescued = 0
    while pending:
        live = _current_worker_addresses(client)
        if not live:
            now = time.monotonic()
            if dead_since is None:
                dead_since = now
            elif now - dead_since >= _dead_cluster_horizon_seconds():
                raise RuntimeError(
                    f"Residual-field {barrier_name} barrier: no live workers "
                    f"for {_dead_cluster_horizon_seconds():.0f}s with "
                    f"{len(pending)} owner-pinned task(s) outstanding."
                )
            time.sleep(min(5.0, timeout_seconds))
            continue
        dead_since = None
        live_set = set(live)
        for key, future in list(pending.items()):
            status = getattr(future, "status", "")
            if status in ("finished", "error"):
                continue
            owner = owners.get(key)
            if status not in ("cancelled", "lost") and (
                owner is None or owner in live_set
            ):
                continue
            try:
                future.cancel()
            except Exception:
                pass
            new_owner = live[rescued % len(live)]
            rescued += 1
            logger.warning(
                "Residual-field %s barrier: owner %s for %s is gone; "
                "resubmitting on %s",
                barrier_name,
                owner,
                key,
                new_owner,
            )
            pending[key] = resubmit(key, new_owner)
            owners[key] = new_owner
        try:
            _distributed_wait(
                list(pending.values()),
                timeout=timeout_seconds,
                return_when="FIRST_COMPLETED",
            )
        except TimeoutError:
            continue
        except Exception:
            # Comm hiccups/test doubles can make wait() raise immediately —
            # take one completion through the blocking generator instead of
            # spinning on retry.
            for future, ok in yield_futures_with_results(
                list(pending.values()), client
            ):
                for key, pending_future in list(pending.items()):
                    if pending_future is future:
                        del pending[key]
                        yield key, future, ok
                        break
                break
            continue
        for key, future in list(pending.items()):
            done = getattr(future, "done", None)
            try:
                is_done = future.done() if callable(done) else True
            except Exception:
                is_done = True
            if not is_done:
                continue
            del pending[key]
            yield key, future, _barrier_future_ok(future)


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
    worker_addresses = _current_worker_addresses(client)

    def _submit_inspect(target_key, owner_address):
        chunk_id, partition_id = target_key
        return client.submit(
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

    futures_by_key = {}
    owner_by_key = {}
    for chunk_id, partition_id in target_keys:
        target_key = (int(chunk_id), partition_id)
        owner_address = _resolve_owner_address(
            target_key=target_key,
            target_owners=target_owners,
            worker_addresses=worker_addresses,
        )
        if owner_address is None:
            raise RuntimeError(
                "Residual-field owner-local finalize requires target ownership for "
                f"reducer target {target_key}."
            )
        futures_by_key[target_key] = _submit_inspect(target_key, owner_address)
        owner_by_key[target_key] = owner_address
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None] = {}
    # Inspection reads durable state from the shared output dir, so a dead
    # owner's inspect is safely remapped to any live worker.
    for target_key, future, ok in _drain_owner_pinned_barrier(
        client=client,
        futures_by_key=futures_by_key,
        owner_by_key=owner_by_key,
        resubmit=_submit_inspect,
        barrier_name="inspect",
    ):
        try:
            inspected_target_states[target_key] = future.result()
        except Exception:
            inspected_target_states[target_key] = None
    return inspected_target_states


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
    pre_submitted_futures: dict[tuple[int, int | None], object] | None = None,
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
    worker_addresses = _current_worker_addresses(client)

    def _submit_flush(target_key, owner_address):
        chunk_id, partition_id = target_key
        return client.submit(
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

    futures_by_key = {}
    owner_by_key = {}
    for chunk_id, partition_id in target_keys:
        target_key = (int(chunk_id), partition_id)
        pre_submitted = (pre_submitted_futures or {}).get(target_key)
        if pre_submitted is not None:
            pre_future, pre_owner = pre_submitted
            pre_status = getattr(pre_future, "status", "")
            if pre_status == "finished" or (
                pre_status not in ("error", "cancelled", "lost")
                and pre_owner in worker_addresses
            ):
                # Early per-target flush already done or in flight on a
                # LIVE owner — the barrier just waits on it.
                futures_by_key[target_key] = pre_future
                owner_by_key[target_key] = pre_owner
                continue
            # Owner died (a worker-pinned future for a dead worker parks in
            # no-worker state as 'pending' FOREVER — reusing it would hang
            # the barrier). Cancel and resubmit through the owner remap.
            cancel = getattr(pre_future, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
        owner_address = _resolve_owner_address(
            target_key=target_key,
            target_owners=target_owners,
            worker_addresses=worker_addresses,
        )
        if owner_address is None:
            continue
        futures_by_key[target_key] = _submit_flush(target_key, owner_address)
        owner_by_key[target_key] = owner_address
    # A dead owner's RAM state is unrecoverable, so a remapped flush is a
    # no-op on the new worker — the point is converting a silent barrier
    # hang into the durable-coverage validation's clean fail-stop.
    for _key, _future, _ok in _drain_owner_pinned_barrier(
        client=client,
        futures_by_key=futures_by_key,
        owner_by_key=owner_by_key,
        resubmit=_submit_flush,
        barrier_name="flush",
    ):
        pass


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
        # Only the sync-client owner-local path reaches here; the old
        # reconcile pre-check served the non-owner-local shard/attempt
        # universe, which the two backend-kind guards make unreachable.
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
            expected_interval_ids=expected_interval_ids_by_chunk.get(int(chunk_id)),
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


def _drop_mask_emptied_interval_chunks(
    interval_chunk_pairs,
    *,
    output_dir,
    transient_interval_payloads,
    parameter_digest,
    logger,
):
    """Drop (interval, chunk) pairs whose interval produced no scattering output.

    A reciprocal-space mask can eliminate every Q-point in a subvolume (e.g. a
    half-integer superlattice mask empties the entire l=0 plane).  Such intervals
    have no precomputed ``interval_<id>.hdf5`` artifact and no transient in-memory
    payload; their residual-field contribution is exactly zero.  The scattering
    stage marks them complete, but ``rebuild_sqlite_cache_from_manifests`` resets
    every interval-chunk to unsaved and only re-marks intervals that have a
    manifest, so mask-emptied intervals resurface here and would otherwise make
    the residual stage try to open files that were never written.

    A genuine scattering failure aborts Stage-1 before the residual stage runs,
    but "no artifact and no payload" alone is NOT enough to conclude the mask
    emptied the interval: on a restart, an interval whose transient payload was
    already consumed and folded into a committed durable generation looks
    identical.  Such pairs must stay in the plan so the finalize/repair path can
    re-mark them saved and apply the cleanup policy; only intervals that are
    also absent from the chunk's durable reducer progress are dropped.
    """
    payloads = transient_interval_payloads or {}
    available: dict[int, bool] = {}
    incorporated_by_chunk: dict[int, frozenset[int]] = {}

    def _has_payload(interval_id: int) -> bool:
        cached = available.get(interval_id)
        if cached is not None:
            return cached
        present = int(interval_id) in payloads
        if not present:
            artifact_path = build_interval_artifact_ref(output_dir, int(interval_id)).path
            present = artifact_path is not None and os.path.exists(artifact_path)
        available[interval_id] = present
        return present

    def _durably_incorporated(interval_id: int, chunk_id: int) -> bool:
        incorporated = incorporated_by_chunk.get(chunk_id)
        if incorporated is None:
            progress = discover_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            incorporated = (
                frozenset(int(value) for value in progress.incorporated_interval_ids)
                if progress is not None
                else frozenset()
            )
            incorporated_by_chunk[chunk_id] = incorporated
        return interval_id in incorporated

    kept = [
        (interval_id, chunk_id)
        for interval_id, chunk_id in interval_chunk_pairs
        if _has_payload(int(interval_id))
        or _durably_incorporated(int(interval_id), int(chunk_id))
    ]
    dropped_intervals = sorted(
        {int(interval_id) for interval_id, _ in interval_chunk_pairs}
        - {int(interval_id) for interval_id, _ in kept}
    )
    if dropped_intervals:
        logger.info(
            "Residual-field: skipping %d mask-emptied interval(s) with no scattering "
            "output (zero contribution): %s%s",
            len(dropped_intervals),
            dropped_intervals[:20],
            " ..." if len(dropped_intervals) > 20 else "",
        )
    return kept


def _lattice_grid_capped_intervals_per_shard(
    intervals,
    supercell,
    budget_bytes: int,
) -> tuple[int, int]:
    """Grid-size-aware cap on how many intervals fold into one lattice shard.

    The lattice (scatter+type-2) residual path materialises one dense
    coefficient grid per shard spanning the bounding box of the shard's
    intervals at a reciprocal pitch of ``1/supercell_axis`` r.l.u. per axis.
    Model: the dense grid for the FULL interval union spans, per axis,
    ``(global_qmax - global_qmin) / pitch + 1`` points, and costs
    ``prod(dims) * 16 bytes * 2`` (two complex128 transforms).  The per-shard
    bounding box of a contiguous run of reciprocal-space-sorted interval ids
    scales roughly with its interval fraction, so the shard size is capped at
    ``max(1, floor(n_intervals * budget_bytes / full_union_grid_bytes))``;
    the memmap spill remains the safety net for outlier shards.

    This bounds the DEFAULT durable-mode shard fold so hkl40-scale runs no
    longer build whole-extent (tens-of-GiB) grids by default.  Returns
    ``(capped_max_intervals_per_shard, full_union_grid_bytes)``; when the
    full-union grid fits the budget the cap equals ``len(intervals)`` (the
    existing fold-everything behaviour is kept).
    """
    interval_dicts = [to_interval_dict(interval) for interval in intervals]
    n_intervals = max(1, len(interval_dicts))
    supercell = np.asarray(supercell, dtype=float)
    grid_points = 1
    for axis, cells in zip(("h", "k", "l"), supercell):
        starts = [d[f"{axis}_start"] for d in interval_dicts if f"{axis}_start" in d]
        ends = [d[f"{axis}_end"] for d in interval_dicts if f"{axis}_end" in d]
        if not starts or not ends:
            continue
        pitch = 1.0 / float(cells)
        span = max(ends) - min(starts)
        grid_points *= max(1, int(math.floor(span / pitch + 0.5)) + 1)
    full_union_grid_bytes = int(grid_points) * 16 * 2
    if full_union_grid_bytes <= int(budget_bytes):
        return n_intervals, full_union_grid_bytes
    cap = max(1, (n_intervals * int(budget_bytes)) // full_union_grid_bytes)
    return min(n_intervals, cap), full_union_grid_bytes


def _adaptive_residual_intervals_per_shard(
    *,
    artifacts,
    structure,
    source_budget: int,
) -> int:
    intervals = list(artifacts.padded_intervals)
    n_intervals = max(1, len(intervals))
    # The residual inverse concatenates the actual q_grid rows as source points.
    # Those rows are multiplicity-free; half-space conjugate reconstruction is
    # applied after the inverse.  Using multiplicity-folded dense counts here
    # overestimates source memory and splits batches unnecessarily.
    total_source_points = sum(
        int(
            reciprocal_space_points_counter(
                to_interval_dict(interval),
                structure.supercell,
                include_multiplicity=False,
            )
        )
        for interval in intervals
    )
    avg_source_points = max(1, int(total_source_points) // n_intervals)
    return max(1, min(n_intervals, int(source_budget) // avg_source_points))


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
    _env_shard = os.getenv("MOSAIC_RESIDUAL_INTERVALS_PER_SHARD")
    _configured_shard = workflow_parameters.runtime_info.get("residual_shard_batch_size")
    if _env_shard is not None:
        max_intervals_per_shard = max(1, int(_env_shard))
    elif _configured_shard is not None:
        max_intervals_per_shard = max(1, int(_configured_shard))
    else:
        # Adaptive default: fold as many intervals as possible into ONE shard so the
        # inverse transform is issued as FEW, LARGE GPU calls (concat source-batching,
        # high GPU utilisation) instead of one tiny task per interval. Bound the fold by
        # a source-point budget so wide-hkl 3D keeps the concatenated q-list -- and hence
        # the type-3 fine grid -- within VRAM. 2D folds all intervals; huge 3D caps.
        try:
            if _residual_lattice_fft_enabled():
                # Lattice (scatter+type-2) path: one shard covering every
                # interval means ONE scattered coefficient grid, cached and
                # reused by all (chunk, partition) work units. The type-3
                # source-point budget is irrelevant -- lattice memory is set by
                # the grid dims, not the point count. Bound the fold by a
                # projected-grid-bytes budget so wide-hkl 3D (e.g. hkl40) does
                # not build a whole-extent dense grid by default.
                _lattice_intervals = list(artifacts.padded_intervals)
                max_intervals_per_shard = max(1, len(_lattice_intervals))
                _grid_budget = resolve_residual_shard_grid_budget_bytes()
                _grid_cap, _full_grid_bytes = _lattice_grid_capped_intervals_per_shard(
                    _lattice_intervals,
                    structure.supercell,
                    _grid_budget,
                )
                if _grid_cap < max_intervals_per_shard:
                    logger.info(
                        "Residual-field lattice shard fold capped by projected "
                        "grid size: %d -> %d intervals/shard (full-union grid "
                        "~%.2f GiB > budget %.2f GiB)",
                        max_intervals_per_shard,
                        _grid_cap,
                        _full_grid_bytes / float(1024**3),
                        _grid_budget / float(1024**3),
                    )
                    max_intervals_per_shard = _grid_cap
            else:
                _budget = resolve_residual_shard_source_budget()
                max_intervals_per_shard = _adaptive_residual_intervals_per_shard(
                    artifacts=artifacts,
                    structure=structure,
                    source_budget=_budget,
                )
        except Exception:
            logger.debug("Adaptive residual shard sizing failed; using default.", exc_info=True)
            max_intervals_per_shard = DEFAULT_RESIDUAL_INTERVALS_PER_SHARD
    logger.info(
        "Residual-field interval shard size | max_intervals_per_shard=%d "
        "(env=%s, config=%s)",
        max_intervals_per_shard,
        _env_shard,
        _configured_shard,
    )
    if explicit_scratch_root is not None:
        preferred_scratch = explicit_scratch_root
    elif preliminary_backend.layout.kind == "local_restartable":
        # Live accumulators are node-local WORKING state; only durable
        # snapshots belong on the shared output dir. Defaulting scratch to
        # output_dir streamed every GB-scale live memmap over NFS on
        # multi-node runs — the multi-node-safety rationale in the backend
        # selection assumes node-local scratch. Default to the worker
        # scratch base instead, unless that base is tmpfs (RAM), where the
        # shared output dir remains the safer default.
        import tempfile as _tempfile

        preferred_scratch = None
        if os.getenv("MOSAIC_WORKER_SCRATCH_ROOT") is None and path_is_tmpfs(
            _tempfile.gettempdir()
        ):
            preferred_scratch = str(Path(artifacts.output_dir) / ".local_restartable")
    else:
        preferred_scratch = None
    scratch_root = resolve_worker_scratch_root(
        preferred=preferred_scratch,
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
        short_path(reducer_runtime_state.local_scratch_root),
        short_path(reducer_runtime_state.durable_root),
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
    cleanup_policy = _residual_attempt_cleanup_policy(workflow_parameters)
    streaming_context = (getattr(artifacts, "streaming_state", None) or {}).get(
        "compute_context"
    )
    if streaming_context is not None and (
        not owner_local_reducer
        or reducer_backend.layout.kind != "local_restartable"
    ):
        raise RuntimeError(
            "Streaming stage-2 mode requires the local_restartable owner-local "
            "reducer: its amplitudes exist only inside worker accumulators, and "
            "the durable_shared generation checkpoints carry no interval-axis "
            "(subchunk) semantics. Disable streaming or switch the reducer "
            "backend."
        )
    all_interval_chunk_pairs = list(
        artifacts.db_manager.get_interval_chunks()
        if hasattr(artifacts.db_manager, "get_interval_chunks")
        else artifacts.db_manager.get_unsaved_interval_chunks()
    )
    if streaming_context is None:
        all_interval_chunk_pairs = _drop_mask_emptied_interval_chunks(
            all_interval_chunk_pairs,
            output_dir=artifacts.output_dir,
            transient_interval_payloads=getattr(artifacts, "transient_interval_payloads", {}) or {},
            parameter_digest=build_residual_field_parameter_digest(workflow_parameters),
            logger=logger,
        )
    # else: streaming mode has no interval artifacts or payload dict to probe;
    # mask-emptiness is discovered inside the work unit, which folds an exact
    # zero and still records the interval as incorporated.
    #
    # Interval GEOMETRY for shard packing: interval ids run one reciprocal
    # axis at a time, so a contiguous id run spans nearly the full q-volume
    # and its dense lattice grid blows past every budget (hkl40: 31 GiB ->
    # RAM-admission failure -> silent type-3 fallback). Handing the planner
    # each id's bounds lets it pack shards by spatial bounding box instead.
    interval_geometry: dict[int, dict] | None = None
    try:
        unique_interval_ids = sorted(
            {int(interval_id) for interval_id, _chunk in all_interval_chunk_pairs}
        )
        interval_geometry = {
            int(record.interval_id): {
                "h_start": record.h_range[0],
                "h_end": record.h_range[1],
                "k_start": record.k_range[0],
                "k_end": record.k_range[1],
                "l_start": record.l_range[0],
                "l_end": record.l_range[1],
            }
            for record in artifacts.db_manager.get_intervals_by_ids(
                unique_interval_ids
            )
        }
    except Exception:
        logger.debug(
            "Interval geometry unavailable; shard packing falls back to "
            "contiguous id slicing.",
            exc_info=True,
        )
        interval_geometry = None
    _shard_packing_kwargs = {
        "interval_geometry": interval_geometry,
        "supercell": getattr(structure, "supercell", None),
        "grid_budget_bytes": resolve_residual_shard_grid_budget_bytes(),
    }
    initial_work_units = build_residual_field_work_units(
        all_interval_chunk_pairs,
        parameters=workflow_parameters,
        output_dir=artifacts.output_dir,
        max_intervals_per_shard=max_intervals_per_shard,
        **_shard_packing_kwargs,
    )
    initial_chunk_ids = sorted({work_unit.chunk_id for work_unit in initial_work_units})
    point_data_list: list[dict] = []
    for chunk_id in initial_chunk_ids:
        point_data_list.extend(artifacts.db_manager.get_point_data_for_chunk(int(chunk_id)))
    initial_identity_units = _identity_complete_residual_work_units(
        work_units=initial_work_units,
        point_data_list=point_data_list,
        workflow_parameters=workflow_parameters,
        backend_kind=reducer_backend.layout.kind,
        max_intervals_per_batch=max_intervals_per_shard,
        force_partition=not owner_local_reducer,
    )
    if initial_identity_units:
        run_digest = str(initial_identity_units[0].run_digest)
        snapshot = rebuild_sqlite_cache_from_manifests(
            artifacts.db_manager,
            output_dir=artifacts.output_dir,
            run_digest=run_digest,
            # Credits COMMITTED streaming reducer progress (status-only
            # results write no payload manifests) so a completed case's
            # retry skips the residual stage instead of re-deriving it.
            residual_parameter_digest=str(
                initial_identity_units[0].parameter_digest
            ),
        )
        pending_pairs = pending_residual_interval_chunks(
            snapshot,
            all_interval_chunk_pairs,
            output_dir=artifacts.output_dir,
            residual_parameter_digest=str(
                initial_identity_units[0].parameter_digest
            ),
        )
    else:
        pending_pairs = []

    work_units = build_residual_field_work_units(
        pending_pairs,
        parameters=workflow_parameters,
        output_dir=artifacts.output_dir,
        max_intervals_per_shard=max_intervals_per_shard,
        **_shard_packing_kwargs,
    )
    planned_target_metrics: dict[tuple[int, int | None], dict[str, object]] = {}
    chunk_ids = sorted({work_unit.chunk_id for work_unit in work_units})
    point_data_list = []
    for chunk_id in chunk_ids:
        point_data_list.extend(artifacts.db_manager.get_point_data_for_chunk(int(chunk_id)))

    work_units = _identity_complete_residual_work_units(
        work_units=work_units,
        point_data_list=point_data_list,
        workflow_parameters=workflow_parameters,
        backend_kind=reducer_backend.layout.kind,
        max_intervals_per_batch=max_intervals_per_shard,
        force_partition=not owner_local_reducer,
    )

    if streaming_context is not None and work_units:
        # Streaming subchunks: partition on the INTERVAL axis instead of the
        # atom axis. Every batch unit covers its chunk's full point range and
        # is routed to a content-addressed slot; each slot's accumulator sums
        # a disjoint interval subset, and finalize merges slots by summation
        # (validated by the disjoint-union family checks).
        from core.scattering.streaming import streaming_slot_map

        point_counts_by_chunk: dict[int, int] = {}
        for point_data in point_data_list:
            chunk_key = int(point_data["chunk_id"])
            point_counts_by_chunk[chunk_key] = point_counts_by_chunk.get(chunk_key, 0) + 1
        n_slots = _streaming_subchunk_slot_count(workflow_parameters, client)
        slot_by_batch = streaming_slot_map(
            (
                _work_unit_expected_interval_ids(work_unit)
                for work_unit in work_units
            ),
            n_slots,
        )
        work_units = [
            work_unit.with_subchunk(
                subchunk_id=slot_by_batch[
                    tuple(
                        int(interval_id)
                        for interval_id in _work_unit_expected_interval_ids(work_unit)
                    )
                ],
                point_count=point_counts_by_chunk[int(work_unit.chunk_id)],
            )
            for work_unit in work_units
        ]
        logger.info(
            "Residual-field streaming mode: %d batch unit(s) across %d "
            "subchunk slot(s); interval payloads computed in-task, no "
            "durable interval store.",
            len(work_units),
            n_slots,
        )

    if streaming_context is None and owner_local_reducer and client is not None and not is_sync_client(client) and work_units:
        point_rows_by_chunk = {
            int(chunk_id): [
                point_data
                for point_data in point_data_list
                if int(point_data["chunk_id"]) == int(chunk_id)
            ]
            for chunk_id in chunk_ids
        }
        # Capacity CONSTANT, not live scheduler capacity: partition point
        # ranges are written into snapshots and compared on resume, so
        # deriving this from client.scheduler_info() made durable-mode
        # checkpoints worker-count DEPENDENT — a 4-GPU run resumed on an
        # 8-GPU node changed every atom range and discarded every partition
        # snapshot. Fixed default 8 mirrors the streaming slot count (any
        # realistic worker count keeps folding concurrently);
        # runtime_info.residual_partition_capacity overrides, and the knob
        # is part of the parameter digest.
        raw_partition_capacity = workflow_parameters.runtime_info.get(
            "residual_partition_capacity"
        )
        local_partition_capacity = (
            max(1, int(raw_partition_capacity))
            if raw_partition_capacity is not None
            else 8
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
    fs_capability_digest: str | None = None
    runtime_provenance: dict[str, object] | None = None
    nufft_resources = _residual_nufft_resources(workflow_parameters)
    nufft_settings = _residual_nufft_settings(workflow_parameters)
    if planned_work_units:
        fs_capability = profile_output_filesystem(
            artifacts.output_dir,
            run_digest=str(planned_work_units[0].run_digest),
            client=client,
        )
        fs_capability_digest = fs_capability.capability_digest
        runtime_provenance = _runtime_provenance_for_residual(
            workflow_parameters=workflow_parameters,
            fs_capability_digest=fs_capability_digest,
            client=client,
        )
        if client is not None and not is_sync_client(client):
            require_gpu_admission(
                client,
                policy=_residual_nufft_policy(workflow_parameters),
                required_gpu_tasks=(1 if "gpu" in nufft_resources else 0),
            )
    if owner_local_reducer:
        _invalidate_incompatible_local_checkpoints(
            planned_work_units=planned_work_units,
            reducer_backend=task_reducer_backend,
            output_dir=artifacts.output_dir,
        )
        work_units = _reconcile_and_filter_local_durable_work_units(
            work_units=work_units,
            reducer_backend=task_reducer_backend,
            output_dir=artifacts.output_dir,
        )
    reuse_rifft_payload = _residual_rifft_payload_reuse_enabled(workflow_parameters)
    if reuse_rifft_payload:
        planned_work_units = _sort_work_units_by_target(planned_work_units)
        # Streaming keeps batch-major submission order (below) instead of the
        # target-major re-sort; planned_work_units ordering is untouched — it
        # feeds finalize/invalidation expectations, not submission.
        if streaming_context is None:
            work_units = _sort_work_units_by_target(work_units)
    # Streaming submission order is applied AFTER the owner map is built
    # (below) so the interleave can round-robin by resolved owner.

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
    stage_task_logs = task_progress_enabled(True)
    if reuse_rifft_payload:
        logger.info("Residual-field RIFFT payload reuse enabled.")
    if client is None or is_sync_client(client):
        rec = point_list_to_recarray(point_data_list)
        current_rifft_target: tuple[int, int | None] | None = None
        current_rifft_payload: tuple[np.ndarray, np.ndarray] | None = None
        with progress_bar(total_tasks, desc="Residual-field", unit="batch", force=True) as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                if reuse_rifft_payload:
                    target_key = _reducer_target_key(work_unit)
                    if target_key != current_rifft_target:
                        current_rifft_payload = None
                        current_rifft_payload = build_residual_rifft_payload(
                            atoms,
                            work_unit=work_unit,
                            quiet_logs=False,
                        )
                        current_rifft_target = target_key
                manifest = run_residual_field_interval_chunk_task(
                    work_unit,
                    ()
                    if streaming_context is not None
                    else _interval_inputs_for_work_unit(
                        work_unit,
                        transient_interval_payloads=transient_interval_payloads,
                    ),
                    None if reuse_rifft_payload else atoms,
                    total_reciprocal_points=total_reciprocal_points,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path if owner_local_reducer else None,
                    scratch_root=scratch_root,
                    reducer_backend=task_reducer_backend,
                    total_expected_partials=total_partials_by_target[_reducer_target_key(work_unit)],
                    owner_local_reducer=owner_local_reducer,
                    quiet_logs=False,
                    streaming_compute_context=streaming_context,
                    rifft_payload=current_rifft_payload if reuse_rifft_payload else None,
                    runtime_provenance=runtime_provenance,
                    nufft_eps=nufft_settings.eps,
                    nufft_prefer_cpu=nufft_settings.prefer_cpu,
                    nufft_gpu_only=nufft_settings.gpu_only,
                )
                pbar.update(1)
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
                    expected_partitions=_expected_partition_family_for_chunk(
                        planned_work_units,
                        chunk_id=int(chunk_id),
                    ),
                    expected_interval_ids=expected_interval_ids_by_chunk.get(
                        int(chunk_id)
                    ),
                )
        else:
            # Owner-local reduction is force-enabled for both backend kinds
            # (the two ValueError guards above are the contract), so tasks
            # return status-only results and there is no attempt/candidate
            # universe to quiesce — the old require_chunk_quiescence
            # ceremony here scanned for attempts that can never exist.
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
        _clear_worker_rifft_payload_caches(client)
        logger.info("Residual-field finished (sync).")
        return

    fail_streak, fail_threshold = 0, 3
    gpu_tripped = False
    last_cpu_trip_broadcast = 0.0
    residual_prefetch_factor = _residual_nufft_prefetch_factor(workflow_parameters)
    max_inflight = _cap_async_max_inflight(
        client=client,
        requested=max_inflight,
        prefetch_factor=residual_prefetch_factor,
    )

    def _trip_to_cpu_only() -> None:
        # A worker the nanny restarts AFTER the trip comes back GPU-enabled
        # while the driver still believes gpu_tripped. Its failures rebuild
        # fail_streak past the threshold, which re-enters here — so the
        # set_cpu_only broadcast is re-sent (rate-limited) instead of
        # one-shot, pulling restarted workers back into the CPU-only regime.
        nonlocal gpu_tripped, max_inflight, last_cpu_trip_broadcast
        now = time.monotonic()
        if gpu_tripped and now - last_cpu_trip_broadcast < 60.0:
            return
        if hasattr(client, "run"):
            try:
                from core.adapters.cunufft_wrapper import set_cpu_only

                client.run(set_cpu_only, True)
            except Exception:
                pass
        last_cpu_trip_broadcast = now
        if gpu_tripped:
            return
        max_inflight = min(max_inflight, 256)
        gpu_tripped = True
        logger.warning("Circuit-breaker: switching residual-field to CPU-only & throttling.")

    rec = point_list_to_recarray(point_data_list)
    chunk_futures = {
        chunk_id: client.scatter(rec[rec.chunk_id == chunk_id], broadcast=False, hash=False)
        for chunk_id in chunk_ids
    }
    # Streaming mode: the scattering compute context (structure arrays, mask
    # strategy, form factors) ships to every worker exactly once.
    streaming_context_future = (
        client.scatter(streaming_context, broadcast=True, hash=False)
        if streaming_context is not None
        else None
    )
    worker_addresses = _current_worker_addresses(client)
    _prewarm_stage1_store_if_enabled(
        client=client,
        streaming_context=streaming_context,
        streaming_context_future=streaming_context_future,
        work_units=work_units,
        worker_addresses=worker_addresses,
        workflow_parameters=workflow_parameters,
    )
    owner_local_target_units = planned_work_units if owner_local_reducer else work_units
    # Streaming: ownership keyed by SLOT alone (partition_id is never None for
    # streaming units) — the same batch content hashes to the same slot for
    # every chunk, so slot-keyed ownership makes each batch's stage-1 compute
    # land on one worker and be reused for all chunks (payload memo locality).
    # Non-streaming keeps the byte-identical round-robin by enumeration index.
    target_owners = (
        (
            _streaming_slot_owner_map(
                _unique_reducer_target_keys(owner_local_target_units),
                worker_addresses,
            )
            if streaming_context is not None
            else {
                target_key: worker_addresses[index % len(worker_addresses)]
                for index, target_key in enumerate(_unique_reducer_target_keys(owner_local_target_units))
            }
        )
        if owner_local_reducer and worker_addresses
        else {}
    )
    if streaming_context is not None:
        # Batch-major ordering means the 1 GiB payload memo only ever needs
        # the CURRENT batch (the all-in-RAM constraint): every chunk folds a
        # batch before the next batch's stage-1 payloads are computed.
        work_units = _sort_streaming_work_units_batch_major(
            work_units, target_owners=target_owners
        )
    retries_left = {
        (str(work_unit.artifact_key), int(work_unit.chunk_id)): DEFAULT_TASK_RETRIES
        for work_unit in work_units
    }
    # Infrastructure failures (worker OOM-kill/restart, cancelled/lost comms)
    # say nothing about the task and get their OWN budget with exponential
    # backoff: a nanny-restart window otherwise burned entire retry budgets
    # in seconds (observed: 15/32 units exhausted inside one 3-minute
    # restart storm, every retry dying instantly on comm timeouts).
    try:
        infra_retry_budget = max(
            1, int(os.getenv("MOSAIC_RESIDUAL_INFRA_RETRIES", "12"))
        )
    except ValueError:
        infra_retry_budget = 12
    infra_failures_seen: dict[tuple[str, int], int] = {}
    # A unit that repeatedly KILLS its worker is indistinguishable from
    # infrastructure by error class (KilledWorker) but is really a poison
    # task — without its own cap it would enjoy the LARGEST retry budget
    # while physically destroying workers. Same-key worker kills get a
    # smaller cap; exceeding it fails the unit (no breaker coupling).
    try:
        killed_worker_retry_cap = max(
            1, int(os.getenv("MOSAIC_RESIDUAL_KILLED_WORKER_RETRIES", "4"))
        )
    except ValueError:
        killed_worker_retry_cap = 4
    killed_worker_seen: dict[tuple[str, int], int] = {}
    # Fencing epochs per reducer target: bumped whenever ownership MOVES
    # (dead-owner rescue, retry remap). Passed into the fold task; the
    # accumulator sequences each tenure's snapshots from epoch * STRIDE and
    # the manifest union refuses to move a partition's seq backwards, so a
    # scheduler-declared-dead-but-alive predecessor cannot overwrite or
    # race the replacement's snapshots (it fails loudly at commit instead).
    target_owner_epochs: dict[tuple[int, int | None], int] = {}
    target_last_owner: dict[tuple[int, int | None], str] = {}
    deferred_resubmits: list = []  # (eligible_monotonic_time, work_unit)
    target_rifft_futures: dict[tuple[int, int | None], object] = {}
    target_remaining = {
        target_key: sum(1 for work_unit in work_units if _reducer_target_key(work_unit) == target_key)
        for target_key in _unique_reducer_target_keys(work_units)
    }
    flying: set = set()
    future_meta: dict = {}
    future_pinned_owner: dict = {}
    # Wall-clock since the cluster last had a live worker; the drain loop
    # aborts past the horizon instead of spinning on wait timeouts forever
    # (SLURM allocation revoked / all nodes dead with the scheduler alive).
    dead_cluster_since: list = [None]
    exhausted_failures: list[tuple[ResidualFieldWorkUnit, str]] = []
    submitted = 0
    completed = 0
    last_memory_pressure_trim = 0.0
    last_memory_pressure_check = time.monotonic()

    _residual_start_time = time.monotonic()

    if stage_task_logs:
        logger.info(
            "Residual-field start | batches=%d | chunks=%d | batch_size=%d | max_inflight=%d | cleanup=%s",
            int(total_tasks),
            int(len(chunk_ids)),
            int(max_intervals_per_shard),
            int(max_inflight),
            cleanup_policy,
        )

    def _target_rifft_payload_future(
        work_unit: ResidualFieldWorkUnit,
        *,
        owner_address: str | None = None,
    ):
        if not reuse_rifft_payload:
            return None
        target_key = _reducer_target_key(work_unit)
        existing = target_rifft_futures.get(target_key)
        if existing is not None:
            return existing
        submit_kwargs = dict(
            key=(
                "riff-grid:residual:"
                f"chunk-{int(work_unit.chunk_id)}:"
                f"partition-{work_unit.partition_id if work_unit.partition_id is not None else 'owner'}"
            ),
            pure=False,
        )
        if owner_address is not None:
            submit_kwargs["workers"] = [owner_address]
            submit_kwargs["allow_other_workers"] = False
        future = client.submit(
            build_residual_rifft_payload,
            chunk_futures[int(work_unit.chunk_id)],
            work_unit=work_unit,
            quiet_logs=False,
            **submit_kwargs,
        )
        target_rifft_futures[target_key] = future
        return future

    early_flush_futures: dict[tuple[int, int | None], object] = {}

    def _submit_early_target_flush(target_key) -> None:
        """Flush a reducer target the moment its LAST work unit completes.

        The end-of-stage barrier previously wrote ~271 GB of snapshots with
        zero compute overlap (90-135 s, GPUs idle). Targets finish staggered
        (slot-interleaved submission), so streaming their final snapshots
        during the tail degrades the barrier to a short wait on writes that
        are already running. Failure here is harmless: the key stays absent
        and the barrier resubmits fresh."""
        if (
            not owner_local_reducer
            or not worker_addresses
            or flush_process_local_residual_reducer_target is None
            or target_key not in target_owners
            or target_key in early_flush_futures
        ):
            return
        try:
            owner_address = _resolve_owner_address(
                target_key=target_key,
                target_owners=target_owners,
                worker_addresses=_current_worker_addresses(client),
            )
            if owner_address is None:
                return
            # Store the pinned owner with the future: the barrier must not
            # wait on a future whose only allowed worker has since died.
            early_flush_futures[target_key] = (
                client.submit(
                    flush_process_local_residual_reducer_target,
                    task_reducer_backend,
                    chunk_id=int(target_key[0]),
                    parameter_digest=planned_work_units[0].parameter_digest,
                    output_dir=artifacts.output_dir,
                    db_path=artifacts.db_manager.db_path,
                    partition_id=target_key[1],
                    pure=False,
                    workers=[owner_address],
                    allow_other_workers=False,
                ),
                owner_address,
            )
        except Exception:
            early_flush_futures.pop(target_key, None)

    def _mark_target_work_unit_done(work_unit: ResidualFieldWorkUnit | None) -> None:
        if work_unit is None:
            return
        target_key = _reducer_target_key(work_unit)
        if target_key not in target_remaining:
            return
        target_remaining[target_key] = int(target_remaining[target_key]) - 1
        if target_remaining[target_key] > 0:
            return
        target_remaining.pop(target_key, None)
        if reuse_rifft_payload:
            rifft_future = target_rifft_futures.pop(target_key, None)
            release = getattr(rifft_future, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        _submit_early_target_flush(target_key)

    def _submit(work_unit: ResidualFieldWorkUnit) -> None:
        nonlocal submitted
        target_key = _reducer_target_key(work_unit)
        owner_address = None
        if target_key in target_owners:
            owner_address = _resolve_owner_address(
                target_key=target_key,
                target_owners=target_owners,
                worker_addresses=_current_worker_addresses(client),
            )
        if owner_address is not None:
            previous_owner = target_last_owner.get(target_key)
            if previous_owner is not None and previous_owner != owner_address:
                target_owner_epochs[target_key] = (
                    int(target_owner_epochs.get(target_key, 0)) + 1
                )
                logger.warning(
                    "Residual-field owner epoch bump | target=%s | %s -> %s "
                    "| epoch=%d",
                    target_key,
                    previous_owner,
                    owner_address,
                    target_owner_epochs[target_key],
                )
            target_last_owner[target_key] = owner_address
        submit_kwargs = dict(
            total_reciprocal_points=total_reciprocal_points,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path if owner_local_reducer else None,
            scratch_root=scratch_root,
            reducer_backend=task_reducer_backend,
            total_expected_partials=total_partials_by_target[_reducer_target_key(work_unit)],
            owner_local_reducer=owner_local_reducer,
            quiet_logs=False,
            key=f"residual-{work_unit.artifact_key}",
            pure=False,
            resources=nufft_resources,
            # The driver owns retry (retries_left + infra budget + breaker).
            # Dask-level retries would run invisibly underneath it — in cpu
            # policy a deterministic failure executed up to (1+4)x(1+4)=25
            # times while fail_streak saw one failure per driver attempt.
            retries=0,
            runtime_provenance=runtime_provenance,
            nufft_eps=nufft_settings.eps,
            nufft_prefer_cpu=nufft_settings.prefer_cpu,
            nufft_gpu_only=nufft_settings.gpu_only,
            owner_epoch=int(target_owner_epochs.get(target_key, 0)),
        )
        if streaming_context_future is not None:
            submit_kwargs["streaming_compute_context"] = streaming_context_future
        if reuse_rifft_payload:
            submit_kwargs["rifft_payload"] = _target_rifft_payload_future(
                work_unit,
                owner_address=owner_address,
            )
        if owner_address is not None:
            submit_kwargs["workers"] = [owner_address]
            submit_kwargs["allow_other_workers"] = False
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            ()
            if streaming_context_future is not None
            else _interval_inputs_for_work_unit(
                work_unit,
                transient_interval_payloads=transient_interval_payloads,
            ),
            None if reuse_rifft_payload else chunk_futures[int(work_unit.chunk_id)],
            **submit_kwargs,
        )
        flying.add(future)
        future_meta[future] = work_unit
        future_pinned_owner[future] = owner_address
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

    def _future_completed_successfully(future, result_marker) -> bool:
        if result_marker is False:
            return False
        status = getattr(future, "status", None)
        if status is not None:
            return status == "finished" and result_marker is not None
        return result_marker is not None and result_marker is not False

    def _future_failure_detail(future, result_marker) -> str:
        exception = None
        exception_method = getattr(future, "exception", None)
        if callable(exception_method):
            try:
                # timeout=0 through distributed's sync bridge ALWAYS raises
                # a fabricated 'timed out after 0 s' from a non-loop thread,
                # masking the real KilledWorker/Cancelled and making every
                # failure classify as infrastructure. The state is already
                # resolved locally (as_completed delivered it), so a
                # positive timeout returns on the first loop tick.
                exception = exception_method(timeout=30)
            except TypeError:
                try:
                    exception = exception_method()
                except Exception as err:
                    exception = err
            except Exception as err:
                exception = err
        if exception is not None:
            return f"{type(exception).__name__}: {exception}"
        try:
            value = future.result()
        except Exception as err:
            return f"{type(err).__name__}: {err}"
        if value is None:
            return "task returned None"
        return f"task status={getattr(future, 'status', 'unknown')} result={result_marker!r}"

    def _format_failed_batch(work_unit: ResidualFieldWorkUnit, detail: str) -> str:
        partition = (
            "owner"
            if work_unit.partition_id is None
            else str(int(work_unit.partition_id))
        )
        return (
            f"chunk={int(work_unit.chunk_id)} "
            f"partition={partition} "
            f"intervals={_work_unit_interval_label(work_unit)} "
            f"reason={detail}"
        )

    def _release_finished_future(future) -> None:
        release = getattr(future, "release", None)
        if not callable(release):
            return
        try:
            release()
        except Exception:
            pass

    def _handle_completed_future(future, result_marker, bump, pbar=None) -> None:
        nonlocal completed, fail_streak
        flying.discard(future)
        future_pinned_owner.pop(future, None)
        work_unit = future_meta.pop(future, None)
        ok = _future_completed_successfully(future, result_marker)
        bump()
        completed += 1
        detail = "" if ok else _future_failure_detail(future, result_marker)
        if work_unit is not None and pbar is not None:
            _update_pbar_postfix(pbar, work_unit, ok=ok)
            if not ok:
                logger.warning(
                    "Residual-field batch FAILED | chunk=%d | partition=%s | intervals=%s | %s",
                    work_unit.chunk_id,
                    "owner" if work_unit.partition_id is None else work_unit.partition_id,
                    _work_unit_interval_label(work_unit),
                    detail,
                )
        if not ok and work_unit is not None:
            # Only GENUINE task failures count toward the GPU circuit
            # breaker. Infrastructure casualties — a nanny restarting a
            # worker over its memory budget, cancelled/lost futures, comm
            # drops — say nothing about GPU health, and counting them is
            # what turned every memory hiccup into a full CPU-only run
            # (measured: one 95%-budget restart cascaded into thousands of
            # cancelled batches, all "failures", breaker tripped, run dead).
            infrastructure_failure = any(
                marker in detail
                for marker in (
                    "KilledWorker",
                    "Cancelled",
                    "CommClosed",
                    "TimeoutError",
                    "WorkerProcessDied",
                    "Nanny",
                )
            )
            key = (str(work_unit.artifact_key), int(work_unit.chunk_id))
            _release_finished_future(future)
            if infrastructure_failure:
                if "KilledWorker" in detail:
                    killed = int(killed_worker_seen.get(key, 0)) + 1
                    killed_worker_seen[key] = killed
                    if killed >= killed_worker_retry_cap:
                        logger.error(
                            "Residual-field unit killed its worker %d times "
                            "| chunk=%d | partition=%s | intervals=%s — "
                            "poison task, failing it instead of burning "
                            "more workers.",
                            killed,
                            work_unit.chunk_id,
                            "owner"
                            if work_unit.partition_id is None
                            else work_unit.partition_id,
                            _work_unit_interval_label(work_unit),
                        )
                        exhausted_failures.append((work_unit, detail))
                        _mark_target_work_unit_done(work_unit)
                        return
                # Own budget + exponential backoff: resubmitting into the
                # middle of a worker-restart storm just dies again in
                # seconds and used to exhaust the genuine retry budget.
                used = int(infra_failures_seen.get(key, 0))
                infra_failures_seen[key] = used + 1
                if used < infra_retry_budget:
                    delay = min(60.0, 2.0 ** used)
                    deferred_resubmits.append(
                        (time.monotonic() + delay, work_unit)
                    )
                    logger.warning(
                        "Residual-field infra failure | chunk=%d | partition=%s "
                        "| resubmit deferred %.0fs (infra attempt %d/%d)",
                        work_unit.chunk_id,
                        "owner"
                        if work_unit.partition_id is None
                        else work_unit.partition_id,
                        delay,
                        used + 1,
                        infra_retry_budget,
                    )
                    return
                exhausted_failures.append((work_unit, detail))
                _mark_target_work_unit_done(work_unit)
                return
            fail_streak += 1
            if fail_streak >= fail_threshold:
                _trip_to_cpu_only()
            remaining = int(retries_left.get(key, 0))
            if remaining > 0:
                retries_left[key] = remaining - 1
                logger.warning(
                    "Retrying residual-field batch | chunk=%d | partition=%s | intervals=%s | remaining=%d",
                    work_unit.chunk_id,
                    "owner" if work_unit.partition_id is None else work_unit.partition_id,
                    _work_unit_interval_label(work_unit),
                    int(retries_left[key]),
                )
                _submit(work_unit)
                return
            exhausted_failures.append((work_unit, detail))
            _mark_target_work_unit_done(work_unit)
            return
        fail_streak = 0
        if work_unit is not None:
            try:
                _incorporate_completed_result(future, work_unit)
            finally:
                _release_finished_future(future)
                _mark_target_work_unit_done(work_unit)

    def _process_deferred_resubmits() -> None:
        if not deferred_resubmits:
            return
        now = time.monotonic()
        ready = [item for item in deferred_resubmits if item[0] <= now]
        for item in ready:
            deferred_resubmits.remove(item)
            _submit(item[1])

    def _ensure_streaming_context_replicas() -> None:
        """Re-scatter the streaming compute context when every worker that
        held a replica died — dependents of a lost scattered future fail
        with immediate comm timeouts, and a restarted worker never receives
        the broadcast."""
        nonlocal streaming_context_future
        if streaming_context is None or streaming_context_future is None:
            return
        try:
            key = getattr(streaming_context_future, "key", None)
            holders = client.who_has(streaming_context_future)
            held = holders.get(key) if isinstance(holders, dict) else None
            if held:
                return
        except Exception:
            return
        try:
            logger.warning(
                "Streaming context replicas lost with dead workers; re-scattering."
            )
            streaming_context_future = client.scatter(
                streaming_context, broadcast=True, hash=False
            )
        except Exception:
            logger.exception("Streaming context re-scatter failed.")

    def _rescue_futures_pinned_to_dead_workers() -> None:
        """A queued fold task pinned (allow_other_workers=False) to a dead
        worker parks in no-worker state FOREVER — nanny restarts come back
        on NEW addresses, so nothing ever schedules it and the drain hangs
        (observed: kernel OOM killed 2 of 4 workers; 30 'running' units
        never moved again). Cancel and resubmit through the owner remap.
        Infrastructure recovery, not a task failure: no retry budget spent,
        no circuit-breaker count."""
        live = set(_current_worker_addresses(client))
        if not live:
            return
        _ensure_streaming_context_replicas()
        for future in list(flying):
            pinned = future_pinned_owner.get(future)
            if pinned is None or pinned in live:
                continue
            if getattr(future, "status", "") in ("finished", "error", "cancelled"):
                continue  # completion path will handle it
            work_unit = future_meta.get(future)
            try:
                future.cancel()
            except Exception:
                pass
            flying.discard(future)
            future_pinned_owner.pop(future, None)
            future_meta.pop(future, None)
            if work_unit is None:
                continue
            target_key = _reducer_target_key(work_unit)
            # The cached rifft-payload future is pinned to the dead owner
            # too — drop it so _submit rebuilds it on the new owner.
            stale_rifft = target_rifft_futures.pop(target_key, None)
            release = getattr(stale_rifft, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
            logger.warning(
                "Residual-field owner %s is gone; resubmitting chunk=%d "
                "partition=%s via owner remap",
                pinned,
                int(work_unit.chunk_id),
                "owner" if work_unit.partition_id is None else work_unit.partition_id,
            )
            _submit(work_unit)

    def _drain_one_completion(bump, pbar=None, timeout_seconds: float = 45.0) -> bool:
        """Handle ONE completion, waiting in bounded slices so tasks pinned
        to since-dead workers get rescued instead of blocking as_completed
        forever. Returns False only when nothing is in flight (or the
        scheduler yields nothing for a completed set — anomaly)."""
        from distributed import wait as _distributed_wait

        while flying:
            _process_deferred_resubmits()
            _rescue_futures_pinned_to_dead_workers()
            if not flying:
                return False
            if not is_sync_client(client):
                if _current_worker_addresses(client):
                    dead_cluster_since[0] = None
                else:
                    now = time.monotonic()
                    if dead_cluster_since[0] is None:
                        dead_cluster_since[0] = now
                    elif (
                        now - dead_cluster_since[0]
                        >= _dead_cluster_horizon_seconds()
                    ):
                        raise RuntimeError(
                            "Residual-field drain: no live workers for "
                            f"{_dead_cluster_horizon_seconds():.0f}s with "
                            f"{len(flying)} batch(es) in flight."
                        )
            try:
                _distributed_wait(
                    list(flying),
                    timeout=timeout_seconds,
                    return_when="FIRST_COMPLETED",
                )
            except TimeoutError:
                continue
            except Exception:
                # Sync/test clients (or comm hiccups) can make wait() raise
                # immediately — fall through to the blocking generator path
                # rather than spinning on retry.
                pass
            for future, result in yield_futures_with_results(list(flying), client):
                _handle_completed_future(future, result, bump, pbar=pbar)
                return True
            return False
        return False

    def _harvest_finished_nonblocking(bump, pbar=None) -> None:
        done_now = [future for future in list(flying) if future.done()]
        for future in done_now:
            try:
                result_marker = future.result()
            except Exception:
                result_marker = False
            _handle_completed_future(future, result_marker, bump, pbar=pbar)

    def _apply_memory_backpressure(bump, pbar=None) -> None:
        nonlocal last_memory_pressure_check, last_memory_pressure_trim
        poll_seconds = _memory_backpressure_poll_seconds()
        if poll_seconds <= 0.0:
            return
        now = time.monotonic()
        if now - last_memory_pressure_check < poll_seconds:
            return
        last_memory_pressure_check = now
        if not _cluster_host_memory_pressure(client):
            return
        if now - last_memory_pressure_trim >= 5.0:
            _trim_workers_for_memory_pressure(client)
            last_memory_pressure_trim = now
            logger.warning(
                "Residual-field memory backpressure: worker RSS is near Dask pause threshold; "
                "trimming native pools and waiting for in-flight work before submitting more batches."
            )
        while flying and _cluster_host_memory_pressure(client):
            if not _drain_one_completion(bump, pbar=pbar):
                break

    def _update_pbar_postfix(pbar, work_unit, ok=True):
        elapsed = time.monotonic() - _residual_start_time
        timing = _format_elapsed_eta(elapsed, completed, total_tasks)
        partition = (
            "owner" if work_unit.partition_id is None else f"p{int(work_unit.partition_id)}"
        )
        status = "" if ok else " | FAILED"
        fails = f" | fails={len(exhausted_failures)}" if exhausted_failures else ""
        pbar.set_postfix_str(
            f"done=chunk{int(work_unit.chunk_id)}/{partition}"
            f":{_work_unit_interval_label(work_unit)}"
            f" | running={len(flying)} | {timing}{fails}{status}"
        )

    with logging_redirect_tqdm():
        with progress_bar(total_tasks, desc="Residual-field", unit="batch", force=True) as pbar:

            def bump() -> None:
                pbar.update(1)

            for work_unit in work_units:
                _submit(work_unit)
                _harvest_finished_nonblocking(bump, pbar=pbar)
                _apply_memory_backpressure(bump, pbar=pbar)
                while len(flying) >= max_inflight:
                    # Drain ONE completion then break so the outer
                    # submit-loop can enqueue the next work_unit immediately;
                    # bounded waits inside keep dead-owner rescue running.
                    if not _drain_one_completion(bump, pbar=pbar):
                        break

            while flying or deferred_resubmits:
                _process_deferred_resubmits()
                if not flying:
                    # Everything in flight is waiting out an infra backoff.
                    time.sleep(0.5)
                    continue
                if not _drain_one_completion(bump, pbar=pbar) and flying:
                    raise RuntimeError(
                        "Residual-field scheduler made no progress while draining "
                        f"{len(flying)} in-flight batch(es)."
                    )

    if exhausted_failures:
        formatted = "; ".join(
            _format_failed_batch(work_unit, detail)
            for work_unit, detail in exhausted_failures
        )
        raise RuntimeError(
            "Residual-field batch failed after retries before finalize: "
            f"{formatted}"
        )

    if owner_local_reducer and worker_addresses:
        _flush_local_reducer_targets_or_raise(
            client=client,
            template_backend=task_reducer_backend,
            target_keys=_unique_reducer_target_keys(work_units),
            parameter_digest=planned_work_units[0].parameter_digest,
            output_dir=artifacts.output_dir,
            db_path=artifacts.db_manager.db_path,
            target_owners=target_owners,
            pre_submitted_futures=early_flush_futures,
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
        finalize_futures_by_chunk: dict = {}
        finalize_owner_by_chunk_key: dict = {}
        finalize_live_workers = _current_worker_addresses(client)
        # Chunk-keyed finalize placement for streaming: the slot-keyed fold
        # map sends every chunk's finalize to ONE worker (all chunks share
        # the same content-addressed slot set) — 3-5 min serialized on one
        # worker while the rest idle. Finalize reads durable snapshots from
        # the shared output dir, so any worker can run any chunk.
        finalize_owner_by_chunk = (
            _streaming_finalize_owner_map(chunk_ids, finalize_live_workers)
            if streaming_context is not None and finalize_live_workers
            else {}
        )

        def _submit_finalize(finalize_chunk_id, finalize_worker):
            return client.submit(
                finalize_process_local_residual_chunk,
                task_reducer_backend,
                chunk_id=int(finalize_chunk_id),
                parameter_digest=planned_work_units[0].parameter_digest,
                output_dir=artifacts.output_dir,
                db_path=artifacts.db_manager.db_path,
                cleanup_policy=cleanup_policy,
                scratch_root=scratch_root,
                quiet_logs=False,
                expected_partitions=_expected_partition_family_for_chunk(
                    planned_work_units,
                    chunk_id=int(finalize_chunk_id),
                ),
                expected_interval_ids=expected_interval_ids_by_chunk.get(
                    int(finalize_chunk_id)
                ),
                # Streaming finalizes run on multiple workers concurrently;
                # SQLite marking moves to the driver (single writer).
                mark_intervals_saved=streaming_context is None,
                pure=False,
                workers=[finalize_worker],
                allow_other_workers=False,
            )

        for chunk_id in chunk_ids:
            finalizer_owner = None
            if streaming_context is not None:
                finalizer_owner = finalize_owner_by_chunk.get(int(chunk_id))
            elif any(int(work_unit.chunk_id) == int(chunk_id) for work_unit in owner_local_target_units):
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
            finalize_futures_by_chunk[int(chunk_id)] = _submit_finalize(
                int(chunk_id), finalizer_owner
            )
            finalize_owner_by_chunk_key[int(chunk_id)] = finalizer_owner
        # Finalize reads durable snapshots from the shared output dir, so a
        # dead finalizer's chunk is safely remapped to any live worker. This
        # is the highest-exposure barrier (finalize reads/writes tens of GB
        # per chunk — the phase most likely to OOM-kill a worker).
        for finalized_chunk, future, ok in _drain_owner_pinned_barrier(
            client=client,
            futures_by_key=finalize_futures_by_chunk,
            owner_by_key=finalize_owner_by_chunk_key,
            resubmit=_submit_finalize,
            barrier_name="finalize",
        ):
            if not ok:
                raise RuntimeError(
                    "Owner-local residual finalization failed for chunk "
                    f"{int(finalized_chunk)}."
                )
            if streaming_context is not None:
                _mark_finalized_chunk_intervals_saved(
                    db_path=artifacts.db_manager.db_path,
                    chunk_id=finalized_chunk,
                    interval_ids=expected_interval_ids_by_chunk.get(
                        finalized_chunk, ()
                    ),
                )
    else:
        # Sync-client path (no worker addresses). Owner-local reduction is
        # force-enabled for both backend kinds, tasks return status-only
        # results, and the attempt/candidate universe is never written —
        # quiescence scanning here was ceremony for a mode that cannot occur.
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
    _clear_worker_rifft_payload_caches(client)
    logger.info("Residual-field finished – %d tasks submitted", submitted)


__all__ = ["build_residual_field_work_units", "run_residual_field_stage"]
