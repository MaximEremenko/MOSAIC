from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any, Dict

import numpy as np

from core.residual_field.artifacts import write_stage2_replacement_expected_manifest
from core.residual_field.commit import (
    create_residual_commit_candidate,
    discover_residual_attempts,
    load_residual_attempt_payload,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.residual_field import commit as residual_commit
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.execution import _residual_work_unit_digest
from core.runtime import (
    DEFAULT_TASK_RETRIES,
    is_sync_client,
    logging_redirect_tqdm,
    progress_bar,
    require_chunk_quiescence,
    yield_futures_with_results,
)
from core.runtime.nufft_policy import nufft_task_retries
from core.scattering.kernels import point_list_to_recarray
from core.scattering.planning import ScatteringWorkIdentity
from core.scattering.runtime import (
    _add_nufft_task_kwargs,
    _call_accepts_kwarg,
    _current_worker_addresses,
    _interval_payload_input,
    _nufft_execution_settings,
    _require_scheduler_resource_capacity,
    _runtime_info,
)
from core.scattering.tasks import IntervalPayloadRef
from core.storage.agreement import DEFAULT_NUFFT_EPS
from core.storage.database_manager import DatabaseManager, create_db_manager_for_thread
from core.storage.digests import digest_dict

logger = logging.getLogger(__name__)


def _stage2_replacement_enabled(parameters: Dict[str, Any]) -> bool:
    runtime_info = _runtime_info(parameters)
    mode = runtime_info.get("scattering_stage2_mode")
    if mode is None:
        mode = runtime_info.get("stage2_mode")
    if mode is not None:
        return str(mode).strip().lower().replace("-", "_") == "replacement"
    enabled = runtime_info.get("scattering_stage2_replacement")
    if enabled is None:
        enabled = os.getenv("MOSAIC_SCATTERING_STAGE2_REPLACEMENT")
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"1", "true", "yes", "on", "replacement"}
    return bool(enabled)


def _stage2_pair_execution_enabled(parameters: Dict[str, Any]) -> bool:
    runtime_info = _runtime_info(parameters)
    mode = runtime_info.get("scattering_stage2_mode")
    if mode is None:
        mode = runtime_info.get("stage2_mode")
    if mode is not None:
        normalized = str(mode).strip().lower().replace("-", "_")
        return normalized in {
            "legacy",
            "pair",
            "pairs",
            "chunk",
            "chunks",
            "scattering",
            "scattering_pairs",
            "scattering_chunks",
        }
    enabled = runtime_info.get("scattering_stage2_pair_execution")
    if enabled is None:
        enabled = os.getenv("MOSAIC_SCATTERING_STAGE2_PAIR_EXECUTION")
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"1", "true", "yes", "on", "legacy"}
    return bool(enabled)


def _stage2_replacement_batch_size(parameters: Dict[str, Any]) -> int:
    runtime_info = _runtime_info(parameters)
    value = runtime_info.get(
        "scattering_stage2_batch_size",
        runtime_info.get("residual_shard_batch_size", 4),
    )
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError("scattering_stage2_batch_size must be positive.")
    return batch_size


def _stage2_replacement_max_inflight(parameters: Dict[str, Any]) -> int:
    runtime_info = _runtime_info(parameters)
    value = runtime_info.get("scattering_stage2_max_inflight", 5_000)
    max_inflight = int(value)
    if max_inflight <= 0:
        raise ValueError("scattering_stage2_max_inflight must be positive.")
    return max_inflight


def _build_stage2_replacement_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    parameter_digest: str,
    output_dir: str,
    max_intervals_per_shard: int,
) -> list[ResidualFieldWorkUnit]:
    grouped: dict[int, list[int]] = {}
    for interval_id, chunk_id in sorted(
        {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
    ):
        grouped.setdefault(int(chunk_id), []).append(int(interval_id))
    work_units: list[ResidualFieldWorkUnit] = []
    for chunk_id in sorted(grouped):
        interval_ids = grouped[chunk_id]
        for start in range(0, len(interval_ids), int(max_intervals_per_shard)):
            batch = tuple(interval_ids[start : start + int(max_intervals_per_shard)])
            if len(batch) == 1:
                work_units.append(
                    ResidualFieldWorkUnit.interval_chunk(
                        interval_id=int(batch[0]),
                        chunk_id=int(chunk_id),
                        parameter_digest=parameter_digest,
                        output_dir=output_dir,
                    )
                )
            else:
                work_units.append(
                    ResidualFieldWorkUnit.interval_chunk_batch(
                        interval_ids=batch,
                        chunk_id=int(chunk_id),
                        parameter_digest=parameter_digest,
                        output_dir=output_dir,
                    )
                )
    return work_units


def _identity_complete_stage2_replacement_work_units(
    *,
    work_units: list[ResidualFieldWorkUnit],
    point_data_list: list[dict],
    work_identity: ScatteringWorkIdentity,
    max_intervals_per_batch: int,
) -> list[ResidualFieldWorkUnit]:
    point_counts_by_chunk: dict[int, int] = {}
    for row in point_data_list:
        chunk_id = int(row["chunk_id"])
        point_counts_by_chunk[chunk_id] = point_counts_by_chunk.get(chunk_id, 0) + 1
    partition_plan_digest = digest_dict(
        {
            "source": "scattering-stage2-replacement",
            "run_digest": work_identity.run_digest,
            "point_counts_by_chunk": point_counts_by_chunk,
            "work_units": [
                {
                    "chunk_id": int(work_unit.chunk_id),
                    "interval_ids": [
                        int(interval_id)
                        for interval_id in (work_unit.interval_ids or (work_unit.interval_id,))
                        if interval_id is not None
                    ],
                }
                for work_unit in work_units
            ],
        },
        domain="mosaic.stage2_replacement.partition_plan.v1",
    )
    expected_output_digest = digest_dict(
        {
            "run_digest": work_identity.run_digest,
            "partition_plan_digest": partition_plan_digest,
            "max_intervals_per_batch": int(max_intervals_per_batch),
        },
        domain="mosaic.stage2_replacement.expected_output.v1",
    )
    source_scattering_commit_digest = _stage2_replacement_source_scattering_commit_digest(
        work_identity
    )
    completed: list[ResidualFieldWorkUnit] = []
    for work_unit in work_units:
        point_count = int(point_counts_by_chunk.get(int(work_unit.chunk_id), 0))
        if point_count <= 0:
            raise RuntimeError(
                f"Stage-2 replacement chunk {int(work_unit.chunk_id)} has no point rows."
            )
        with_identity = replace(
            work_unit,
            run_digest=work_identity.run_digest,
            partition_plan_digest=partition_plan_digest,
            source_scattering_commit_digest=source_scattering_commit_digest,
            source_replacement_digest=None,
            backend_policy_digest=work_identity.backend_policy_digest,
            expected_output_digest=expected_output_digest,
        )
        if with_identity.partition_id is None:
            with_identity = with_identity.with_partition(
                partition_id=0,
                point_start=0,
                point_stop=point_count,
            )
        completed.append(with_identity)
    return completed


def _stage2_replacement_source_scattering_commit_digest(
    work_identity: ScatteringWorkIdentity,
) -> str:
    return digest_dict(
        work_identity.to_work_unit_kwargs(),
        domain="mosaic.stage2_replacement.source_scattering_identity.v1",
    )


def _commit_stage2_replacement_attempts(
    *,
    output_dir: str,
    run_digest: str,
    planned_work_units: list[ResidualFieldWorkUnit],
    db_manager: DatabaseManager | None = None,
    db_path: str | None = None,
    db_dimension: int = 3,
    eps: float = DEFAULT_NUFFT_EPS,
) -> dict[int, tuple[int, ...]]:
    owns_db = False
    if db_manager is None:
        if db_path is None:
            raise ValueError("Stage-2 replacement commit needs db_manager or db_path.")
        db_manager = create_db_manager_for_thread(db_path, dimension=int(db_dimension))
        owns_db = True
    expected_by_chunk: dict[int, tuple[int, ...]] = {
        int(chunk_id): tuple(
            sorted(
                {
                    int(interval_id)
                    for work_unit in planned_work_units
                    if int(work_unit.chunk_id) == int(chunk_id)
                    for interval_id in (work_unit.interval_ids or (work_unit.interval_id,))
                    if interval_id is not None
                }
            )
        )
        for chunk_id in sorted({int(work_unit.chunk_id) for work_unit in planned_work_units})
    }
    expected_partition_ids_by_chunk: dict[int, tuple[int, ...]] = {
        int(chunk_id): tuple(
            sorted(
                {
                    int(work_unit.partition_id)
                    for work_unit in planned_work_units
                    if int(work_unit.chunk_id) == int(chunk_id) and work_unit.partition_id is not None
                }
            )
        )
        for chunk_id in expected_by_chunk
    }
    write_residual_stage_plan(
        output_dir=output_dir,
        run_digest=run_digest,
        expected_by_chunk=expected_partition_ids_by_chunk,
    )
    for chunk_id, partition_ids in expected_partition_ids_by_chunk.items():
        require_chunk_quiescence(
            (),
            client=None,
            output_dir=output_dir,
            run_digest=run_digest,
            stage="residual_field",
            chunk_id=int(chunk_id),
            expected_work_unit_digests=tuple(
                sorted(
                    _residual_work_unit_digest(work_unit)
                    for work_unit in planned_work_units
                    if int(work_unit.chunk_id) == int(chunk_id)
                )
            ),
        )
        attempts = discover_residual_attempts(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=int(chunk_id),
        )
        by_partition: dict[int, list] = {}
        for attempt in attempts:
            by_partition.setdefault(int(attempt.partition_id), []).append(attempt)
        expected_partitions: dict[int, tuple[int, ...]] = {}
        expected_reciprocal_count: int | None = None
        for partition_id in partition_ids:
            candidates = by_partition.get(int(partition_id), [])
            if not candidates:
                raise RuntimeError(
                    f"Missing stage-2 replacement attempt for chunk {chunk_id} partition {partition_id}."
                )
            # do NOT enforce bitwise payload_sha256 equality here. Same-work
            # attempts (a non-deterministic GPU relaunch, or a CPU vs GPU attempt for
            # the same partition science) may differ in bytes; their NUMERICAL
            # agreement is reconciled by the predicted-tolerance gate inside
            # create_residual_commit_candidate (_select_attempts_by_partition ->
            # _assert_residual_partials_agree), which fails closed on a real
            # divergence. point_ids and reciprocal_point_count are STRUCTURAL
            # (input-addressed) and identical across valid attempts, so any
            # representative is safe to read them from.
            selected = sorted(candidates, key=lambda item: (item.attempt_id, item.payload_path))[0]
            datasets, _attrs = load_residual_attempt_payload(selected, output_dir=output_dir)
            expected_partitions[int(partition_id)] = tuple(
                int(item) for item in np.asarray(datasets["point_ids"]).reshape(-1)
            )
            if expected_reciprocal_count is None:
                expected_reciprocal_count = int(selected.contribution_reciprocal_points)
        create_residual_commit_candidate(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=int(chunk_id),
            expected_partitions=expected_partitions,
            expected_reciprocal_point_count=expected_reciprocal_count,
            eps=eps,
        )
        getattr(residual_commit, "promote_residual" "_chunk_commit_by_scan")(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=int(chunk_id),
        )
        for interval_id in expected_by_chunk[int(chunk_id)]:
            db_manager.update_interval_chunk_status(
                int(interval_id),
                int(chunk_id),
                saved=True,
            )
    try:
        write_residual_stage_commit(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_ids=tuple(sorted(expected_partition_ids_by_chunk)),
        )
        return expected_by_chunk
    finally:
        if owns_db:
            db_manager.close()


def _stage2_replacement_interval_inputs(work_unit: ResidualFieldWorkUnit) -> tuple[IntervalPayloadRef | str, ...]:
    interval_refs = tuple(
        _interval_payload_input(None, artifact.path)
        for artifact in work_unit.source_artifacts
        if artifact.kind == "interval-precompute" and artifact.path is not None
    )
    if not interval_refs:
        raise ValueError("Stage-2 replacement work unit is missing interval artifacts.")
    return interval_refs


def _residual_future_ok(future, result_marker) -> bool:
    if result_marker is False:
        return False
    status = getattr(future, "status", None)
    if status is not None:
        return status == "finished" and result_marker is not None
    return result_marker is not None and result_marker is not False


def _residual_future_failure_detail(future, result_marker) -> str:
    exception_method = getattr(future, "exception", None)
    if callable(exception_method):
        try:
            exc = exception_method(timeout=0)
        except TypeError:
            exc = exception_method()
        except Exception as err:
            exc = err
        if exc is not None:
            return f"{type(exc).__name__}: {exc}"
    try:
        value = future.result()
    except Exception as err:
        return f"{type(err).__name__}: {err}"
    return f"task returned {value!r} (result marker {result_marker!r})"


def run_stage2_replacement_execution(
    *,
    unsaved_interval_chunks: list[tuple[int, int]],
    total_reciprocal_points: int,
    point_data_list: list[dict],
    db_manager: DatabaseManager,
    client: "Client | None",
    output_dir: str,
    parameter_digest: str,
    work_identity: ScatteringWorkIdentity,
    max_intervals_per_shard: int,
    max_inflight: int,
    runtime_provenance: dict[str, Any] | None = None,
    nufft_resources: dict[str, int] | None = None,
    nufft_settings=None,
) -> dict[int, tuple[int, ...]]:
    from core.residual_field.tasks import run_residual_field_interval_chunk_task

    work_units = _build_stage2_replacement_work_units(
        unsaved_interval_chunks,
        parameter_digest=parameter_digest,
        output_dir=output_dir,
        max_intervals_per_shard=max_intervals_per_shard,
    )
    work_units = _identity_complete_stage2_replacement_work_units(
        work_units=work_units,
        point_data_list=point_data_list,
        work_identity=work_identity,
        max_intervals_per_batch=max_intervals_per_shard,
    )
    source_scattering_commit_digest = _stage2_replacement_source_scattering_commit_digest(
        work_identity
    )
    if nufft_settings is None:
        nufft_settings = _nufft_execution_settings(
            {"runtime_info": {"nufft_policy": "cpu-only"}}
        )
    if not work_units:
        write_stage2_replacement_expected_manifest(
            output_dir=output_dir,
            parameter_digest=parameter_digest,
            expected_by_chunk={},
            run_digest=work_identity.run_digest,
            source_scattering_commit_digest=source_scattering_commit_digest,
        )
        logger.info("Stage-2 replacement skipped – no unsaved interval/chunk pairs.")
        return {}

    expected_by_chunk: dict[int, tuple[int, ...]] = {
        int(chunk_id): tuple(
            sorted(
                {
                    int(interval_id)
                    for work_unit in work_units
                    if int(work_unit.chunk_id) == int(chunk_id)
                    for interval_id in (work_unit.interval_ids or (work_unit.interval_id,))
                    if interval_id is not None
                }
            )
        )
        for chunk_id in sorted({int(work_unit.chunk_id) for work_unit in work_units})
    }
    write_stage2_replacement_expected_manifest(
        output_dir=output_dir,
        parameter_digest=parameter_digest,
        expected_by_chunk=expected_by_chunk,
        run_digest=work_identity.run_digest,
        source_scattering_commit_digest=source_scattering_commit_digest,
    )
    task_resources = dict(nufft_resources or {"nufft": 1})
    rec = point_list_to_recarray(point_data_list)

    if client is None or is_sync_client(client):
        failures: list[tuple[ResidualFieldWorkUnit, str]] = []
        with progress_bar(
            len(work_units),
            desc="Stage 2 replacement",
            unit="batch",
        ) as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                try:
                    task_kwargs = dict(
                        total_reciprocal_points=total_reciprocal_points,
                        output_dir=output_dir,
                        db_path=None,
                        scratch_root=None,
                        reducer_backend=None,
                        owner_local_reducer=False,
                        quiet_logs=False,
                    )
                    if _call_accepts_kwarg(
                        run_residual_field_interval_chunk_task,
                        "runtime_provenance",
                    ):
                        task_kwargs["runtime_provenance"] = runtime_provenance
                    _add_nufft_task_kwargs(
                        run_residual_field_interval_chunk_task,
                        task_kwargs,
                        nufft_settings,
                    )
                    run_residual_field_interval_chunk_task(
                        work_unit,
                        _stage2_replacement_interval_inputs(work_unit),
                        atoms,
                        **task_kwargs,
                    )
                except Exception as exc:
                    failures.append((work_unit, f"{type(exc).__name__}: {exc}"))
                pbar.update(1)
                pbar.refresh()
        if failures:
            formatted = "; ".join(
                f"chunk={work_unit.chunk_id} intervals={list(work_unit.interval_ids)} reason={detail}"
                for work_unit, detail in failures
            )
            raise RuntimeError(f"Stage-2 replacement map failed before reduce: {formatted}")
        expected_by_chunk = _commit_stage2_replacement_attempts(
            output_dir=output_dir,
            run_digest=work_identity.run_digest,
            db_manager=db_manager,
            planned_work_units=work_units,
            eps=float(nufft_settings.eps),
        )
        logger.info("Stage-2 replacement finished (sync).")
        return expected_by_chunk

    _require_scheduler_resource_capacity(client, "nufft")
    worker_addresses = _current_worker_addresses(client)
    chunk_ids = sorted(expected_by_chunk)
    chunk_owners = (
        {
            int(chunk_id): worker_addresses[index % len(worker_addresses)]
            for index, chunk_id in enumerate(chunk_ids)
        }
        if worker_addresses
        else {}
    )
    chunk_futures = {}
    for chunk_id in chunk_ids:
        scatter_kwargs = dict(broadcast=False, hash=False)
        if int(chunk_id) in chunk_owners:
            scatter_kwargs["workers"] = [chunk_owners[int(chunk_id)]]
        chunk_futures[int(chunk_id)] = client.scatter(
            rec[rec.chunk_id == int(chunk_id)],
            **scatter_kwargs,
        )
    flying: set = set()
    future_meta: dict = {}
    futures_by_chunk: dict[int, list] = {int(chunk_id): [] for chunk_id in expected_by_chunk}
    failures: list[tuple[ResidualFieldWorkUnit, str]] = []

    def _submit(work_unit: ResidualFieldWorkUnit) -> None:
        submit_kwargs = dict(
            total_reciprocal_points=total_reciprocal_points,
            output_dir=output_dir,
            db_path=None,
            scratch_root=None,
            reducer_backend=None,
            owner_local_reducer=False,
            quiet_logs=True,
            key=f"stage2-replacement-{work_unit.artifact_key}",
            pure=False,
            resources=task_resources,
            retries=nufft_task_retries(nufft_settings.execution_policy, DEFAULT_TASK_RETRIES),
        )
        if _call_accepts_kwarg(
            run_residual_field_interval_chunk_task,
            "runtime_provenance",
        ):
            submit_kwargs["runtime_provenance"] = runtime_provenance
        _add_nufft_task_kwargs(
            run_residual_field_interval_chunk_task,
            submit_kwargs,
            nufft_settings,
        )
        owner = chunk_owners.get(int(work_unit.chunk_id))
        if owner is not None:
            submit_kwargs["workers"] = [owner]
            submit_kwargs["allow_other_workers"] = False
        future = client.submit(
            run_residual_field_interval_chunk_task,
            work_unit,
            _stage2_replacement_interval_inputs(work_unit),
            chunk_futures[int(work_unit.chunk_id)],
            **submit_kwargs,
        )
        flying.add(future)
        future_meta[future] = work_unit
        futures_by_chunk.setdefault(int(work_unit.chunk_id), []).append(future)

    def _drain_one(bump) -> bool:
        for future, result in yield_futures_with_results(list(flying), client):
            flying.discard(future)
            work_unit = future_meta.pop(future, None)
            bump()
            if not _residual_future_ok(future, result) and work_unit is not None:
                failures.append((work_unit, _residual_future_failure_detail(future, result)))
            return True
        return False

    with logging_redirect_tqdm():
        with progress_bar(
            len(work_units),
            desc="Stage 2 replacement",
            unit="batch",
        ) as pbar:

            def bump() -> None:
                pbar.update(1)
                pbar.refresh()

            for work_unit in work_units:
                _submit(work_unit)
                while len(flying) >= int(max_inflight):
                    if not _drain_one(bump):
                        break
            while flying:
                if not _drain_one(bump):
                    raise RuntimeError(
                        "Stage-2 replacement scheduler made no progress while draining "
                        f"{len(flying)} in-flight batch(es)."
                    )

    if failures:
        formatted = "; ".join(
            f"chunk={work_unit.chunk_id} intervals={list(work_unit.interval_ids)} reason={detail}"
            for work_unit, detail in failures
        )
        raise RuntimeError(f"Stage-2 replacement map failed after Dask retries: {formatted}")

    reducer_futures = []
    future_chunk: dict = {}
    for chunk_id in expected_by_chunk:
        reducer_kwargs = dict(
            output_dir=output_dir,
            run_digest=work_identity.run_digest,
            db_path=db_manager.db_path,
            planned_work_units=[
                work_unit
                for work_unit in work_units
                if int(work_unit.chunk_id) == int(chunk_id)
            ],
            eps=float(nufft_settings.eps),
            pure=False,
            retries=DEFAULT_TASK_RETRIES,
        )
        owner = chunk_owners.get(int(chunk_id))
        if owner is not None:
            reducer_kwargs["workers"] = [owner]
            reducer_kwargs["allow_other_workers"] = False
        future = client.submit(_commit_stage2_replacement_attempts, **reducer_kwargs)
        reducer_futures.append(future)
        future_chunk[future] = int(chunk_id)
    reducer_failures: list[str] = []
    for future, result in yield_futures_with_results(reducer_futures, client):
        if not _residual_future_ok(future, result):
            reducer_failures.append(
                f"chunk={future_chunk.get(future)} "
                f"reason={_residual_future_failure_detail(future, result)}"
            )
    if reducer_failures:
        raise RuntimeError(
            "Stage-2 replacement reduce failed before committed residual outputs: "
            + "; ".join(reducer_failures)
        )

    logger.info(
        "Stage-2 replacement finished via residual attempt reducer – %d batch tasks",
        len(work_units),
    )
    return expected_by_chunk


__all__ = [
    "run_stage2_replacement_execution",
    "_build_stage2_replacement_work_units",
    "_commit_stage2_replacement_attempts",
    "_identity_complete_stage2_replacement_work_units",
    "_stage2_replacement_batch_size",
    "_stage2_replacement_enabled",
    "_stage2_replacement_interval_inputs",
    "_stage2_replacement_max_inflight",
    "_stage2_replacement_source_scattering_commit_digest",
]
