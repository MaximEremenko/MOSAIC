from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

from core.residual_field.backend import (
    ScatteringIntervalArtifactPolicy,
    build_residual_field_reducer_backend,
    is_same_node_local_client,
    resolve_residual_field_reducer_backend_kind,
)
from core.scattering.artifacts import (
    is_interval_artifact_committed,
    persist_precomputed_interval_artifact,
)
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import (
    IntervalTask,
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
    compute_scattering_interval_payload,
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


def _runtime_info(parameters: Dict[str, Any]) -> dict[str, Any]:
    runtime_info = parameters.get("runtime_info") or {}
    return runtime_info if isinstance(runtime_info, dict) else {}


def _save_interval_outputs_requested(
    *,
    parameters: Dict[str, Any],
    default: bool,
) -> bool:
    runtime_info = _runtime_info(parameters)
    override = runtime_info.get("save_scattering_interval_artifacts")
    if override is None:
        override = runtime_info.get("save_interval_artifacts")
    if override is None:
        env = os.getenv("MOSAIC_SAVE_SCATTERING_INTERVAL_ARTIFACTS")
        if env is not None:
            override = env == "1"
    return default if override is None else bool(override)


def _resolve_scattering_interval_artifact_policy(
    *,
    parameters: Dict[str, Any],
    client,
) -> ScatteringIntervalArtifactPolicy:
    runtime_info = _runtime_info(parameters)
    backend_kind = resolve_residual_field_reducer_backend_kind(
        runtime_info=runtime_info,
        client=client,
    )
    backend = build_residual_field_reducer_backend(backend_kind)
    default_policy = backend.layout.checkpoint_policy.interval_artifacts

    raw_policy = runtime_info.get("scattering_interval_artifact_policy")
    if raw_policy is None:
        raw_policy = runtime_info.get("interval_artifact_policy")
    if raw_policy is None:
        raw_policy = os.getenv("MOSAIC_SCATTERING_INTERVAL_ARTIFACT_POLICY")
    if raw_policy is None:
        requested_policy = (
            "required_transport"
            if _save_interval_outputs_requested(
                parameters=parameters,
                default=backend.persist_interval_artifacts_by_default(),
            )
            else "optional_output"
        )
    else:
        requested_policy = _normalize_scattering_interval_artifact_policy(raw_policy)

    if backend.interval_artifacts_required_for_transport():
        return "required_transport"
    if requested_policy == "required_transport":
        return "required_transport"
    return default_policy


def _normalize_scattering_interval_artifact_policy(
    value: object,
) -> ScatteringIntervalArtifactPolicy:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"required", "required_transport", "transport_required"}:
        return "required_transport"
    if normalized in {"optional", "optional_output", "inspection_only", "saved_output"}:
        return "optional_output"
    raise ValueError(
        "Scattering interval artifact policy must be 'required_transport' or "
        "'optional_output'."
    )


def _scatter_shared_precompute_inputs(
    client,
    *,
    B_: np.ndarray,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    coeff_val,
):
    payloads = {
        "B_": client.scatter(B_, broadcast=True, hash=False),
        "supercell": client.scatter(supercell, broadcast=True, hash=False),
        "original_coords": client.scatter(original_coords, broadcast=True, hash=False),
        "cells_origin": client.scatter(cells_origin, broadcast=True, hash=False),
        "elements_arr": client.scatter(elements_arr, broadcast=True, hash=False),
    }
    if coeff_val is not None:
        payloads["coeff_val"] = client.scatter(coeff_val, broadcast=True, hash=False)
    else:
        payloads["coeff_val"] = None
    return payloads


def _local_fast_handoff_enabled(
    *,
    parameters: Dict[str, Any],
    client,
) -> bool:
    if "runtime_info" not in parameters:
        return False
    backend_kind = resolve_residual_field_reducer_backend_kind(
        runtime_info=_runtime_info(parameters),
        client=client,
    )
    return backend_kind == "local_restartable" and is_same_node_local_client(client)


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
    transient_interval_payloads: dict[int, IntervalTask] | None = None,
) -> list[Path]:
    payload_cache = transient_interval_payloads if transient_interval_payloads is not None else {}
    local_fast_handoff = _local_fast_handoff_enabled(parameters=parameters, client=client)
    interval_artifact_policy = _resolve_scattering_interval_artifact_policy(
        parameters=parameters,
        client=client,
    )
    persist_interval_artifacts = interval_artifact_policy == "required_transport"
    pending = [
        work_unit
        for work_unit in work_units
        if (
            int(work_unit.interval_id) not in payload_cache
            and not (
                persist_interval_artifacts
                and is_interval_artifact_committed(work_unit, db_path=db.db_path)
            )
        )
    ]
    cached = [
        Path(work_unit.interval_artifact.path)
        for work_unit in work_units
        if work_unit.interval_artifact is not None
        and work_unit.interval_artifact.path is not None
        and persist_interval_artifacts
        and is_interval_artifact_committed(work_unit, db_path=db.db_path)
    ]
    cached_payloads = [
        int(work_unit.interval_id)
        for work_unit in work_units
        if int(work_unit.interval_id) in payload_cache
    ]

    if not pending:
        logger.info(
            "Stage-1 complete: %d written, %d cached, %d cached-in-memory, %d skipped | transport=%s | interval_policy=%s",
            0,
            len(cached),
            len(cached_payloads),
            0,
            "direct-handoff" if local_fast_handoff else "durable-interval-artifacts",
            interval_artifact_policy,
        )
        return cached

    written_files: list[Path] = []
    produced_payloads = 0
    if local_fast_handoff and (client is not None) and (not is_sync_client(client)):
        shared_inputs = _scatter_shared_precompute_inputs(
            client,
            B_=B_,
            supercell=supercell,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            coeff_val=parameters.get("coeff"),
        )
        futures = [
            client.submit(
                compute_scattering_interval_payload,
                interval_lookup[work_unit.interval_id],
                B_=shared_inputs["B_"],
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=shared_inputs["supercell"],
                original_coords=shared_inputs["original_coords"],
                cells_origin=shared_inputs["cells_origin"],
                elements_arr=shared_inputs["elements_arr"],
                charge=charge,
                use_coeff=("coeff" in parameters),
                coeff_val=shared_inputs["coeff_val"],
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                pure=False,
                resources={"nufft": 1},
            )
            for work_unit in pending
        ]
        future_meta = {future: work_unit for future, work_unit in zip(futures, pending)}
        with logging_redirect_tqdm():
            with progress_bar(len(futures), desc="Precompute intervals", unit="intervals") as pbar:
                for future, _ in yield_futures_with_results(futures, client):
                    work_unit = future_meta[future]
                    try:
                        interval_task = future.result()
                    except Exception:
                        interval_task = None
                    if interval_task is not None:
                        payload_cache[int(work_unit.interval_id)] = interval_task
                        produced_payloads += 1
                        if persist_interval_artifacts:
                            manifest = persist_precomputed_interval_artifact(
                                work_unit,
                                interval_task,
                                db_path=db.db_path,
                            )
                            if manifest is not None and manifest.artifacts:
                                artifact_path = manifest.artifacts[0].path
                                if artifact_path is not None:
                                    written_files.append(Path(artifact_path))
                    pbar.update(1)
                    pbar.refresh()
        logger.info(
            "Stage-1 complete: %d written, %d cached, %d cached-in-memory, %d skipped | transport=%s | interval_policy=%s",
            len(written_files),
            len(cached),
            len(cached_payloads) + produced_payloads,
            len(pending) - produced_payloads,
            "direct-handoff",
            interval_artifact_policy,
        )
        return cached + written_files

    if local_fast_handoff:
        with progress_bar(len(pending), desc="Precompute intervals", unit="intervals") as pbar:
            for work_unit in pending:
                interval_task = compute_scattering_interval_payload(
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
                )
                if interval_task is not None:
                    payload_cache[int(work_unit.interval_id)] = interval_task
                    produced_payloads += 1
                    if persist_interval_artifacts:
                        manifest = persist_precomputed_interval_artifact(
                            work_unit,
                            interval_task,
                            db_path=db.db_path,
                        )
                        if manifest is not None and manifest.artifacts:
                            artifact_path = manifest.artifacts[0].path
                            if artifact_path is not None:
                                written_files.append(Path(artifact_path))
                pbar.update(1)
                pbar.refresh()

        logger.info(
            "Stage-1 complete: %d written, %d cached, %d cached-in-memory, %d skipped | transport=%s | interval_policy=%s",
            len(written_files),
            len(cached),
            len(cached_payloads) + produced_payloads,
            len(pending) - produced_payloads,
            "direct-handoff",
            interval_artifact_policy,
        )
        return cached + written_files

    if (client is not None) and (not is_sync_client(client)):
        shared_inputs = _scatter_shared_precompute_inputs(
            client,
            B_=B_,
            supercell=supercell,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            coeff_val=parameters.get("coeff"),
        )
        futures = [
            client.submit(
                run_scattering_interval_task,
                work_unit,
                interval_lookup[work_unit.interval_id],
                B_=shared_inputs["B_"],
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=shared_inputs["supercell"],
                original_coords=shared_inputs["original_coords"],
                cells_origin=shared_inputs["cells_origin"],
                elements_arr=shared_inputs["elements_arr"],
                charge=charge,
                use_coeff=("coeff" in parameters),
                coeff_val=shared_inputs["coeff_val"],
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
            "Stage-1 complete: %d written, %d cached, %d cached-in-memory, %d skipped | transport=%s | interval_policy=%s",
            len(written_files),
            len(cached),
            len(cached_payloads),
            len(pending) - len(written_files),
            "durable-interval-artifacts",
            interval_artifact_policy,
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
        "Stage-1 complete: %d written, %d cached, %d cached-in-memory, %d skipped | transport=%s | interval_policy=%s",
        len(written_files),
        len(cached),
        len(cached_payloads),
        len(pending) - len(written_files),
        "durable-interval-artifacts",
        interval_artifact_policy,
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
            transient_interval_payloads=parameters.get("transient_interval_payloads"),
        )
    logger.info("Completed scattering interval precompute stage")


__all__ = [
    "run_interval_chunk_execution",
    "run_interval_precompute",
    "run_scattering_stage",
]
