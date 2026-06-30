from __future__ import annotations

import logging
import os
import inspect
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

from core.residual_field.backend import (
    ScatteringIntervalArtifactPolicy,
    build_residual_field_reducer_backend,
    is_same_node_local_client,
    resolve_residual_field_reducer_backend_kind,
)
from core.residual_field import commit as residual_commit
from core.residual_field.commit import (
    create_residual_commit_candidate,
    discover_residual_attempts,
    load_residual_attempt_payload,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.residual_field.execution import _residual_work_unit_digest
from core.residual_field.artifacts import write_stage2_replacement_expected_manifest
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.scattering.artifacts import (
    is_interval_artifact_committed,
    mark_empty_interval_precomputed,
    persist_precomputed_interval_artifact,
)
from core.scattering.commit import (
    ScatteringChunkCommitManifest,
    build_scattering_work_unit_digest,
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    promote_scattering_chunk_commit_by_scan,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)
from core.storage.agreement import DEFAULT_NUFFT_EPS
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import (
    IntervalTask,
    point_list_to_recarray,
    reciprocal_space_points_counter,
    to_interval_dict,
)
from core.scattering.planning import (
    ScatteringWorkIdentity,
    build_scattering_interval_chunk_work_units,
    build_scattering_execution_plan,
    build_scattering_interval_lookup,
    chunk_ids_for_work_units,
    interval_paths_for_work_units,
    prepare_scattering_run_identity,
)
from core.runtime import (
    DEFAULT_TASK_RETRIES,
    is_sync_client,
    logging_redirect_tqdm,
    nufft_task_resources,
    profile_output_filesystem,
    progress_bar,
    quiet_loggers,
    require_chunk_quiescence,
    require_gpu_admission,
    register_cleanup_plugin,
    runtime_provenance_for_attempt,
    resolve_nufft_execution_settings,
    yield_futures_with_results,
)
from core.scattering.tasks import (
    IntervalPayloadRef,
    compute_scattering_interval_payload,
    run_scattering_interval_chunk_task,
    run_scattering_interval_task,
)
from core.storage.database_manager import DatabaseManager, create_db_manager_for_thread
from core.storage.digests import digest_dict
from core.storage.fingerprint import file_sha256
from core.storage.run_state_cache import (
    pending_scattering_interval_chunks,
    rebuild_sqlite_cache_from_manifests,
    scan_run_state,
)
from core.runtime.nufft_policy import nufft_task_retries

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)
_LOCAL_DIRECT_HANDOFF_MAX_INTERVALS_DEFAULT = 64
_LOCAL_DIRECT_HANDOFF_MAX_BYTES_DEFAULT = 256 << 20


def _chunk_task_key(work_unit: ScatteringWorkUnit) -> str:
    if work_unit.chunk_id is None:
        raise ValueError("Chunk task key requires a chunk-scoped work unit.")
    return f"proc-{work_unit.interval_id}-{work_unit.chunk_id}"


def _current_worker_addresses(client) -> list[str]:
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return []
    return sorted(str(address) for address in workers)


def _require_scheduler_resource_capacity(client, resource_name: str) -> None:
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return
    if not workers:
        return
    total = 0.0
    for worker in workers.values():
        resources = worker.get("resources", {}) or {}
        try:
            total += float(resources.get(resource_name, 0.0))
        except (TypeError, ValueError):
            pass
    if total <= 0:
        raise RuntimeError(
            f"Dask scheduler reports zero total {resource_name!r} resource capacity; "
            "MOSAIC NUFFT tasks would stay queued or overbook. Configure worker "
            f"resources with {resource_name}=N."
        )


def _runtime_info(parameters: Dict[str, Any]) -> dict[str, Any]:
    runtime_info = parameters.get("runtime_info") or {}
    return runtime_info if isinstance(runtime_info, dict) else {}


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
    if _stage2_replacement_enabled(parameters):
        return "required_transport"

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


def _scheduler_kind(client) -> str:
    if client is None or is_sync_client(client):
        return "sync"
    return "dask"


def _nufft_execution_policy(parameters: Dict[str, Any]) -> str:
    return _nufft_execution_settings(parameters).execution_policy


def _nufft_execution_settings(parameters: Dict[str, Any]):
    runtime_info = _runtime_info(parameters)
    requested = runtime_info.get("scattering_nufft_policy")
    if requested is None:
        requested = runtime_info.get("nufft_execution_policy")
    if requested is None:
        requested = runtime_info.get("nufft_policy")
    eps = runtime_info.get("scattering_nufft_eps", runtime_info.get("nufft_eps", 1e-12))
    dtype = runtime_info.get("scattering_dtype", runtime_info.get("nufft_dtype", "complex128"))
    return resolve_nufft_execution_settings(requested, eps=eps, dtype=dtype)


def _nufft_resources_for_parameters(parameters: Dict[str, Any]) -> dict[str, int]:
    return nufft_task_resources(_nufft_execution_policy(parameters))


def _runtime_provenance_for_scattering(
    *,
    parameters: Dict[str, Any],
    fs_capability_digest: str | None,
    client,
) -> dict[str, Any]:
    settings = _nufft_execution_settings(parameters)
    provenance = runtime_provenance_for_attempt(
        fs_capability_digest=fs_capability_digest,
        scheduler_kind=_scheduler_kind(client),
        nufft_policy=settings.execution_policy,
        resource_requirements=_nufft_resources_for_parameters(parameters),
        cuda_probe=settings.gpu_only,
    )
    provenance["nufft_execution_settings"] = settings.identity_payload()
    return provenance


def _call_accepts_kwarg(func, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _add_nufft_task_kwargs(func, kwargs: dict[str, Any], nufft_settings) -> None:
    for name, value in (
        ("nufft_eps", nufft_settings.eps),
        ("nufft_prefer_cpu", nufft_settings.prefer_cpu),
        ("nufft_gpu_only", nufft_settings.gpu_only),
    ):
        if _call_accepts_kwarg(func, name):
            kwargs[name] = value


def _scattering_work_unit_digest(work_unit: ScatteringWorkUnit) -> str:
    # P11 C-2: device-independent checkpoint address (execution/backend policy are
    # metadata, not identity) so CPU and GPU work units share one address.
    return build_scattering_work_unit_digest(
        interval_id=int(work_unit.interval_id),
        chunk_id=int(work_unit.chunk_id),
        scientific_digest=str(work_unit.scientific_digest),
        qspace_plan_digest=str(work_unit.qspace_plan_digest),
        source_structure_digest=str(work_unit.source_structure_digest),
    )



def _current_scattering_identity(
    *,
    parameters: Dict[str, Any],
    output_dir: str,
    B_: np.ndarray,
    mask_params: Dict[str, Any],
    MaskStrategy,
    client,
) -> ScatteringWorkIdentity:
    runtime_info = _runtime_info(parameters)
    nufft_settings = _nufft_execution_settings(parameters)
    interval_artifact_policy = _resolve_scattering_interval_artifact_policy(
        parameters=parameters,
        client=client,
    )
    return prepare_scattering_run_identity(
        parameters=parameters,
        output_dir=output_dir,
        B_=B_,
        mask_params=mask_params,
        MaskStrategy=MaskStrategy,
        backend=nufft_settings.backend,
        eps=nufft_settings.eps,
        dtype=nufft_settings.dtype,
        pre_sum_mode=str(runtime_info.get("scattering_pre_sum_mode", "off")),
        reducer_strategy=(
            "stage2-replacement"
            if _stage2_replacement_enabled(parameters)
            else "attempt-commit"
        ),
        scheduler_kind=_scheduler_kind(client),
        interval_artifact_policy=str(interval_artifact_policy),
        deterministic_mode=nufft_settings.deterministic_mode,
        thread_count=nufft_settings.thread_count,
        requested_nufft_policy=nufft_settings.requested_policy,
        execution_nufft_policy=nufft_settings.execution_policy,
    )


def _require_stage2_work_identity(
    work_units: list[ScatteringWorkUnit],
) -> ScatteringWorkIdentity:
    identity_values: set[tuple[str, str, str, str, str, str]] = set()
    for work_unit in work_units:
        values = (
            work_unit.scientific_digest,
            work_unit.execution_digest,
            work_unit.run_digest,
            work_unit.qspace_plan_digest,
            work_unit.backend_policy_digest,
            work_unit.source_structure_digest,
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(
                "Current-run scattering stage-2 work units must include complete "
                "scientific, execution, run, qspace-plan, backend-policy, and "
                "source-structure identity."
            )
        identity_values.add(values)  # type: ignore[arg-type]
    if len(identity_values) != 1:
        raise ValueError("Scattering stage-2 work units must share one run identity.")
    (
        scientific_digest,
        execution_digest,
        run_digest,
        qspace_plan_digest,
        backend_policy_digest,
        source_structure_digest,
    ) = next(iter(identity_values))
    return ScatteringWorkIdentity(
        scientific_digest=scientific_digest,
        execution_digest=execution_digest,
        run_digest=run_digest,
        qspace_plan_digest=qspace_plan_digest,
        backend_policy_digest=backend_policy_digest,
        source_structure_digest=source_structure_digest,
    )


def _interval_payload_input(
    interval_id: int | None,
    path: Path | str,
) -> IntervalPayloadRef | Path:
    payload_path = Path(path)
    if not payload_path.exists():
        return payload_path
    return IntervalPayloadRef(
        path=str(payload_path),
        file_sha256=file_sha256(payload_path),
        interval_id=None if interval_id is None else int(interval_id),
    )


def commit_scattering_attempts_for_chunk(
    *,
    output_dir: str,
    run_digest: str,
    chunk_id: int,
    expected_interval_ids: tuple[int, ...],
    expected_work_unit_digests: tuple[str, ...] = (),
    eps: float = DEFAULT_NUFFT_EPS,
) -> ScatteringChunkCommitManifest:
    require_chunk_quiescence(
        (),
        client=None,
        output_dir=output_dir,
        run_digest=run_digest,
        stage="scattering",
        chunk_id=int(chunk_id),
        expected_work_unit_digests=expected_work_unit_digests,
    )
    # eps drives the PREDICTED same-work agreement tolerance (core/storage/agreement.py).
    candidate = create_scattering_commit_candidate(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_id=int(chunk_id),
        expected_interval_ids=expected_interval_ids,
        eps=eps,
    )
    return promote_scattering_chunk_commit_by_scan(
        output_dir=output_dir,
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
    )


def _mark_scattering_chunk_saved(
    db_manager: DatabaseManager,
    *,
    chunk_id: int,
    expected_interval_ids: tuple[int, ...],
) -> None:
    for interval_id in expected_interval_ids:
        db_manager.update_interval_chunk_status(
            int(interval_id),
            int(chunk_id),
            saved=True,
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


def _store_transient_interval_payload(
    payload_cache: dict[int, object],
    *,
    interval_id: int,
    interval_task: IntervalTask,
    client,
) -> None:
    if client is not None and not is_sync_client(client):
        payload_cache.update(
            client.scatter(
                {int(interval_id): interval_task},
                broadcast=False,
                hash=False,
            )
        )
        return
    payload_cache[int(interval_id)] = interval_task


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


def _estimate_interval_payload_bytes_upper_bound(interval: dict, supercell: np.ndarray) -> int:
    q_points = reciprocal_space_points_counter(to_interval_dict(interval), supercell)
    dim = int(np.asarray(supercell).size)
    return int(q_points) * int((dim * 8) + 16 + 16)


def _local_direct_handoff_limits(parameters: Dict[str, Any]) -> tuple[int, int]:
    runtime_info = _runtime_info(parameters)
    max_intervals = int(
        runtime_info.get(
            "local_direct_handoff_max_intervals",
            _LOCAL_DIRECT_HANDOFF_MAX_INTERVALS_DEFAULT,
        )
    )
    max_bytes = int(
        runtime_info.get(
            "local_direct_handoff_max_bytes",
            _LOCAL_DIRECT_HANDOFF_MAX_BYTES_DEFAULT,
        )
    )
    return max_intervals, max_bytes


def _interval_artifact_present(work_unit: ScatteringWorkUnit) -> bool:
    return (
        work_unit.interval_artifact is not None
        and work_unit.interval_artifact.path is not None
        and Path(work_unit.interval_artifact.path).exists()
    )


def _sqlite_cache_path(db) -> str | None:
    if not getattr(db, "cache_enabled", True):
        return None
    return getattr(db, "db_path", None)


def _local_direct_handoff_is_safe(
    *,
    pending: list[ScatteringWorkUnit],
    interval_lookup: dict[int, dict],
    supercell: np.ndarray,
    parameters: Dict[str, Any],
) -> bool:
    max_intervals, max_bytes = _local_direct_handoff_limits(parameters)
    if len(pending) > max_intervals:
        logger.info(
            "Disabling local direct-handoff: %d intervals exceed safe in-memory limit %d.",
            len(pending),
            max_intervals,
        )
        return False
    estimated_bytes = sum(
        _estimate_interval_payload_bytes_upper_bound(
            interval_lookup[work_unit.interval_id],
            supercell,
        )
        for work_unit in pending
    )
    if estimated_bytes > max_bytes:
        logger.info(
            "Disabling local direct-handoff: estimated interval payload volume %.1f MiB exceeds safe limit %.1f MiB.",
            estimated_bytes / float(1 << 20),
            max_bytes / float(1 << 20),
        )
        return False
    return True


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
    nufft_settings = _nufft_execution_settings(parameters)
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
                and _interval_artifact_present(work_unit)
            )
        )
    ]
    cached = [
        Path(work_unit.interval_artifact.path)
        for work_unit in work_units
        if work_unit.interval_artifact is not None
        and work_unit.interval_artifact.path is not None
        and persist_interval_artifacts
        and _interval_artifact_present(work_unit)
    ]
    cached_payloads = [
        int(work_unit.interval_id)
        for work_unit in work_units
        if int(work_unit.interval_id) in payload_cache
    ]
    if local_fast_handoff and not _local_direct_handoff_is_safe(
        pending=pending,
        interval_lookup=interval_lookup,
        supercell=supercell,
        parameters=parameters,
    ):
        local_fast_handoff = False
        interval_artifact_policy = "required_transport"
        persist_interval_artifacts = True

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
        _require_scheduler_resource_capacity(client, "nufft")
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
                db_path=None,
                nufft_eps=nufft_settings.eps,
                nufft_prefer_cpu=nufft_settings.prefer_cpu,
                nufft_gpu_only=nufft_settings.gpu_only,
                pure=False,
                resources=_nufft_resources_for_parameters(parameters),
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
                    except Exception as exc:
                        raise RuntimeError(
                            "Scattering interval precompute failed before empty-mask "
                            f"classification: interval={int(work_unit.interval_id)}"
                        ) from exc
                    if interval_task is not None:
                        _store_transient_interval_payload(
                            payload_cache,
                            interval_id=int(work_unit.interval_id),
                            interval_task=interval_task,
                            client=client,
                        )
                        produced_payloads += 1
                        if persist_interval_artifacts:
                            manifest = persist_precomputed_interval_artifact(
                                work_unit,
                                interval_task,
                                db_path=_sqlite_cache_path(db),
                            )
                            if manifest is not None and manifest.artifacts:
                                artifact_path = manifest.artifacts[0].path
                                if artifact_path is not None:
                                    written_files.append(Path(artifact_path))
                    else:
                        mark_empty_interval_precomputed(
                            work_unit.interval_id, db_path=_sqlite_cache_path(db),
                        )
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
                    nufft_eps=nufft_settings.eps,
                    nufft_prefer_cpu=nufft_settings.prefer_cpu,
                    nufft_gpu_only=nufft_settings.gpu_only,
                )
                if interval_task is not None:
                    _store_transient_interval_payload(
                        payload_cache,
                        interval_id=int(work_unit.interval_id),
                        interval_task=interval_task,
                        client=client,
                    )
                    produced_payloads += 1
                    if persist_interval_artifacts:
                        manifest = persist_precomputed_interval_artifact(
                            work_unit,
                            interval_task,
                            db_path=_sqlite_cache_path(db),
                        )
                        if manifest is not None and manifest.artifacts:
                            artifact_path = manifest.artifacts[0].path
                            if artifact_path is not None:
                                written_files.append(Path(artifact_path))
                else:
                    mark_empty_interval_precomputed(
                        work_unit.interval_id, db_path=_sqlite_cache_path(db),
                    )
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
        _require_scheduler_resource_capacity(client, "nufft")
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
                db_path=None,
                nufft_eps=nufft_settings.eps,
                nufft_prefer_cpu=nufft_settings.prefer_cpu,
                nufft_gpu_only=nufft_settings.gpu_only,
                pure=False,
                resources=_nufft_resources_for_parameters(parameters),
            )
            for work_unit in pending
        ]
        future_meta = {future: work_unit for future, work_unit in zip(futures, pending)}
        with logging_redirect_tqdm():
            with progress_bar(len(futures), desc="Precompute intervals", unit="intervals") as pbar:
                for future, _ in yield_futures_with_results(futures, client):
                    try:
                        manifest = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            "Scattering durable interval precompute failed: "
                            f"interval={int(getattr(future_meta.get(future), 'interval_id', -1))}"
                        ) from exc
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
                db_path=_sqlite_cache_path(db),
                nufft_eps=nufft_settings.eps,
                nufft_prefer_cpu=nufft_settings.prefer_cpu,
                nufft_gpu_only=nufft_settings.gpu_only,
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
    runtime_provenance: dict[str, Any] | None = None,
    nufft_resources: dict[str, int] | None = None,
    nufft_settings=None,
    transient_interval_payloads: dict[int, object] | None = None,
) -> None:
    total_tasks = len(work_units)
    if total_tasks == 0:
        logger.info("Stage-2 skipped – no unsaved (interval, chunk) pairs.")
        return

    work_identity = _require_stage2_work_identity(work_units)
    interval_paths = interval_paths_for_work_units(work_units)
    if nufft_settings is None:
        nufft_settings = _nufft_execution_settings(
            {"runtime_info": {"nufft_policy": "cpu-only"}}
        )
    interval_refs = {
        int(interval_id): _interval_payload_input(int(interval_id), path)
        for interval_id, path in interval_paths.items()
    }
    payload_cache = transient_interval_payloads if transient_interval_payloads is not None else {}

    def _interval_input_for(work_unit: ScatteringWorkUnit):
        interval_id = int(work_unit.interval_id)
        return payload_cache.get(interval_id, interval_refs[interval_id])

    expected_by_chunk: dict[int, tuple[int, ...]] = {
        chunk_id: tuple(
            sorted(
                int(work_unit.interval_id)
                for work_unit in work_units
                if int(work_unit.chunk_id) == int(chunk_id)
            )
        )
        for chunk_id in chunk_ids_for_work_units(work_units)
    }
    expected_digests_by_chunk: dict[int, tuple[str, ...]] = {
        chunk_id: tuple(
            sorted(
                _scattering_work_unit_digest(work_unit)
                for work_unit in work_units
                if int(work_unit.chunk_id) == int(chunk_id)
            )
        )
        for chunk_id in chunk_ids_for_work_units(work_units)
    }
    task_resources = dict(nufft_resources or {"nufft": 1})
    write_scattering_stage_plan(
        output_dir=output_dir,
        run_digest=work_identity.run_digest,
        expected_by_chunk=expected_by_chunk,
    )
    if client is None or is_sync_client(client):
        rec = point_list_to_recarray(point_data_list)
        failures: list[tuple[ScatteringWorkUnit, str]] = []
        with progress_bar(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:
            for work_unit in work_units:
                atoms = rec[rec.chunk_id == int(work_unit.chunk_id)]
                try:
                    task_kwargs = dict(
                        total_reciprocal_points=total_reciprocal_points,
                        output_dir=output_dir,
                        db_path=None,
                        quiet_logs=False,
                    )
                    if _call_accepts_kwarg(
                        run_scattering_interval_chunk_task,
                        "runtime_provenance",
                    ):
                        task_kwargs["runtime_provenance"] = runtime_provenance
                    _add_nufft_task_kwargs(
                        run_scattering_interval_chunk_task,
                        task_kwargs,
                        nufft_settings,
                    )
                    run_scattering_interval_chunk_task(
                        work_unit,
                        _interval_input_for(work_unit),
                        atoms,
                        **task_kwargs,
                    )
                except Exception as exc:
                    failures.append((work_unit, f"{type(exc).__name__}: {exc}"))
                pbar.update(1)
                pbar.refresh()
        if failures:
            formatted = "; ".join(
                f"iv={work_unit.interval_id} chunk={work_unit.chunk_id} reason={detail}"
                for work_unit, detail in failures
            )
            raise RuntimeError(f"Scattering Stage-2 map failed before reduce: {formatted}")
        for chunk_id, expected_interval_ids in expected_by_chunk.items():
            commit_scattering_attempts_for_chunk(
                output_dir=output_dir,
                run_digest=work_identity.run_digest,
                chunk_id=chunk_id,
                expected_interval_ids=expected_interval_ids,
                expected_work_unit_digests=expected_digests_by_chunk[chunk_id],
                eps=float(nufft_settings.eps),
            )
            _mark_scattering_chunk_saved(
                db_manager,
                chunk_id=chunk_id,
                expected_interval_ids=expected_interval_ids,
            )
        write_scattering_stage_commit(
            output_dir=output_dir,
            run_digest=work_identity.run_digest,
            chunk_ids=tuple(sorted(expected_by_chunk)),
        )
        logger.info("Stage-2 finished (sync) via attempt commit reducer.")
        return

    fail_streak, fail_threshold = 0, 3
    gpu_tripped = False
    _require_scheduler_resource_capacity(client, "nufft")

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
    worker_addresses = _current_worker_addresses(client)
    chunk_owners = {
        int(chunk_id): worker_addresses[index % len(worker_addresses)]
        for index, chunk_id in enumerate(chunk_ids_for_work_units(work_units))
    } if worker_addresses else {}
    chunk_futures = {
        chunk_id: client.scatter(rec[rec.chunk_id == chunk_id], broadcast=False, hash=False)
        for chunk_id in chunk_ids_for_work_units(work_units)
    }
    interval_inputs = {
        interval_id: payload_cache.get(interval_id)
        for interval_id in interval_paths
        if interval_id in payload_cache
    }
    interval_inputs.update(
        {
            interval_id: client.scatter(interval_refs[interval_id], broadcast=False)
            for interval_id in interval_paths
            if interval_id not in interval_inputs
        }
    )

    flying: set = set()
    future_meta: dict = {}
    futures_by_chunk: dict[int, list] = {int(chunk_id): [] for chunk_id in expected_by_chunk}
    failures: list[tuple[ScatteringWorkUnit, str]] = []
    submitted = 0

    def _submit(work_unit: ScatteringWorkUnit) -> None:
        nonlocal submitted
        submit_kwargs = dict(
            total_reciprocal_points=total_reciprocal_points,
            output_dir=output_dir,
            db_path=None,
            quiet_logs=True,
            key=_chunk_task_key(work_unit),
            pure=False,
            resources=task_resources,
            retries=nufft_task_retries(nufft_settings.execution_policy, DEFAULT_TASK_RETRIES),
        )
        if _call_accepts_kwarg(run_scattering_interval_chunk_task, "runtime_provenance"):
            submit_kwargs["runtime_provenance"] = runtime_provenance
        _add_nufft_task_kwargs(
            run_scattering_interval_chunk_task,
            submit_kwargs,
            nufft_settings,
        )
        owner = chunk_owners.get(int(work_unit.chunk_id))
        if owner is not None:
            submit_kwargs["workers"] = [owner]
            submit_kwargs["allow_other_workers"] = True
        future = client.submit(
            run_scattering_interval_chunk_task,
            work_unit,
            interval_inputs[int(work_unit.interval_id)],
            chunk_futures[int(work_unit.chunk_id)],
            **submit_kwargs,
        )
        flying.add(future)
        future_meta[future] = work_unit
        futures_by_chunk.setdefault(int(work_unit.chunk_id), []).append(future)
        submitted += 1

    def _failure_detail(future, result_marker) -> str:
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

    def _future_ok(future, result_marker) -> bool:
        if result_marker is False:
            return False
        status = getattr(future, "status", None)
        if status is not None:
            return status == "finished" and result_marker is not None
        return result_marker is not None

    def _harvest_finished_nonblocking(bump) -> None:
        nonlocal fail_streak
        done_now = [future for future in list(flying) if future.done()]
        for future in done_now:
            flying.discard(future)
            work_unit = future_meta.pop(future, None)
            try:
                result_marker = future.result()
            except Exception:
                result_marker = False
            ok = _future_ok(future, result_marker)
            bump()

            if not ok and work_unit is not None:
                fail_streak += 1
                if fail_streak >= fail_threshold:
                    _trip_to_cpu_only()
                failures.append((work_unit, _failure_detail(future, result_marker)))
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
                    # T2: drain ONE completion then break so the outer
                    # submit-loop can enqueue the next work_unit immediately.
                    drained_one = False
                    for future, result in yield_futures_with_results(list(flying), client):
                        flying.discard(future)
                        completed_work_unit = future_meta.pop(future, None)
                        ok = _future_ok(future, result)
                        bump()
                        if not ok and completed_work_unit is not None:
                            fail_streak += 1
                            if fail_streak >= fail_threshold:
                                _trip_to_cpu_only()
                            failures.append(
                                (completed_work_unit, _failure_detail(future, result))
                            )
                        else:
                            fail_streak = 0
                        drained_one = True
                        break
                    if not drained_one:
                        break

            for future, result in yield_futures_with_results(list(flying), client):
                completed_work_unit = future_meta.pop(future, None)
                flying.discard(future)
                bump()
                if not _future_ok(future, result) and completed_work_unit is not None:
                    failures.append(
                        (completed_work_unit, _failure_detail(future, result))
                    )

    if failures:
        formatted = "; ".join(
            f"iv={work_unit.interval_id} chunk={work_unit.chunk_id} reason={detail}"
            for work_unit, detail in failures
        )
        raise RuntimeError(f"Scattering Stage-2 map failed after Dask retries: {formatted}")

    for chunk_id in sorted(expected_by_chunk):
        require_chunk_quiescence(
            futures_by_chunk.get(int(chunk_id), ()),
            client=client,
            output_dir=output_dir,
            run_digest=work_identity.run_digest,
            stage="scattering",
            chunk_id=int(chunk_id),
            expected_work_unit_digests=expected_digests_by_chunk[int(chunk_id)],
        )

    reducer_futures = []
    future_chunk: dict = {}
    for chunk_id, expected_interval_ids in expected_by_chunk.items():
        reducer_kwargs = dict(
            output_dir=output_dir,
            run_digest=work_identity.run_digest,
            chunk_id=chunk_id,
            expected_interval_ids=expected_interval_ids,
            expected_work_unit_digests=expected_digests_by_chunk[chunk_id],
            pure=False,
            retries=DEFAULT_TASK_RETRIES,
        )
        owner = chunk_owners.get(int(chunk_id))
        if owner is not None:
            reducer_kwargs["workers"] = [owner]
            reducer_kwargs["allow_other_workers"] = True
        future = client.submit(
            commit_scattering_attempts_for_chunk,
            **reducer_kwargs,
        )
        reducer_futures.append(future)
        future_chunk[future] = int(chunk_id)
    reducer_failures: list[str] = []
    for future, result in yield_futures_with_results(reducer_futures, client):
        if not _future_ok(future, result):
            reducer_failures.append(
                f"chunk={future_chunk.get(future)} reason={_failure_detail(future, result)}"
            )
        else:
            chunk_id = int(future_chunk[future])
            _mark_scattering_chunk_saved(
                db_manager,
                chunk_id=chunk_id,
                expected_interval_ids=expected_by_chunk[chunk_id],
            )
    if reducer_failures:
        raise RuntimeError(
            "Scattering Stage-2 reduce failed before publish completion: "
            + "; ".join(reducer_failures)
        )

    write_scattering_stage_commit(
        output_dir=output_dir,
        run_digest=work_identity.run_digest,
        chunk_ids=tuple(sorted(expected_by_chunk)),
    )
    logger.info(
        "Stage-2 finished via attempt commit reducer – %d map tasks submitted",
        submitted,
    )


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
            payload_hashes = {candidate.payload_sha256 for candidate in candidates}
            if len(payload_hashes) != 1:
                raise RuntimeError(
                    f"Conflicting stage-2 replacement attempts for chunk {chunk_id} partition {partition_id}."
                )
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


def run_scattering_stage(
    parameters: Dict[str, Any],
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters: Dict[str, Any],
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor,
    client: "Client | None",
) -> dict[str, object]:
    register_cleanup_plugin(client, is_sync_client=is_sync_client)

    reciprocal_space_intervals = parameters["reciprocal_space_intervals"]
    point_data_list = parameters.get("point_data_list", [])
    original_coords = parameters["original_coords"]
    cells_origin = parameters["cells_origin"]
    elements_arr = parameters["elements"]
    vectors = parameters["vectors"]
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0.0)

    B_ = np.linalg.inv(vectors / supercell)
    unique_elements = np.unique(elements_arr)
    work_identity = _current_scattering_identity(
        parameters=parameters,
        output_dir=output_dir,
        B_=B_,
        mask_params=MaskStrategyParameters,
        MaskStrategy=MaskStrategy,
        client=client,
    )
    fs_capability = profile_output_filesystem(
        output_dir,
        run_digest=work_identity.run_digest,
        client=client,
    )
    nufft_resources = _nufft_resources_for_parameters(parameters)
    if client is not None and not is_sync_client(client):
        require_gpu_admission(
            client,
            policy=_nufft_execution_policy(parameters),
            required_gpu_tasks=(1 if "gpu" in nufft_resources else 0),
        )
    runtime_provenance = _runtime_provenance_for_scattering(
        parameters=parameters,
        fs_capability_digest=fs_capability.capability_digest,
        client=client,
    )
    execution_plan = build_scattering_execution_plan(
        parameters=parameters,
        db_manager=db_manager,
        output_dir=output_dir,
        work_identity=work_identity,
    )
    all_interval_chunk_pairs = list(
        db_manager.get_interval_chunks()
        if hasattr(db_manager, "get_interval_chunks")
        else db_manager.get_unsaved_interval_chunks()
    )
    run_state_snapshot = rebuild_sqlite_cache_from_manifests(
        db_manager,
        output_dir=output_dir,
        run_digest=work_identity.run_digest,
    )
    pending_interval_chunk_pairs = pending_scattering_interval_chunks(
        run_state_snapshot,
        all_interval_chunk_pairs,
    )
    interval_lookup = build_scattering_interval_lookup(reciprocal_space_intervals)
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        run_interval_precompute(
            list(execution_plan.interval_work_units),
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
        if _stage2_replacement_enabled(parameters):
            expected_by_chunk = run_stage2_replacement_execution(
                unsaved_interval_chunks=pending_interval_chunk_pairs,
                total_reciprocal_points=int(execution_plan.total_reciprocal_points),
                point_data_list=list(point_data_list),
                db_manager=db_manager,
                client=client,
                output_dir=output_dir,
                parameter_digest=str(parameters["residual_parameter_digest"]),
                work_identity=work_identity,
                max_intervals_per_shard=_stage2_replacement_batch_size(parameters),
                max_inflight=_stage2_replacement_max_inflight(parameters),
                runtime_provenance=runtime_provenance,
                nufft_resources=nufft_resources,
                nufft_settings=_nufft_execution_settings(parameters),
            )
            parameters["stage2_replacement_expected_by_chunk"] = expected_by_chunk
        else:
            stage2_work_units = build_scattering_interval_chunk_work_units(
                list(pending_interval_chunk_pairs),
                dimension=int(len(supercell)),
                output_dir=output_dir,
                work_identity=work_identity,
            )
            run_interval_chunk_execution(
                stage2_work_units,
                total_reciprocal_points=int(execution_plan.total_reciprocal_points),
                point_data_list=list(point_data_list),
                db_manager=db_manager,
                client=client,
                output_dir=output_dir,
                max_inflight=_stage2_replacement_max_inflight(parameters),
                runtime_provenance=runtime_provenance,
                nufft_resources=nufft_resources,
                nufft_settings=_nufft_execution_settings(parameters),
                transient_interval_payloads=parameters.get("transient_interval_payloads"),
            )
    logger.info("Completed scattering interval precompute stage")
    return {
        "scattering_run_digest": work_identity.run_digest,
        "source_scattering_commit_digest": _stage2_replacement_source_scattering_commit_digest(
            work_identity
        ),
        "stage2_replacement_expected_by_chunk": parameters.get(
            "stage2_replacement_expected_by_chunk",
            {},
        )
    }


__all__ = [
    "run_interval_chunk_execution",
    "run_interval_precompute",
    "run_scattering_stage",
    "run_stage2_replacement_execution",
]
