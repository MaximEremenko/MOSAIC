from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

from core.residual_field.backend import (
    build_residual_field_reducer_backend,
    resolve_residual_field_reducer_backend_kind,
)
from core.scattering.artifacts import (
    discard_stale_interval_artifact,
    interval_artifact_reusable,
    mark_empty_interval_precomputed,
    persist_precomputed_interval_artifact,
)
from core.scattering.contracts import (
    ScatteringIntervalArtifactPolicy,
    ScatteringWorkUnit,
    interval_artifact_dir,
)
from core.scattering.kernels import (
    IntervalTask,
    ReferenceSpec,
    reciprocal_space_points_counter,
    to_interval_dict,
)
from core.scattering.planning import (
    ScatteringWorkIdentity,
    build_scattering_execution_plan,
    build_scattering_interval_lookup,
    prepare_scattering_run_identity,
)
from core.runtime import (
    is_same_node_local_client,
    is_sync_client,
    logging_redirect_tqdm,
    nufft_task_resources,
    profile_output_filesystem,
    progress_bar,
    quiet_loggers,
    require_gpu_admission,
    register_cleanup_plugin,
    yield_futures_with_results,
)
from core.scattering.runtime import (
    _nufft_execution_settings,
    _require_scheduler_resource_capacity,
    _runtime_info,
)
from core.scattering.tasks import (
    compute_scattering_interval_payload,
    run_scattering_interval_task,
)
from core.storage.database_manager import DatabaseManager
from core.storage.digests import digest_dict
from core.workflow.run_state_cache import rebuild_sqlite_cache_from_manifests
from core.scattering.streaming import (
    StreamingComputeContext,
    resolve_stage1_payload_store_dir,
    stage2_streaming_enabled,
)

if TYPE_CHECKING:
    from dask.distributed import Client


logger = logging.getLogger(__name__)
_LOCAL_DIRECT_HANDOFF_MAX_INTERVALS_DEFAULT = 64
_LOCAL_DIRECT_HANDOFF_MAX_BYTES_DEFAULT = 256 << 20


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


def _scheduler_kind(client) -> str:
    if client is None or is_sync_client(client):
        return "sync"
    return "dask"


def _clear_worker_type1_plan_caches(client) -> None:
    """Release the per-process type-1 forward plan caches after Stage-1.

    The plans' fine-grid scratch is raw cudaMalloc outside the CuPy pool, so
    leaving them alive would pin VRAM through the residual-field stage."""
    from core.adapters.cunufft_wrapper import clear_lattice_type1_plan_cache

    try:
        clear_lattice_type1_plan_cache()
    except Exception:
        pass
    if client is None or is_sync_client(client):
        return
    run = getattr(client, "run", None)
    if not callable(run):
        return
    try:
        run(clear_lattice_type1_plan_cache)
    except Exception:
        pass


def _nufft_execution_policy(parameters: Dict[str, Any]) -> str:
    return _nufft_execution_settings(parameters).execution_policy


def _nufft_resources_for_parameters(parameters: Dict[str, Any]) -> dict[str, int]:
    return nufft_task_resources(_nufft_execution_policy(parameters))


def _source_scattering_commit_digest(
    work_identity: ScatteringWorkIdentity,
) -> str:
    # Digest of the full scattering work identity, handed to the residual
    # stage as source_scattering_commit_digest. The domain string predates
    # the stage-2 mode consolidation and is kept for value stability.
    return digest_dict(
        work_identity.to_work_unit_kwargs(),
        domain="mosaic.stage2_replacement.source_scattering_identity.v1",
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
            "stage2-streaming"
            if stage2_streaming_enabled(parameters)
            else "attempt-commit"
        ),
        scheduler_kind=_scheduler_kind(client),
        interval_artifact_policy=str(interval_artifact_policy),
        deterministic_mode=nufft_settings.deterministic_mode,
        thread_count=nufft_settings.thread_count,
        requested_nufft_policy=nufft_settings.requested_policy,
        execution_nufft_policy=nufft_settings.execution_policy,
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
    reference: "ReferenceSpec | None" = None,
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
    if reference is not None:
        payloads["reference"] = client.scatter(reference, broadcast=True, hash=False)
    else:
        payloads["reference"] = None
    return payloads


def _resolve_reference_spec(parameters: Dict[str, Any]) -> "ReferenceSpec | None":
    """Reference for the average-amplitude channel, from run parameters.

    'factorized' (or absent) keeps the crystal average and returns None —
    the key is deliberately not written into the parameters for that mode,
    so pre-existing crystal digests stay byte-identical. 'direct' binds the
    reference to average_coords (the amorphous reference configuration);
    'homogeneous' zeroes the average channel."""
    mode = str(parameters.get("reference_mode") or "factorized").strip().lower()
    if mode == "factorized":
        return None
    if mode == "homogeneous":
        return ReferenceSpec(mode="homogeneous")
    if mode == "direct":
        reference_coords = parameters.get("average_coords")
        if reference_coords is None:
            raise ValueError(
                "reference_mode 'direct' requires average_coords (the "
                "reference configuration) in the scattering parameters."
            )
        return ReferenceSpec(
            mode="direct", coords=np.asarray(reference_coords, dtype=float)
        )
    raise ValueError(
        f"Unknown reference_mode {mode!r}; expected 'factorized', 'direct' "
        "or 'homogeneous'."
    )


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


def _import_stage1_store_payloads(
    work_units: list[ScatteringWorkUnit],
    *,
    store_dir: str | None,
    payload_identity: str | None,
    db,
) -> tuple[list[ScatteringWorkUnit], list[Path]]:
    """Adopt payloads a STREAMING run of the same physics already computed.

    Both modes now persist the identical payload object in the identical
    format; only the directory layout differs. So a mode switch that would
    otherwise repeat every stage-1 transform can instead copy — the store
    entry becomes this run's interval artifact, and the interval is
    committed exactly as if precompute had produced it.

    Only identity-matching entries are adopted. Returns the work units
    still needing compute, plus the artifact paths adopted."""
    if not store_dir or not payload_identity or not work_units:
        return work_units, []
    from core.scattering.streaming import (
        _STORE_MISS,
        read_stored_interval_payload,
    )

    db_path = _sqlite_cache_path(db)
    remaining: list[ScatteringWorkUnit] = []
    adopted: list[Path] = []
    for work_unit in work_units:
        stored = _STORE_MISS
        try:
            stored = read_stored_interval_payload(
                store_dir,
                int(work_unit.interval_id),
                expect_identity=payload_identity,
            )
        except Exception:
            logger.warning(
                "Stage-1 store adoption failed for interval %d; recomputing.",
                int(work_unit.interval_id),
                exc_info=True,
            )
        if stored is _STORE_MISS:
            remaining.append(work_unit)
            continue
        try:
            if stored is None:
                # The store records mask-emptiness durably; precompute mode
                # expresses the same answer as "no artifact, marked done" --
                # which requires clearing any artifact a previous mask left.
                discard_stale_interval_artifact(work_unit)
                mark_empty_interval_precomputed(
                    work_unit.interval_id, db_path=db_path
                )
                continue
            manifest = persist_precomputed_interval_artifact(
                work_unit,
                stored,
                db_path=db_path,
                payload_identity=payload_identity,
            )
        except Exception:
            logger.warning(
                "Could not adopt stage-1 store entry for interval %d; recomputing.",
                int(work_unit.interval_id),
                exc_info=True,
            )
            remaining.append(work_unit)
            continue
        if manifest is not None and manifest.artifacts:
            artifact_path = manifest.artifacts[0].path
            if artifact_path is not None:
                adopted.append(Path(artifact_path))
    if adopted or len(remaining) != len(work_units):
        logger.info(
            "Adopted %d stage-1 payload(s) from the streaming store; %d "
            "interval(s) still need computing.",
            len(work_units) - len(remaining),
            len(remaining),
        )
    return remaining, adopted


def _sqlite_cache_path(db) -> str | None:
    # Both DatabaseManagerProtocol implementations expose these directly.
    if not db.cache_enabled:
        return None
    return db.db_path


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
    payload_identity: str | None = None,
    stage1_store_dir: str | None = None,
    reference: ReferenceSpec | None = None,
) -> list[Path]:
    payload_cache = transient_interval_payloads if transient_interval_payloads is not None else {}
    nufft_settings = _nufft_execution_settings(parameters)
    local_fast_handoff = _local_fast_handoff_enabled(parameters=parameters, client=client)
    interval_artifact_policy = _resolve_scattering_interval_artifact_policy(
        parameters=parameters,
        client=client,
    )
    persist_interval_artifacts = interval_artifact_policy == "required_transport"
    # Validated ONCE per interval: proving reusability opens the artifact,
    # and both the pending and the cached list ask the same question.
    reusable = {
        int(work_unit.interval_id): (
            persist_interval_artifacts
            and interval_artifact_reusable(
                work_unit, payload_identity=payload_identity
            )
        )
        for work_unit in work_units
    }
    pending = [
        work_unit
        for work_unit in work_units
        if int(work_unit.interval_id) not in payload_cache
        and not reusable[int(work_unit.interval_id)]
    ]
    cached = [
        Path(work_unit.interval_artifact.path)
        for work_unit in work_units
        if work_unit.interval_artifact is not None
        and work_unit.interval_artifact.path is not None
        and reusable[int(work_unit.interval_id)]
    ]
    cached_payloads = [
        int(work_unit.interval_id)
        for work_unit in work_units
        if int(work_unit.interval_id) in payload_cache
    ]
    # A streaming run of the same physics may already hold these payloads.
    if persist_interval_artifacts:
        pending, adopted = _import_stage1_store_payloads(
            pending,
            store_dir=stage1_store_dir,
            payload_identity=payload_identity,
            db=db,
        )
        cached.extend(adopted)
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
            reference=reference,
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
                reference=shared_inputs["reference"],
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
                                payload_identity=payload_identity,
                            )
                            if manifest is not None and manifest.artifacts:
                                artifact_path = manifest.artifacts[0].path
                                if artifact_path is not None:
                                    written_files.append(Path(artifact_path))
                    else:
                        discard_stale_interval_artifact(work_unit)
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
                    reference=reference,
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
                            payload_identity=payload_identity,
                        )
                        if manifest is not None and manifest.artifacts:
                            artifact_path = manifest.artifacts[0].path
                            if artifact_path is not None:
                                written_files.append(Path(artifact_path))
                else:
                    discard_stale_interval_artifact(work_unit)
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
            reference=reference,
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
                payload_identity=payload_identity,
                reference=shared_inputs["reference"],
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
                payload_identity=payload_identity,
                reference=reference,
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
    original_coords = parameters["original_coords"]
    cells_origin = parameters["cells_origin"]
    elements_arr = parameters["elements"]
    vectors = parameters["vectors"]
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0.0)

    B_ = np.linalg.inv(vectors / supercell)
    unique_elements = np.unique(elements_arr)
    reference = _resolve_reference_spec(parameters)
    work_identity = _current_scattering_identity(
        parameters=parameters,
        output_dir=output_dir,
        B_=B_,
        mask_params=MaskStrategyParameters,
        MaskStrategy=MaskStrategy,
        client=client,
    )
    profile_output_filesystem(
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
    execution_plan = build_scattering_execution_plan(
        parameters=parameters,
        db_manager=db_manager,
        output_dir=output_dir,
        work_identity=work_identity,
    )
    rebuild_sqlite_cache_from_manifests(
        db_manager,
        output_dir=output_dir,
        run_digest=work_identity.run_digest,
    )
    # What a durable stage-1 payload must match to be reused — by either
    # mode, from either directory. Derived once, with the run identity.
    payload_identity = work_identity.interval_payload_identity or None
    interval_lookup = build_scattering_interval_lookup(reciprocal_space_intervals)
    if stage2_streaming_enabled(parameters):
        # Streaming (fused stage-1) mode: compute NO interval payloads and
        # write NO durable interval store here. Publish the compute context;
        # the residual stage's work units run the stage-1 kernels themselves,
        # fold into worker-local subchunk accumulators, and discard the
        # amplitudes. Set runtime_info.save_scattering_interval_artifacts to
        # additionally persist inspection copies (not implemented in
        # streaming mode yet -- artifacts would not be consumed).
        streaming_sink = parameters.get("streaming_state")
        if not isinstance(streaming_sink, dict):
            raise RuntimeError(
                "Streaming stage-2 mode requires the workflow artifacts "
                "streaming_state sink in the scattering parameters."
            )
        streaming_nufft = _nufft_execution_settings(parameters)
        # Durable stage-1 payload store: computed payloads persist under the
        # output dir, scoped by the scattering identity, so stage-1 runs once
        # per interval across shards, owners, restarts, and cluster sizes.
        # MOSAIC_STREAMING_PAYLOAD_STORE=0 disables; a path value relocates it
        # (e.g. onto a parallel filesystem for multi-node runs).
        payload_store_dir = (
            resolve_stage1_payload_store_dir(output_dir, payload_identity)
            if payload_identity
            else None
        )
        if payload_store_dir is not None:
            Path(payload_store_dir).mkdir(parents=True, exist_ok=True)

        streaming_sink["compute_context"] = StreamingComputeContext(
            cache_token=str(work_identity.run_digest),
            interval_lookup=dict(interval_lookup),
            B_=B_,
            mask_params=MaskStrategyParameters,
            MaskStrategy=MaskStrategy,
            supercell=supercell,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            charge=charge,
            use_coeff=("coeff" in parameters),
            coeff_val=parameters.get("coeff"),
            unique_elements=tuple(str(element) for element in unique_elements),
            ff_factory=FormFactorFactoryProducer,
            nufft_eps=float(streaming_nufft.eps),
            nufft_prefer_cpu=bool(streaming_nufft.prefer_cpu),
            nufft_gpu_only=bool(streaming_nufft.gpu_only),
            payload_store_dir=payload_store_dir,
            precomputed_artifact_dir=str(interval_artifact_dir(output_dir)),
            payload_identity=payload_identity,
            reference=reference,
        )
        logger.info(
            "Scattering stage-1/stage-2 skipped (streaming mode): %d interval(s) "
            "computed inside residual work units; durable stage-1 payload "
            "store: %s; precompute artifacts reusable from %s.",
            len(interval_lookup),
            payload_store_dir or "disabled",
            interval_artifact_dir(output_dir),
        )
        return {
            "scattering_run_digest": work_identity.run_digest,
            "source_scattering_commit_digest": _source_scattering_commit_digest(
                work_identity
            ),
        }
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        try:
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
                payload_identity=payload_identity,
                # A streaming run of the same physics leaves its payloads
                # here; adopting them beats recomputing stage-1.
                stage1_store_dir=(
                    resolve_stage1_payload_store_dir(output_dir, payload_identity)
                    if payload_identity
                    else None
                ),
                reference=reference,
            )
        finally:
            # The type-1 plans only fill during interval precompute; release
            # their raw-cudaMalloc scratch BEFORE the stage-2 / residual
            # inverse transforms run (and on failure paths), or it pins VRAM
            # their budget models assume is free.
            _clear_worker_type1_plan_caches(client)
        logger.info(
            "Scattering Stage-2 deferred: residual-field stage will consume "
            "precomputed interval artifacts."
        )
    logger.info("Completed scattering interval precompute stage")
    return {
        "scattering_run_digest": work_identity.run_digest,
        "source_scattering_commit_digest": _source_scattering_commit_digest(
            work_identity
        ),
    }


__all__ = [
    "run_interval_precompute",
    "run_scattering_stage",
]
