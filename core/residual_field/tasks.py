from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from core.residual_field.backend import (
    ResidualFieldReducerBackend,
    build_residual_field_reducer_backend,
    get_process_local_residual_field_backend,
)
from core.scattering.accumulation import apply_half_space_conjugate_reconstruction
from core.scattering.kernels import build_rifft_grid_for_chunk
from core.scattering.kernels import IntervalTask
from core.scattering.tasks import (
    IntervalPayloadRef,
    load_interval_task_payload,
    scattering_contribution_point_count,
)
from core.residual_field.contracts import (
    ResidualFieldAccumulatorStatus,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_super_batch,
)
from core.residual_field.commit import write_residual_attempt
from core.runtime import handle_worker_gpu_failure, task_progress_enabled


logger = logging.getLogger(__name__)

_worker_logging_configured = False
_RIFFT_PAYLOAD_CACHE: "OrderedDict[tuple, tuple[tuple[np.ndarray, np.ndarray], int]]" = OrderedDict()
_RIFFT_PAYLOAD_CACHE_BYTES = 0
_RIFFT_PAYLOAD_CACHE_LOCK = threading.Lock()
_RIFFT_PAYLOAD_CACHE_MAX_BYTES_DEFAULT = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ResidualChunkComputeResult:
    grid_shape_nd: np.ndarray
    contribution_reciprocal_points: int
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray
    point_ids: np.ndarray


def _ensure_worker_logging() -> None:
    """Ensure root logger has a handler in Dask worker processes."""
    global _worker_logging_configured
    if _worker_logging_configured:
        return
    _worker_logging_configured = True
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s:%(lineno)d) - %(message)s"
        ))
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def _task_progress_enabled(quiet_logs: bool) -> bool:
    if not quiet_logs:
        return True
    return task_progress_enabled(False)


def _same_q_grid_presum_enabled() -> bool:
    raw = os.getenv("MOSAIC_RESIDUAL_SAME_Q_GRID_PRESUM")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _riff_payload_cache_enabled() -> bool:
    raw = os.getenv("MOSAIC_RESIDUAL_RIFFT_PAYLOAD_CACHE")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _riff_payload_cache_max_bytes() -> int:
    raw = os.getenv("MOSAIC_RESIDUAL_RIFFT_PAYLOAD_CACHE_MAX_BYTES")
    try:
        value = int(raw) if raw is not None else _RIFFT_PAYLOAD_CACHE_MAX_BYTES_DEFAULT
    except ValueError:
        value = _RIFFT_PAYLOAD_CACHE_MAX_BYTES_DEFAULT
    return max(0, int(value))


def _riff_payload_cache_key(
    atoms: np.recarray,
    work_unit: ResidualFieldWorkUnit,
) -> tuple:
    return (
        id(build_rifft_grid_for_chunk),
        id(atoms),
        str(work_unit.parameter_digest),
        int(work_unit.chunk_id),
        None if work_unit.partition_id is None else int(work_unit.partition_id),
        None if work_unit.point_start is None else int(work_unit.point_start),
        None if work_unit.point_stop is None else int(work_unit.point_stop),
        int(getattr(atoms, "shape", (len(atoms),))[0]),
    )


def clear_residual_rifft_payload_cache() -> None:
    global _RIFFT_PAYLOAD_CACHE_BYTES
    with _RIFFT_PAYLOAD_CACHE_LOCK:
        _RIFFT_PAYLOAD_CACHE.clear()
        _RIFFT_PAYLOAD_CACHE_BYTES = 0


def _cached_rifft_payload(key: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    if not _riff_payload_cache_enabled():
        return None
    with _RIFFT_PAYLOAD_CACHE_LOCK:
        entry = _RIFFT_PAYLOAD_CACHE.get(key)
        if entry is None:
            return None
        payload, _size = entry
        _RIFFT_PAYLOAD_CACHE.move_to_end(key)
        return payload


def _store_rifft_payload_cache(
    key: tuple,
    payload: tuple[np.ndarray, np.ndarray],
) -> None:
    global _RIFFT_PAYLOAD_CACHE_BYTES
    if not _riff_payload_cache_enabled():
        return
    max_bytes = _riff_payload_cache_max_bytes()
    if max_bytes <= 0:
        return
    payload_bytes = int(payload[0].nbytes + payload[1].nbytes)
    if payload_bytes > max_bytes:
        return
    with _RIFFT_PAYLOAD_CACHE_LOCK:
        existing = _RIFFT_PAYLOAD_CACHE.pop(key, None)
        if existing is not None:
            _RIFFT_PAYLOAD_CACHE_BYTES -= int(existing[1])
        _RIFFT_PAYLOAD_CACHE[key] = (payload, payload_bytes)
        _RIFFT_PAYLOAD_CACHE_BYTES += payload_bytes
        while _RIFFT_PAYLOAD_CACHE_BYTES > max_bytes and _RIFFT_PAYLOAD_CACHE:
            _old_key, (_old_payload, old_size) = _RIFFT_PAYLOAD_CACHE.popitem(last=False)
            _RIFFT_PAYLOAD_CACHE_BYTES -= int(old_size)


def _normalize_interval_inputs(
    interval_inputs: Path | str | IntervalTask | IntervalPayloadRef | Sequence[Path | str | IntervalTask | IntervalPayloadRef],
) -> tuple[Path | IntervalTask | IntervalPayloadRef, ...]:
    if isinstance(interval_inputs, IntervalTask):
        return (interval_inputs,)
    if isinstance(interval_inputs, IntervalPayloadRef):
        return (interval_inputs,)
    if isinstance(interval_inputs, (str, Path)):
        return (Path(interval_inputs),)
    normalized: list[Path | IntervalTask | IntervalPayloadRef] = []
    for item in interval_inputs:
        if isinstance(item, IntervalTask):
            normalized.append(item)
        elif isinstance(item, IntervalPayloadRef):
            normalized.append(item)
        else:
            normalized.append(Path(item))
    return tuple(normalized)


def _q_grid_signature(q_grid: np.ndarray, q_grid_digest: str | None = None) -> tuple:
    arr = np.asarray(q_grid)
    if q_grid_digest:
        return (tuple(arr.shape), str(arr.dtype), str(q_grid_digest))
    return (tuple(arr.shape), str(arr.dtype), arr.tobytes())


def _interval_task_sort_key(interval_task: IntervalTask) -> tuple:
    return (
        int(interval_task.irecip_id),
        str(interval_task.half_space_role),
        int(interval_task.reciprocal_multiplicity),
    )


def _atoms_to_chunk_data(atoms: np.recarray) -> list[dict]:
    return [
        {
            "coordinates": atoms["coordinates"][index],
            "dist_from_atom_center": atoms["dist_from_atom_center"][index],
            "step_in_frac": atoms["step_in_frac"][index],
        }
        for index in range(atoms.shape[0])
    ]


def _slice_work_unit_atoms(
    atoms: np.recarray,
    work_unit: ResidualFieldWorkUnit,
) -> np.recarray:
    if work_unit.point_start is None and work_unit.point_stop is None:
        return atoms
    start = int(work_unit.point_start or 0)
    stop = int(work_unit.point_stop or len(atoms))
    return atoms[start:stop]


def _normalize_rifft_payload(rifft_payload) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(rifft_payload, "result") and not isinstance(rifft_payload, tuple):
        rifft_payload = rifft_payload.result()
    if (
        not isinstance(rifft_payload, tuple)
        or len(rifft_payload) != 2
    ):
        raise ValueError("Residual-field RIFFT payload must be a (rifft_grid, grid_shape_nd) tuple.")
    rifft_grid, grid_shape_nd = rifft_payload
    return (
        np.asarray(rifft_grid, dtype=np.float64),
        np.asarray(grid_shape_nd, dtype=np.int64),
    )


def _has_residual_attempt_identity(work_unit: ResidualFieldWorkUnit) -> bool:
    required = (
        work_unit.run_digest,
        work_unit.partition_plan_digest,
        work_unit.source_scattering_commit_digest,
        work_unit.backend_policy_digest,
        work_unit.expected_output_digest,
    )
    return (
        all(isinstance(value, str) and value for value in required)
        and work_unit.partition_id is not None
        and work_unit.point_start is not None
        and work_unit.point_stop is not None
    )


def build_residual_rifft_payload(
    atoms: np.recarray,
    *,
    work_unit: ResidualFieldWorkUnit,
    quiet_logs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    show_progress = _task_progress_enabled(quiet_logs)
    cache_key = _riff_payload_cache_key(atoms, work_unit)
    cached = _cached_rifft_payload(cache_key)
    if cached is not None:
        if show_progress:
            logger.debug(
                "Residual RIFFT grid cache hit | chunk=%d | partition=%s | rifft_points=%d | bytes=%d",
                int(work_unit.chunk_id),
                work_unit.partition_id,
                int(cached[0].shape[0]),
                int(cached[0].nbytes + cached[1].nbytes),
            )
        return cached
    partition_atoms = _slice_work_unit_atoms(atoms, work_unit)
    build_start = time.perf_counter()
    rifft_grid, grid_shape_nd = build_rifft_grid_for_chunk(
        _atoms_to_chunk_data(partition_atoms)
    )
    rifft_grid = np.asarray(rifft_grid, dtype=np.float64)
    grid_shape_nd = np.asarray(grid_shape_nd, dtype=np.int64)
    if show_progress:
        logger.debug(
            "Residual RIFFT grid build | chunk=%d | partition=%s | points=%d | rifft_points=%d | bytes=%d | duration=%.3fs",
            int(work_unit.chunk_id),
            work_unit.partition_id,
            int(partition_atoms.shape[0]),
            int(rifft_grid.shape[0]),
            int(rifft_grid.nbytes + grid_shape_nd.nbytes),
            time.perf_counter() - build_start,
        )
    payload = (rifft_grid, grid_shape_nd)
    _store_rifft_payload_cache(cache_key, payload)
    return payload


def _pre_sum_same_q_grid_weights(
    interval_tasks: Sequence[IntervalTask],
    *,
    reference_q_grid: np.ndarray,
) -> np.ndarray:
    q_point_count = int(np.asarray(reference_q_grid).shape[0])
    summed_weights = np.zeros((2, q_point_count), dtype=np.complex128)
    has_intervals = False
    for interval_task in interval_tasks:
        q_amp = np.asarray(interval_task.q_amp, dtype=np.complex128).reshape(-1)
        q_amp_av = np.asarray(interval_task.q_amp_av, dtype=np.complex128).reshape(-1)
        if q_amp.shape[0] != q_point_count:
            raise ValueError(
                "Residual-field interval q_amp length does not match q_grid: "
                f"interval={int(interval_task.irecip_id)} q_amp={q_amp.shape[0]} q_grid={q_point_count}"
            )
        if q_amp_av.shape[0] != q_point_count:
            raise ValueError(
                "Residual-field interval q_amp_av length does not match q_grid: "
                f"interval={int(interval_task.irecip_id)} q_amp_av={q_amp_av.shape[0]} q_grid={q_point_count}"
            )
        summed_weights[0] += q_amp
        summed_weights[0] -= q_amp_av
        summed_weights[1] += q_amp_av
        has_intervals = True
    if not has_intervals:
        raise ValueError("Residual-field same-q-grid group is empty.")
    return summed_weights


def _validate_same_q_grid_weight_shapes(
    interval_tasks: Sequence[IntervalTask],
    *,
    reference_q_grid: np.ndarray,
) -> None:
    q_point_count = int(np.asarray(reference_q_grid).shape[0])
    if not interval_tasks:
        raise ValueError("Residual-field same-q-grid group is empty.")
    for interval_task in interval_tasks:
        q_amp = np.asarray(interval_task.q_amp).reshape(-1)
        q_amp_av = np.asarray(interval_task.q_amp_av).reshape(-1)
        if q_amp.shape[0] != q_point_count:
            raise ValueError(
                "Residual-field interval q_amp length does not match q_grid: "
                f"interval={int(interval_task.irecip_id)} q_amp={q_amp.shape[0]} q_grid={q_point_count}"
            )
        if q_amp_av.shape[0] != q_point_count:
            raise ValueError(
                "Residual-field interval q_amp_av length does not match q_grid: "
                f"interval={int(interval_task.irecip_id)} q_amp_av={q_amp_av.shape[0]} q_grid={q_point_count}"
            )


def _call_inverse_super_batch(
    *,
    q_coords: np.ndarray,
    weights: np.ndarray,
    real_coords: np.ndarray,
    eps: float,
    prefer_cpu: bool,
    gpu_only: bool,
) -> np.ndarray:
    kwargs = {
        "q_coords": q_coords,
        "weights": weights,
        "real_coords": real_coords,
        "eps": eps,
    }
    try:
        signature = inspect.signature(execute_inverse_cunufft_super_batch)
    except (TypeError, ValueError):
        kwargs.update({"prefer_cpu": prefer_cpu, "gpu_only": gpu_only})
    else:
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs or "prefer_cpu" in signature.parameters:
            kwargs["prefer_cpu"] = prefer_cpu
        if accepts_var_kwargs or "gpu_only" in signature.parameters:
            kwargs["gpu_only"] = gpu_only
    return execute_inverse_cunufft_super_batch(**kwargs)


def compute_residual_field_interval_chunk_arrays(
    interval_tasks: Sequence[IntervalTask],
    *,
    rifft_grid: np.ndarray,
    grid_shape_nd: np.ndarray,
    point_start: int = 0,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> ResidualChunkComputeResult:
    ordered_interval_tasks = sorted(interval_tasks, key=_interval_task_sort_key)
    grouped_interval_tasks: dict[tuple, list[IntervalTask]] = {}
    contribution_reciprocal_points = 0
    for interval_task in ordered_interval_tasks:
        grouped_interval_tasks.setdefault(
            (
                _q_grid_signature(
                    interval_task.q_grid,
                    getattr(interval_task, "q_grid_digest", None),
                ),
                interval_task.half_space_role,
            ),
            [],
        ).append(interval_task)
        contribution_reciprocal_points += scattering_contribution_point_count(interval_task)
    if not grouped_interval_tasks:
        raise ValueError("Residual-field batch task produced no interval contributions.")

    amplitudes_delta = None
    amplitudes_average = None
    use_presum = _same_q_grid_presum_enabled()
    ordered_groups = [
        grouped_interval_tasks[key]
        for key in sorted(grouped_interval_tasks, key=lambda item: (str(item[1]), item[0]))
    ]
    for grouped_tasks in ordered_groups:
        grouped_tasks = sorted(grouped_tasks, key=_interval_task_sort_key)
        reference_q_grid = grouped_tasks[0].q_grid
        if use_presum:
            inverse_weights = _pre_sum_same_q_grid_weights(
                grouped_tasks,
                reference_q_grid=reference_q_grid,
            )
        else:
            _validate_same_q_grid_weight_shapes(
                grouped_tasks,
                reference_q_grid=reference_q_grid,
            )
            stacked_weights = []
            for interval_task in grouped_tasks:
                stacked_weights.extend(
                    [
                        interval_task.q_amp - interval_task.q_amp_av,
                        interval_task.q_amp_av,
                    ]
                )
            inverse_weights = np.stack(stacked_weights, axis=0)
            del stacked_weights
        inverse_outputs = _call_inverse_super_batch(
            q_coords=reference_q_grid,
            weights=inverse_weights,
            real_coords=rifft_grid,
            eps=nufft_eps,
            prefer_cpu=nufft_prefer_cpu,
            gpu_only=nufft_gpu_only,
        )
        del inverse_weights
        inverse_outputs = np.asarray(inverse_outputs, dtype=np.complex128)
        if use_presum:
            if inverse_outputs.shape[0] != 2:
                raise ValueError(
                    "Residual-field same-q-grid inverse expected two output transforms; "
                    f"got {inverse_outputs.shape[0]}"
                )
            grouped_delta = inverse_outputs[0]
            grouped_average = inverse_outputs[1]
        else:
            grouped_delta = np.sum(inverse_outputs[0::2], axis=0, dtype=np.complex128)
            grouped_average = np.sum(inverse_outputs[1::2], axis=0, dtype=np.complex128)
        grouped_delta = apply_half_space_conjugate_reconstruction(
            grouped_delta,
            reference_q_grid,
            grouped_tasks[0].half_space_role,
        )
        grouped_average = apply_half_space_conjugate_reconstruction(
            grouped_average,
            reference_q_grid,
            grouped_tasks[0].half_space_role,
        )
        del inverse_outputs
        if amplitudes_delta is None:
            amplitudes_delta = grouped_delta
        else:
            amplitudes_delta += grouped_delta
            del grouped_delta
        if amplitudes_average is None:
            amplitudes_average = grouped_average
        else:
            amplitudes_average += grouped_average
            del grouped_average
    if amplitudes_delta is None or amplitudes_average is None:
        raise ValueError("Residual-field batch task produced no inverse outputs.")
    point_offset = int(point_start or 0)
    point_ids = point_offset + np.arange(amplitudes_delta.shape[0], dtype=np.int64)
    return ResidualChunkComputeResult(
        grid_shape_nd=np.asarray(grid_shape_nd, dtype=np.int64),
        contribution_reciprocal_points=int(contribution_reciprocal_points),
        amplitudes_delta=np.asarray(amplitudes_delta, dtype=np.complex128),
        amplitudes_average=np.asarray(amplitudes_average, dtype=np.complex128),
        point_ids=point_ids,
    )


def run_residual_field_interval_chunk_task(
    work_unit: ResidualFieldWorkUnit,
    interval_paths: Path | str | IntervalTask | IntervalPayloadRef | Sequence[Path | str | IntervalTask | IntervalPayloadRef],
    atoms: np.recarray | None,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    db_path: str | None = None,
    scratch_root: str | None = None,
    reducer_backend: ResidualFieldReducerBackend | None = None,
    total_expected_partials: int | None = None,
    owner_local_reducer: bool = False,
    quiet_logs: bool = False,
    rifft_payload: tuple[np.ndarray, np.ndarray] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> ResidualFieldShardManifest | ResidualFieldAccumulatorStatus | None:
    _ensure_worker_logging()
    interval_ids = work_unit.interval_ids or ((work_unit.interval_id,) if work_unit.interval_id is not None else ())
    try:
        show_progress = _task_progress_enabled(quiet_logs)
        resolved_backend = reducer_backend or build_residual_field_reducer_backend(
            "local_restartable"
        )
        loaded_interval_inputs = _normalize_interval_inputs(interval_paths)
        if not loaded_interval_inputs:
            raise ValueError("Residual-field batch task requires at least one interval artifact path.")
        if owner_local_reducer:
            if scratch_root is None or db_path is None or total_expected_partials is None:
                raise ValueError(
                    "Owner-local reducer tasks require scratch_root, db_path, and total_expected_partials."
                )
            if not callable(getattr(resolved_backend, "accept_local_contribution", None)):
                raise ValueError(
                    "Residual-field owner-local reduction requires accept_local_contribution support."
                )
            worker_backend = get_process_local_residual_field_backend(
                resolved_backend
            )
            if not callable(getattr(worker_backend, "accept_local_contribution", None)):
                raise ValueError(
                    "Residual-field owner-local reduction requires process-local accept_local_contribution support."
                )
            already_durable = getattr(
                worker_backend,
                "local_intervals_already_durable",
                lambda *_args, **_kwargs: False,
            )
            if already_durable(
                work_unit,
                output_dir=output_dir,
            ):
                return ResidualFieldAccumulatorStatus(
                    artifact_key=work_unit.artifact_key,
                    chunk_id=work_unit.chunk_id,
                    parameter_digest=work_unit.parameter_digest,
                    interval_ids=work_unit.interval_ids,
                    partition_id=work_unit.partition_id,
                    contribution_reciprocal_point_count=0,
                    total_reciprocal_points=total_reciprocal_points,
                )
        if rifft_payload is None:
            if atoms is None:
                raise ValueError(
                    "Residual-field batch task requires atoms when no RIFFT payload is supplied."
                )
            rifft_grid, grid_shape_nd = build_residual_rifft_payload(
                atoms,
                work_unit=work_unit,
                quiet_logs=quiet_logs,
            )
        else:
            rifft_grid, grid_shape_nd = _normalize_rifft_payload(rifft_payload)
        if show_progress:
            logger.debug(
                "Residual batch start | chunk=%d | partition=%s | intervals=%s | rifft_points=%d",
                int(work_unit.chunk_id),
                work_unit.partition_id,
                ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
                int(rifft_grid.shape[0]),
            )
        interval_tasks = sorted(
            [
                interval_input
                if isinstance(interval_input, IntervalTask)
                else load_interval_task_payload(interval_input)
                for interval_input in loaded_interval_inputs
            ],
            key=_interval_task_sort_key,
        )
        compute_result = compute_residual_field_interval_chunk_arrays(
            interval_tasks,
            rifft_grid=rifft_grid,
            grid_shape_nd=grid_shape_nd,
            point_start=int(work_unit.point_start or 0),
            nufft_eps=nufft_eps,
            nufft_prefer_cpu=nufft_prefer_cpu,
            nufft_gpu_only=nufft_gpu_only,
        )
        grid_shape_nd = compute_result.grid_shape_nd
        contribution_reciprocal_points = compute_result.contribution_reciprocal_points
        amplitudes_delta = compute_result.amplitudes_delta
        amplitudes_average = compute_result.amplitudes_average
        point_ids = compute_result.point_ids
        if owner_local_reducer:
            if db_path is None:
                raise ValueError("Owner-local residual reducer requires a driver DB cache path.")
            worker_backend.accept_local_contribution(
                work_unit,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                contribution_reciprocal_points=contribution_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
                point_ids=point_ids,
                output_dir=output_dir,
                scratch_root=scratch_root,
                db_path=db_path,
                total_expected_partials=total_expected_partials,
                cleanup_policy="off",
            )
            return ResidualFieldAccumulatorStatus(
                artifact_key=work_unit.artifact_key,
                chunk_id=work_unit.chunk_id,
                parameter_digest=work_unit.parameter_digest,
                interval_ids=work_unit.interval_ids,
                partition_id=work_unit.partition_id,
                contribution_reciprocal_point_count=contribution_reciprocal_points,
                total_reciprocal_points=total_reciprocal_points,
            )
        if show_progress:
            logger.debug(
                "Residual batch persisting durable shard | chunk=%d | intervals=%s",
                int(work_unit.chunk_id),
                ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
            )
        if _has_residual_attempt_identity(work_unit):
            return write_residual_attempt(
                output_dir=output_dir,
                run_digest=str(work_unit.run_digest),
                chunk_id=int(work_unit.chunk_id),
                partition_id=int(work_unit.partition_id),
                point_start=int(work_unit.point_start),
                point_stop=int(work_unit.point_start) + int(amplitudes_delta.shape[0]),
                interval_ids=tuple(int(interval_id) for interval_id in interval_ids),
                attempt_id=f"partition-{int(work_unit.partition_id)}-attempt-{uuid4().hex}",
                parameter_digest=str(work_unit.parameter_digest),
                partition_plan_digest=str(work_unit.partition_plan_digest),
                source_scattering_commit_digest=str(work_unit.source_scattering_commit_digest),
                source_replacement_digest=work_unit.source_replacement_digest,
                backend_policy_digest=str(work_unit.backend_policy_digest),
                expected_output_digest=str(work_unit.expected_output_digest),
                grid_shape_nd=grid_shape_nd,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
                contribution_reciprocal_points=contribution_reciprocal_points,
                point_ids=point_ids,
                runtime_provenance=runtime_provenance,
            )
        current_checkpoint = getattr(resolved_backend, "persist" "_shard_checkpoint", None)
        if current_checkpoint is None:
            raise ValueError(
                "Current residual-field tasks require identity-complete work units and "
                "run-scoped attempt manifests."
            )
        return current_checkpoint(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_ids=point_ids,
            output_dir=output_dir,
            scratch_root=scratch_root,
            quiet_logs=quiet_logs,
        )
    except Exception as err:
        logger.error(
            "chunk %d | batch %s FAILED: %s",
            work_unit.chunk_id,
            ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
            err,
            exc_info=True,
        )
        handle_worker_gpu_failure(err, logger=logger)
        raise


__all__ = [
    "ResidualChunkComputeResult",
    "build_residual_rifft_payload",
    "clear_residual_rifft_payload_cache",
    "compute_residual_field_interval_chunk_arrays",
    "run_residual_field_interval_chunk_task",
]
