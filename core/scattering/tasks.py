from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import h5py
import numpy as np

from core.scattering.accumulation import apply_half_space_conjugate_reconstruction
from core.scattering.half_space import (
    classify_interval_half_space_role,
    half_space_role_multiplicity,
    normalize_half_space_role,
)
from core.scattering.artifacts import (
    build_scattering_interval_manifest,
    is_interval_artifact_committed,
    mark_empty_interval_precomputed,
    persist_precomputed_interval_artifact,
    persist_scattering_interval_chunk_result,
    persist_scattering_interval_chunk_shard,
)
from core.scattering.contracts import ScatteringArtifactManifest, ScatteringWorkUnit
from core.scattering.kernels import (
    IntervalTask,
    aggregate_interval_contributions,
    build_interval_lattice_meta,
    build_rifft_grid_for_chunk,
    compute_interval_coeff_contribution,
    compute_interval_element_contribution,
    forward_interval_amplitudes,
    generate_q_space_grid_sync,
)
from core.adapters.cunufft_wrapper import (
    execute_inverse_cunufft_batch_materialize_once,
)
from core.contracts import CompletionStatus
from core.runtime import handle_worker_gpu_failure
from core.storage.fingerprint import file_sha256


logger = logging.getLogger(__name__)
_INTERVAL_PAYLOAD_CACHE: "OrderedDict[tuple[str, str], tuple[IntervalTask, int]]" = OrderedDict()
_INTERVAL_PAYLOAD_CACHE_BYTES = 0
_INTERVAL_PAYLOAD_CACHE_LOCK = threading.Lock()
_INTERVAL_PAYLOAD_CACHE_MAX_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class IntervalPayloadRef:
    path: str
    file_sha256: str
    interval_id: int | None = None
    q_grid_digest: str | None = None


@dataclass(frozen=True)
class ScatteringChunkComputeResult:
    grid_shape_nd: np.ndarray
    contribution_reciprocal_points: int
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray


def _lower_worker_log_levels() -> None:
    for name in (
        __name__,
        "core.storage.database_manager",
        "DatabaseManager",
        "core.patch_centers.point_data",
        "PointDataProcessor",
        "RIFFTInDataSaver",
    ):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:
            pass


def _interval_payload_cache_enabled() -> bool:
    raw = os.getenv("MOSAIC_SCATTERING_INTERVAL_PAYLOAD_CACHE")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _interval_payload_cache_max_bytes() -> int:
    raw = os.getenv("MOSAIC_SCATTERING_INTERVAL_PAYLOAD_CACHE_MAX_BYTES")
    try:
        value = int(raw) if raw is not None else _INTERVAL_PAYLOAD_CACHE_MAX_BYTES_DEFAULT
    except ValueError:
        value = _INTERVAL_PAYLOAD_CACHE_MAX_BYTES_DEFAULT
    return max(0, int(value))


def clear_scattering_interval_payload_cache() -> None:
    global _INTERVAL_PAYLOAD_CACHE_BYTES
    with _INTERVAL_PAYLOAD_CACHE_LOCK:
        _INTERVAL_PAYLOAD_CACHE.clear()
        _INTERVAL_PAYLOAD_CACHE_BYTES = 0


def _interval_task_nbytes(interval_task: IntervalTask) -> int:
    return int(
        np.asarray(interval_task.q_grid).nbytes
        + np.asarray(interval_task.q_amp).nbytes
        + np.asarray(interval_task.q_amp_av).nbytes
    )


def _cached_interval_payload(key: tuple[str, str]) -> IntervalTask | None:
    if not _interval_payload_cache_enabled():
        return None
    with _INTERVAL_PAYLOAD_CACHE_LOCK:
        entry = _INTERVAL_PAYLOAD_CACHE.get(key)
        if entry is None:
            return None
        payload, _size = entry
        _INTERVAL_PAYLOAD_CACHE.move_to_end(key)
        return payload


def _store_interval_payload_cache(key: tuple[str, str], payload: IntervalTask) -> None:
    global _INTERVAL_PAYLOAD_CACHE_BYTES
    if not _interval_payload_cache_enabled():
        return
    max_bytes = _interval_payload_cache_max_bytes()
    if max_bytes <= 0:
        return
    payload_bytes = _interval_task_nbytes(payload)
    if payload_bytes > max_bytes:
        return
    with _INTERVAL_PAYLOAD_CACHE_LOCK:
        existing = _INTERVAL_PAYLOAD_CACHE.pop(key, None)
        if existing is not None:
            _INTERVAL_PAYLOAD_CACHE_BYTES -= int(existing[1])
        _INTERVAL_PAYLOAD_CACHE[key] = (payload, payload_bytes)
        _INTERVAL_PAYLOAD_CACHE_BYTES += payload_bytes
        while _INTERVAL_PAYLOAD_CACHE_BYTES > max_bytes and _INTERVAL_PAYLOAD_CACHE:
            _old_key, (_old_payload, old_size) = _INTERVAL_PAYLOAD_CACHE.popitem(last=False)
            _INTERVAL_PAYLOAD_CACHE_BYTES -= int(old_size)


def _read_interval_task_payload(path: Path) -> IntervalTask:
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("Current-run interval payloads must be HDF5 artifacts.")
    with h5py.File(path, "r") as data:
        if "half_space_role" not in data or "reciprocal_multiplicity" not in data:
            raise ValueError(
                "Current-run interval payloads must include half_space_role "
                "and reciprocal_multiplicity metadata."
            )
        element = data["element"][()]
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        q_grid_digest = None
        if "q_grid_digest" in data:
            q_grid_digest = data["q_grid_digest"][()]
            if isinstance(q_grid_digest, bytes):
                q_grid_digest = q_grid_digest.decode("ascii")
            else:
                q_grid_digest = str(q_grid_digest)
        half_space_role = normalize_half_space_role(data["half_space_role"][()])
        reciprocal_multiplicity = int(np.asarray(data["reciprocal_multiplicity"]).reshape(-1)[0])
        return IntervalTask(
            int(np.asarray(data["irecip_id"]).reshape(-1)[0]),
            str(element),
            np.asarray(data["q_grid"]),
            np.asarray(data["q_amp"]),
            np.asarray(data["q_amp_av"]),
            q_grid_digest=q_grid_digest,
            half_space_role=half_space_role,
            reciprocal_multiplicity=reciprocal_multiplicity,
        )


def load_interval_task_payload(
    interval_path: Path | str | IntervalTask | IntervalPayloadRef,
) -> IntervalTask:
    if isinstance(interval_path, IntervalTask):
        return interval_path
    expected_file_sha = None
    expected_interval_id = None
    expected_q_grid_digest = None
    if isinstance(interval_path, IntervalPayloadRef):
        path = Path(interval_path.path)
        expected_file_sha = str(interval_path.file_sha256)
        expected_interval_id = interval_path.interval_id
        expected_q_grid_digest = interval_path.q_grid_digest
    else:
        path = Path(interval_path)

    cache_key = None
    if expected_file_sha:
        cache_key = (str(path), expected_file_sha)
        cached = _cached_interval_payload(cache_key)
        if cached is not None:
            return cached
        actual_file_sha = file_sha256(path)
        if actual_file_sha != expected_file_sha:
            raise ValueError(
                "Interval payload file_sha256 mismatch: "
                f"expected={expected_file_sha} actual={actual_file_sha}"
            )

    interval_task = _read_interval_task_payload(path)
    if expected_interval_id is not None and int(interval_task.irecip_id) != int(expected_interval_id):
        raise ValueError(
            "Interval payload interval_id mismatch: "
            f"expected={int(expected_interval_id)} actual={int(interval_task.irecip_id)}"
        )
    if (
        expected_q_grid_digest
        and interval_task.q_grid_digest
        and str(interval_task.q_grid_digest) != str(expected_q_grid_digest)
    ):
        raise ValueError(
            "Interval payload q_grid_digest mismatch: "
            f"expected={expected_q_grid_digest} actual={interval_task.q_grid_digest}"
        )
    if cache_key is not None:
        _store_interval_payload_cache(cache_key, interval_task)
    return interval_task


def scattering_contribution_point_count(interval_task: IntervalTask) -> int:
    q_grid = interval_task.q_grid
    multiplicity = half_space_role_multiplicity(interval_task.half_space_role)
    return int(q_grid.shape[0]) * int(multiplicity)


def compute_scattering_interval_chunk_arrays(
    interval_task: IntervalTask,
    atoms: np.recarray,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> ScatteringChunkComputeResult:
    chunk_data = [
        {
            "coordinates": atoms["coordinates"][index],
            "dist_from_atom_center": atoms["dist_from_atom_center"][index],
            "step_in_frac": atoms["step_in_frac"][index],
        }
        for index in range(atoms.shape[0])
    ]
    rifft_grid, grid_shape_nd = build_rifft_grid_for_chunk(chunk_data)
    inverse_pair = execute_inverse_cunufft_batch_materialize_once(
        q_coords=interval_task.q_grid,
        weights=np.stack(
            [
                interval_task.q_amp - interval_task.q_amp_av,
                interval_task.q_amp_av,
            ],
            axis=0,
        ),
        real_coords=rifft_grid,
        eps=nufft_eps,
        prefer_cpu=nufft_prefer_cpu,
        gpu_only=nufft_gpu_only,
    )
    amplitudes_delta = apply_half_space_conjugate_reconstruction(
        inverse_pair[0],
        interval_task.q_grid,
        interval_task.half_space_role,
    )
    amplitudes_average = apply_half_space_conjugate_reconstruction(
        inverse_pair[1],
        interval_task.q_grid,
        interval_task.half_space_role,
    )
    return ScatteringChunkComputeResult(
        grid_shape_nd=grid_shape_nd,
        contribution_reciprocal_points=scattering_contribution_point_count(interval_task),
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
    )


def compute_scattering_interval_payload(
    interval: dict,
    *,
    B_: np.ndarray,
    mask_params: dict,
    MaskStrategy,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    use_coeff: bool,
    coeff_val: np.ndarray | None,
    unique_elements: list[str],
    ff_factory,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> IntervalTask | None:
    q_grid = generate_q_space_grid_sync(interval, B_, mask_params, MaskStrategy, supercell)
    if q_grid.size == 0:
        return None

    # One lattice plan serves every forward transform of this interval; the
    # underlying type-1 plan+setpts are additionally shared ACROSS intervals
    # (they depend only on the sources and the global lattice pitch).
    lattice_meta = build_interval_lattice_meta(q_grid)
    contributions: list[tuple] = []
    if use_coeff:
        contributions.append(
            compute_interval_coeff_contribution(
                interval,
                q_grid,
                coeff_val,
                original_coords,
                cells_origin,
                nufft_eps=nufft_eps,
                nufft_prefer_cpu=nufft_prefer_cpu,
                nufft_gpu_only=nufft_gpu_only,
                lattice_meta=lattice_meta,
            )
        )
    else:
        # the average-structure transform is identical for every element:
        # compute it once per interval instead of once per element
        shared_q_av = forward_interval_amplitudes(
            cells_origin,
            np.ones(original_coords.shape[0]),
            q_grid,
            lattice_meta=lattice_meta,
            nufft_eps=nufft_eps,
            nufft_prefer_cpu=nufft_prefer_cpu,
            nufft_gpu_only=nufft_gpu_only,
        )
        for element in unique_elements:
            contribution = compute_interval_element_contribution(
                interval,
                q_grid,
                element,
                original_coords,
                cells_origin,
                elements_arr,
                charge,
                ff_factory,
                nufft_eps=nufft_eps,
                nufft_prefer_cpu=nufft_prefer_cpu,
                nufft_gpu_only=nufft_gpu_only,
                lattice_meta=lattice_meta,
                q_av=shared_q_av,
            )
            if contribution is not None:
                contributions.append(contribution)

    if not contributions:
        return None

    interval_task = aggregate_interval_contributions(contributions, use_coeff=use_coeff)
    half_space_role = classify_interval_half_space_role(interval, supercell)
    reciprocal_multiplicity = int(half_space_role_multiplicity(half_space_role) or 1)
    return interval_task._replace(
        half_space_role=half_space_role,
        reciprocal_multiplicity=reciprocal_multiplicity,
    )


def run_scattering_interval_task(
    work_unit: ScatteringWorkUnit,
    interval: dict,
    *,
    B_: np.ndarray,
    mask_params: dict,
    MaskStrategy,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    use_coeff: bool,
    coeff_val: np.ndarray | None,
    unique_elements: list[str],
    ff_factory,
    output_dir: str,
    db_path: str | None = None,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> ScatteringArtifactManifest | None:
    if db_path is not None and is_interval_artifact_committed(work_unit, db_path=db_path):
        return build_scattering_interval_manifest(
            work_unit,
            completion_status=CompletionStatus.COMMITTED,
        )

    interval_task = compute_scattering_interval_payload(
        interval,
        B_=B_,
        mask_params=mask_params,
        MaskStrategy=MaskStrategy,
        supercell=supercell,
        original_coords=original_coords,
        cells_origin=cells_origin,
        elements_arr=elements_arr,
        charge=charge,
        use_coeff=use_coeff,
        coeff_val=coeff_val,
        unique_elements=unique_elements,
        ff_factory=ff_factory,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    if interval_task is None:
        # Mask eliminated all Q-points in this interval.  Mark it as
        # precomputed so that downstream consumers (chunk accumulation and
        # residual-field) do not attempt to load a non-existent interval artifact.
        if db_path is not None:
            mark_empty_interval_precomputed(
                work_unit.interval_id, db_path=db_path
            )
        return None
    return persist_precomputed_interval_artifact(work_unit, interval_task, db_path=db_path)


def run_scattering_interval_chunk_task(
    work_unit: ScatteringWorkUnit,
    interval_path: Path | str | IntervalTask | IntervalPayloadRef,
    atoms: np.recarray,
    *,
    total_reciprocal_points: int,
    output_dir: str,
    db_path: str | None = None,
    quiet_logs: bool = False,
    runtime_provenance: Mapping[str, Any] | None = None,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> ScatteringArtifactManifest | None:
    if quiet_logs:
        _lower_worker_log_levels()

    interval_id = work_unit.interval_id
    try:
        interval_task = load_interval_task_payload(interval_path)
        result = compute_scattering_interval_chunk_arrays(
            interval_task,
            atoms,
            nufft_eps=nufft_eps,
            nufft_prefer_cpu=nufft_prefer_cpu,
            nufft_gpu_only=nufft_gpu_only,
        )
        return persist_scattering_interval_chunk_shard(
            work_unit,
            grid_shape_nd=result.grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=result.contribution_reciprocal_points,
            amplitudes_delta=result.amplitudes_delta,
            amplitudes_average=result.amplitudes_average,
            output_dir=output_dir,
            quiet_logs=quiet_logs,
            runtime_provenance=runtime_provenance,
        )
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            int(work_unit.chunk_id) if work_unit.chunk_id is not None else -1,
            interval_id,
            err,
            exc_info=True,
        )
        handle_worker_gpu_failure(err, logger=logger)
        raise


__all__ = [
    "IntervalPayloadRef",
    "ScatteringChunkComputeResult",
    "clear_scattering_interval_payload_cache",
    "compute_scattering_interval_chunk_arrays",
    "compute_scattering_interval_payload",
    "load_interval_task_payload",
    "run_scattering_interval_chunk_task",
    "run_scattering_interval_task",
    "scattering_contribution_point_count",
]
