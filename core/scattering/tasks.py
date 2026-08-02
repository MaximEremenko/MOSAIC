from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from core.scattering.half_space import (
    classify_interval_half_space_role,
    half_space_role_multiplicity,
    normalize_half_space_role,
)
from core.scattering.artifacts import (
    build_scattering_interval_manifest,
    discard_stale_interval_artifact,
    is_interval_artifact_committed,
    mark_empty_interval_precomputed,
    persist_precomputed_interval_artifact,
)
from core.scattering.contracts import ScatteringArtifactManifest, ScatteringWorkUnit
from core.scattering.interval_payload import PAYLOAD_MISS, read_interval_payload
from core.scattering.kernels import (
    IntervalTask,
    aggregate_interval_contributions,
    build_interval_lattice_meta,
    compute_interval_coeff_contribution,
    compute_interval_element_contribution,
    forward_interval_amplitudes,
    generate_q_space_grid_sync,
)
from core.contracts import CompletionStatus
from core.runtime.budgeted_cache import BudgetedLRU
from core.storage.fingerprint import file_sha256


logger = logging.getLogger(__name__)
_INTERVAL_PAYLOAD_CACHE_MAX_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class IntervalPayloadRef:
    path: str
    file_sha256: str
    interval_id: int | None = None
    q_grid_digest: str | None = None


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
    _INTERVAL_PAYLOAD_CACHE.clear()


def _interval_task_nbytes(interval_task: IntervalTask) -> int:
    # Memmap-backed arrays cost evictable page cache, not anonymous RAM:
    # charge them nothing so spilled payloads never evict RAM-resident ones.
    return int(
        sum(
            np.asarray(value).nbytes
            for value in (interval_task.q_grid, interval_task.q_amp, interval_task.q_amp_av)
            if not isinstance(value, np.memmap)
        )
    )


_INTERVAL_PAYLOAD_CACHE = BudgetedLRU(
    max_bytes=_interval_payload_cache_max_bytes,
    size_fn=_interval_task_nbytes,
)


def _cached_interval_payload(key: tuple[str, str]) -> IntervalTask | None:
    if not _interval_payload_cache_enabled():
        return None
    return _INTERVAL_PAYLOAD_CACHE.get(key)


def _store_interval_payload_cache(key: tuple[str, str], payload: IntervalTask) -> None:
    if not _interval_payload_cache_enabled():
        return
    if _interval_payload_cache_max_bytes() <= 0:
        return
    _INTERVAL_PAYLOAD_CACHE.store(key, payload)


def _read_interval_task_payload(path: Path) -> IntervalTask:
    if path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("Current-run interval payloads must be HDF5 artifacts.")
    # Materializing read (mmap is the streaming store's concern): this
    # payload feeds the precompute-mode consumers, whose write access to
    # the arrays a read-only mapping would break.
    payload = read_interval_payload(path)
    if payload is None or payload is PAYLOAD_MISS:
        raise ValueError(f"Interval payload is missing or empty: {path}")
    return payload


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
    payload_identity: str | None = None,
) -> ScatteringArtifactManifest | None:
    if db_path is not None and is_interval_artifact_committed(
        work_unit, db_path=db_path, payload_identity=payload_identity
    ):
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
        # "Non-existent" is only true if a PREVIOUS run's artifact is cleared:
        # the residual stage loads that path without an identity check.
        discard_stale_interval_artifact(work_unit)
        if db_path is not None:
            mark_empty_interval_precomputed(
                work_unit.interval_id, db_path=db_path
            )
        return None
    return persist_precomputed_interval_artifact(
        work_unit,
        interval_task,
        db_path=db_path,
        payload_identity=payload_identity,
    )


__all__ = [
    "IntervalPayloadRef",
    "clear_scattering_interval_payload_cache",
    "compute_scattering_interval_payload",
    "load_interval_task_payload",
    "run_scattering_interval_task",
    "scattering_contribution_point_count",
]
