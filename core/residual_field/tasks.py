from __future__ import annotations

import logging
import os
import sys
import threading
import time
import weakref
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
from core.scattering.accumulation import (
    apply_half_space_conjugate_reconstruction,
    half_space_conjugate_reconstruction_required,
)
from core.scattering.grid import _generate_grid
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
    execute_local_window_inverse,
    execute_type2_on_lattice,
    lattice_host_budget_bytes,
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


def _concat_cross_grid_sources_enabled() -> bool:
    """Concatenate ALL intervals' q-points (within a half-space role) into a single
    inverse type-3 transform, instead of one transform per distinct q_grid.

    An inverse type-3 NUFFT accepts arbitrary (non-grid) source points, and the
    transform is linear, so concatenating N intervals' (q, weight) pairs and
    transforming once is mathematically identical (to NUFFT eps) to transforming
    each interval separately and summing. The single call pays the expensive
    real-space (target) evaluation ONCE instead of once per interval, which is the
    dominant cost. Grouping stays per half-space role because the conjugate
    reconstruction differs by role.
    """
    raw = os.getenv("MOSAIC_RESIDUAL_CONCAT_SOURCES")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _residual_concat_fine_grid_budget_bytes() -> int:
    """Max cuFINUFFT type-3 fine grid a single concatenated source group may
    imply, before it is split into more groups. Concatenating *every* interval is
    a big win when the combined reciprocal extent stays small (2D / small cells),
    but for large cells + wide hkl the combined q-extent explodes the fine grid
    (``nf_i ~ real_extent_i * recip_extent_i``): 87 intervals over a 15^3 cell
    become a ~0.2 TiB grid that no longer fits any GPU. Bounding each group by
    this budget keeps every transform GPU-sized while still folding as many
    intervals together as safely possible. Override with
    ``MOSAIC_RESIDUAL_CONCAT_FINE_GRID_BUDGET`` (bytes)."""
    raw = os.getenv("MOSAIC_RESIDUAL_CONCAT_FINE_GRID_BUDGET")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    return 4 << 30


def _type3_fine_grid_from_spreads(real_spread: np.ndarray, q_spread: np.ndarray) -> int:
    """Fine-grid bytes from coordinate spreads (mirrors the wrapper's model)."""
    upsampfac = 2.0
    nf = np.ceil((upsampfac / np.pi) * np.asarray(real_spread) * np.asarray(q_spread)) + 32.0
    nf = np.ceil(nf / 16.0) * 16.0
    return int(np.prod(nf)) * 16


def _concat_global_fine_grid_max_bytes() -> int:
    """Above this whole-work-unit fine grid, concatenation is auto-disabled.

    Concatenating every interval is a large speed win when the combined fine grid
    stays modest (2D / small 3D cells), but for large cells + wide hkl the
    combined target x source extent makes the *global* fine grid enormous (hkl32:
    ~0.2 TiB). It then only survives via heavy per-group tiling, which inflates
    host memory and drives the run into the OOM-killer. Past this threshold the
    per-interval path (small independent grids, bounded memory) is the safe
    choice. Override with ``MOSAIC_RESIDUAL_CONCAT_MAX_GLOBAL_FINE_GRID`` (bytes)."""
    raw = os.getenv("MOSAIC_RESIDUAL_CONCAT_MAX_GLOBAL_FINE_GRID")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    return 48 << 30


def _concat_global_fine_grid_bytes(
    rifft_grid: np.ndarray, interval_tasks: Sequence[IntervalTask]
) -> int:
    """Fine grid a single transform over ALL targets and ALL concatenated q would
    imply -- the worst case the concat path builds toward before sub-grouping."""
    grid = np.asarray(rifft_grid, dtype=np.float64)
    if grid.ndim != 2 or len(grid) == 0 or not interval_tasks:
        return 0
    real_spread = grid.max(axis=0) - grid.min(axis=0)
    q_min = None
    q_max = None
    for task in interval_tasks:
        g = np.asarray(task.q_grid, dtype=np.float64)
        if g.ndim != 2 or len(g) == 0:
            continue
        gmin = g.min(axis=0)
        gmax = g.max(axis=0)
        q_min = gmin if q_min is None else np.minimum(q_min, gmin)
        q_max = gmax if q_max is None else np.maximum(q_max, gmax)
    if q_min is None:
        return 0
    return _type3_fine_grid_from_spreads(real_spread, q_max - q_min)


def _budget_bounded_concat_subgroups(
    role_tasks: list[IntervalTask],
    *,
    rifft_grid: np.ndarray,
    budget_bytes: int,
) -> list[list[IntervalTask]]:
    """Greedily pack same-role intervals into concat sub-groups whose combined
    type-3 fine grid stays within ``budget_bytes``.

    Intervals are visited in reciprocal-space order so spatially adjacent
    subvolumes fold together (their combined q-extent grows slowly). The exact
    same (q, weight) points are transformed and summed regardless of grouping --
    only *how many* fold into each GPU transform changes -- so the result is
    unchanged to NUFFT eps."""
    if len(role_tasks) <= 1:
        return [list(role_tasks)]
    grid = np.asarray(rifft_grid, dtype=np.float64)
    real_spread = grid.max(axis=0) - grid.min(axis=0)

    def _q_bounds(task: IntervalTask):
        g = np.asarray(task.q_grid, dtype=np.float64)
        return g.min(axis=0), g.max(axis=0)

    ordered = sorted(
        role_tasks,
        key=lambda t: tuple(float(v) for v in _q_bounds(t)[0]),
    )
    groups: list[list[IntervalTask]] = []
    current: list[IntervalTask] = []
    cur_min = cur_max = None
    for task in ordered:
        q_min, q_max = _q_bounds(task)
        new_min = q_min if cur_min is None else np.minimum(cur_min, q_min)
        new_max = q_max if cur_max is None else np.maximum(cur_max, q_max)
        fine = _type3_fine_grid_from_spreads(real_spread, new_max - new_min)
        if current and fine > budget_bytes:
            groups.append(current)
            current = [task]
            cur_min, cur_max = q_min, q_max
        else:
            current.append(task)
            cur_min, cur_max = new_min, new_max
    if current:
        groups.append(current)
    return groups


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


def _rifft_memmap_min_bytes() -> int:
    raw = os.getenv("MOSAIC_RESIDUAL_RIFFT_MEMMAP_MIN_BYTES")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    return 256 << 20


def _rifft_grid_file_key(atoms: np.recarray, work_unit: ResidualFieldWorkUnit) -> str:
    """Process-stable identity of a work unit's target grid (unlike the in-RAM
    cache key, which keys on object ids)."""
    import hashlib

    token = "|".join(
        str(part)
        for part in (
            str(work_unit.parameter_digest),
            int(work_unit.chunk_id),
            work_unit.partition_id,
            work_unit.point_start,
            work_unit.point_stop,
            int(getattr(atoms, "shape", (len(atoms),))[0]),
        )
    )
    return hashlib.sha256(token.encode("ascii")).hexdigest()[:24]


def _build_rifft_grid_bounded(
    chunk_data: list[dict],
    *,
    file_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a work unit's target grid with bounded host memory.

    Small grids take the original single-pass path. Large ones (hkl40: 5.08 GB
    per chunk, with a ~2x vstack peak on top) are built center-by-center
    straight into a scratch-backed ``.npy`` memmap: anonymous RSS stays at one
    patch (~2 MB) and the pages are evictable file cache. The file is written
    once per (digest, chunk, partition) and REUSED by every later work unit,
    resume, and worker on this host via atomic rename."""
    coords = np.array([point["coordinates"] for point in chunk_data])
    dist_vec = np.array([point["dist_from_atom_center"] for point in chunk_data])
    step_vec = np.array([point["step_in_frac"] for point in chunk_data])
    dim = coords.shape[1]

    counts: list[int] = []
    shapes: list[np.ndarray] = []
    for center, dist, step in zip(coords, dist_vec, step_vec):
        grid, shape = _generate_grid(dim, step, center, dist)
        counts.append(int(grid.shape[0]))
        shapes.append(np.asarray(shape))
        del grid
    total = int(sum(counts))
    grid_shape_nd = np.vstack(shapes).astype(np.int64)
    total_bytes = total * dim * 8

    if total_bytes <= _rifft_memmap_min_bytes():
        return build_rifft_grid_for_chunk(chunk_data)

    root = Path(_lattice_scratch_dir()) / "mosaic-rifft-grids"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"rifft-{file_key}.npy"
    if final.exists():
        try:
            existing = np.lib.format.open_memmap(str(final), mode="r")
            if existing.shape == (total, dim):
                return existing, grid_shape_nd
        except (OSError, ValueError):
            pass
    tmp = root / f"rifft-{file_key}.{uuid4().hex}.tmp.npy"
    mm = np.lib.format.open_memmap(
        str(tmp), mode="w+", dtype=np.float64, shape=(total, dim)
    )
    offset = 0
    for center, dist, step, count in zip(coords, dist_vec, step_vec, counts):
        grid, _shape = _generate_grid(dim, step, center, dist)
        mm[offset : offset + count] = grid
        offset += count
        del grid
    mm.flush()
    del mm
    os.replace(tmp, final)
    return np.lib.format.open_memmap(str(final), mode="r"), grid_shape_nd


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
    # A read-only memmap grid costs page cache, not anonymous RAM: charge it
    # nothing so large grids stay cached (evicting them would force a shape
    # recomputation pass on every miss).
    payload_bytes = int(
        sum(0 if isinstance(arr, np.memmap) else int(arr.nbytes) for arr in payload)
    )
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
    rifft_grid, grid_shape_nd = _build_rifft_grid_bounded(
        _atoms_to_chunk_data(partition_atoms),
        file_key=_rifft_grid_file_key(atoms, work_unit),
    )
    if not isinstance(rifft_grid, np.memmap):
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


def _residual_lattice_fft_enabled() -> bool:
    """Evaluate the residual inverse via scatter + type-2 on the reciprocal
    LATTICE instead of type-3 over scattered points.

    MOSAIC's q-points sit on a uniform per-axis lattice (h,k,l at multiples of
    1/N_cell); masks select a subset of lattice sites but never move points off
    it. Scattering the (masked) values onto the dense coefficient grid and
    running one type-2 per half-space role is identical to the summed type-3
    (to NUFFT eps; measured floor ~5e-6 set by float noise in the *stored* q
    coordinates) and removes the type-3 fine-grid blow-up entirely -- measured
    27x per transform on hkl32-scale shards. Eligibility is checked per work
    unit (lattice snap + memory budget); ineligible data falls back to type-3
    automatically."""
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_FFT")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# Cache of scattered lattice grids keyed by (parameter_digest, interval_ids).
# The grids depend only on the intervals' (q, amplitude) data, NOT on the
# chunk/partition, so every work unit of the same interval shard reuses ONE
# scatter (hkl32: 140 work units -> 1 scatter of 539M points instead of 140
# re-loads of ~38 GB each). Guarded by build-events so concurrent worker
# threads wait for the first build instead of duplicating it.
_LATTICE_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_LATTICE_CACHE_BYTES = 0
_LATTICE_CACHE_LOCK = threading.Lock()
_LATTICE_CACHE_BUILDING: dict[tuple, threading.Event] = {}
_LATTICE_EMPTY_Q = np.empty((0, 3))


def _lattice_cache_max_bytes() -> int:
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_CACHE_MAX_BYTES")
    try:
        return int(raw) if raw else 24 << 30
    except (TypeError, ValueError):
        return 24 << 30


def _lattice_entry_bytes(entry: dict) -> int:
    groups = entry.get("groups")
    if not groups:
        return 0
    # Disk-backed (memmap) grids do not consume the RAM cache budget -- their
    # pages live in the evictable OS page cache / on disk.
    return int(
        sum(
            grids.nbytes
            for _role, _meta, grids in groups
            if not isinstance(grids, np.memmap)
        )
    )


def _lattice_cache_get(key: tuple) -> dict | None:
    with _LATTICE_CACHE_LOCK:
        entry = _LATTICE_CACHE.get(key)
        if entry is not None:
            _LATTICE_CACHE.move_to_end(key)
        return entry


def _lattice_cache_max_entries() -> int:
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_CACHE_MAX_ENTRIES")
    try:
        return max(1, int(raw)) if raw else 8
    except (TypeError, ValueError):
        return 8


def _lattice_cache_store(key: tuple, entry: dict) -> None:
    global _LATTICE_CACHE_BYTES
    nbytes = _lattice_entry_bytes(entry)
    if nbytes > _lattice_cache_max_bytes():
        return
    with _LATTICE_CACHE_LOCK:
        old = _LATTICE_CACHE.pop(key, None)
        if old is not None:
            _LATTICE_CACHE_BYTES -= _lattice_entry_bytes(old)
        _LATTICE_CACHE[key] = entry
        _LATTICE_CACHE_BYTES += nbytes
        # Spilled (memmap) entries cost 0 RAM but hold unlinked-file DISK
        # space until evicted, so the byte budget alone would never evict
        # them and distinct keys could fill the scratch filesystem; the
        # entry-count cap bounds that.
        while (
            _LATTICE_CACHE_BYTES > _lattice_cache_max_bytes()
            or len(_LATTICE_CACHE) > _lattice_cache_max_entries()
        ) and len(_LATTICE_CACHE) > 1:
            _k, victim = _LATTICE_CACHE.popitem(last=False)
            _LATTICE_CACHE_BYTES -= _lattice_entry_bytes(victim)


def clear_residual_lattice_cache() -> None:
    global _LATTICE_CACHE_BYTES
    with _LATTICE_CACHE_LOCK:
        _LATTICE_CACHE.clear()
        _LATTICE_CACHE_BYTES = 0


def _lattice_scratch_dir() -> str:
    import tempfile

    return os.getenv("MOSAIC_RESIDUAL_LATTICE_SCRATCH") or tempfile.gettempdir()


# In-RAM dense lattice grids currently alive in this process. MemAvailable
# cannot see a grid whose pages have not been touched yet (np.zeros is lazy),
# so concurrent builds admitted back-to-back would each look at the same
# "available" number; this counter closes that window. Decremented by a
# weakref finalizer when the grid is garbage-collected.
_LATTICE_LIVE_RAM_BYTES = 0
_LATTICE_LIVE_RAM_LOCK = threading.Lock()


def _track_live_lattice_ram(grid: np.ndarray) -> None:
    global _LATTICE_LIVE_RAM_BYTES
    nbytes = int(grid.nbytes)

    def _release(n: int = nbytes) -> None:
        global _LATTICE_LIVE_RAM_BYTES
        with _LATTICE_LIVE_RAM_LOCK:
            _LATTICE_LIVE_RAM_BYTES -= n

    with _LATTICE_LIVE_RAM_LOCK:
        _LATTICE_LIVE_RAM_BYTES += nbytes
    weakref.finalize(grid, _release)


def _mem_available_bytes() -> int | None:
    """MemAvailable clamped to the cgroup/SLURM allocation (containers and
    HPC jobs see the whole host in /proc/meminfo); None where /proc is
    absent."""
    from core.runtime.cpu_resources import available_memory_bytes

    return available_memory_bytes()


def _lattice_ram_admission_fraction() -> float:
    """Per-PROCESS share of MemAvailable a build may claim.

    The live-grid ledger is per-process, so N worker processes each admit
    against the SAME MemAvailable reading — an explicit fraction f is
    effectively N*f host-wide (observed: 4 workers x 0.2 admitted ~0.8 of
    the box and the kernel OOM killer took workers down). Divide the
    configured/host default by the expected worker count."""
    from core.adapters.cunufft_wrapper import _expected_worker_count

    workers = max(1, int(_expected_worker_count()))
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_RAM_FRACTION")
    if raw is not None and str(raw).strip() != "":
        try:
            return min(1.0, max(0.02, float(raw) / workers))
        except (TypeError, ValueError):
            pass
    return min(1.0, max(0.05, 0.5 / workers))


def _lattice_grids_fit_in_ram(total_grid_bytes: int) -> bool:
    """RAM admission for a work unit's dense lattice grids (all roles, both
    transforms).

    Three criteria, all required:
    - the static host budget (``lattice_host_budget_bytes``);
    - a fraction (default 0.5, ``MOSAIC_RESIDUAL_LATTICE_RAM_FRACTION``) of
      MemAvailable *right now*, plus this process's already-live in-RAM grids.
      The static budget alone admitted hkl40's ~32 GiB grid on a 62 GiB box,
      where the rest of the pipeline pushed RSS past the ceiling and the
      OOM-killer SIGKILLed the run -- no exception, so no fallback could fire.
      The admission must therefore be right *a priori*;
    - the lattice entry cache budget (when caching is enabled): an in-RAM
      grid bigger than the cache cannot be stored, so every chunk would
      rebuild it -- and concurrent waiter threads that find no cached entry
      fall back to materialising the full interval payload list. A spilled
      memmap grid is exempt from the cache's RAM budget, making it shareable
      across all chunks, so oversized grids prefer the spill path.

    Rejection here does not lose the lattice path: callers fall through to
    the disk-backed memmap allocation, and only to type-3 when spill is
    disallowed too. Non-Linux hosts (no MemAvailable) keep the static-budget
    behaviour."""
    if total_grid_bytes > lattice_host_budget_bytes():
        return False
    cache_max = _lattice_cache_max_bytes()
    if cache_max > 0 and total_grid_bytes > cache_max:
        return False
    available = _mem_available_bytes()
    if available is None:
        return True
    with _LATTICE_LIVE_RAM_LOCK:
        live = _LATTICE_LIVE_RAM_BYTES
    # Cached grids' pages are resident, so MemAvailable already reflects
    # them; counting them in `live` as well would double-charge and starve
    # every later admission for as long as they sit in the cache. `live`
    # must only cover grids MemAvailable cannot see yet (allocated but
    # untouched pages).
    live = max(0, live - int(_LATTICE_CACHE_BYTES))
    return total_grid_bytes + live <= available * _lattice_ram_admission_fraction()


def _lattice_disk_spill_allowed(nbytes: int) -> bool:
    """May a grid too large for the host-RAM budget spill to a disk-backed
    memmap instead of falling back to type-3?

    Small-RAM machines get the SAME fast lattice path as large ones: the
    transform consumes the grid as contiguous row-slabs, so a memmap streams
    from disk on small boxes while the OS page cache makes it RAM-speed on
    large ones. Requires enough free scratch space (1.25x safety); disable with
    ``MOSAIC_RESIDUAL_LATTICE_DISK_SPILL=0``."""
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_DISK_SPILL")
    if raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import shutil

        free = shutil.disk_usage(_lattice_scratch_dir()).free
    except OSError:
        return False
    return nbytes * 1.25 < free


def _allocate_lattice_grids(shape: tuple, *, in_ram: bool) -> np.ndarray:
    """Zero-initialised grid storage: plain ndarray within the RAM admission
    (budget AND currently-available memory -- see ``_lattice_grids_fit_in_ram``),
    else an anonymous disk-backed memmap (file unlinked immediately, so it can
    never leak; space is reclaimed when the cache entry is garbage-collected)."""
    if in_ram:
        grid = np.zeros(shape, dtype=np.complex128)
        _track_live_lattice_ram(grid)
        return grid
    import tempfile

    fd, path = tempfile.mkstemp(prefix="mosaic_lattice_", suffix=".grid", dir=_lattice_scratch_dir())
    try:
        grid = np.memmap(path, dtype=np.complex128, mode="w+", shape=shape)
    finally:
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
    logger.info(
        "Residual-field lattice grid spilled to disk-backed memmap (%.1f GiB) "
        "in %s -- small-RAM host taking the same lattice path.",
        int(np.prod(shape)) * 16 / (1 << 30),
        _lattice_scratch_dir(),
    )
    return grid


def _infer_axis_steps(q: np.ndarray) -> np.ndarray:
    """Per-axis lattice step candidates; 0 marks a degenerate (single-plane) axis."""
    steps = np.zeros(q.shape[1])
    for ax in range(q.shape[1]):
        col = q[:, ax]
        span = float(col.max() - col.min())
        if span <= 1e-12:
            continue
        u = np.unique(np.round(col, 9))
        gaps = np.diff(u)
        gaps = gaps[gaps > max(1e-9, span * 1e-6)]
        steps[ax] = float(gaps.min()) if gaps.size else 0.0
    return steps


def _refine_axis_steps(q: np.ndarray, dq: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Least-squares refinement of the per-axis lattice step.

    The min-gap estimate carries the float noise of a single pair of points, so
    the index error grows linearly across the lattice (projected 2D data:
    ~2.5e-5 of a step by index ~1600). Fitting the step to ALL of one
    interval's points averages that noise down by ~sqrt(n), which directly
    reduces the phase-error floor of the lattice transform."""
    refined = dq.copy()
    for ax in range(q.shape[1]):
        if dq[ax] <= 0:
            continue
        rel = q[:, ax] - origin[ax]
        idx = np.round(rel / dq[ax])
        denom = float(np.dot(idx, idx))
        if denom <= 0:
            continue
        step = float(np.dot(idx, rel)) / denom
        if step > 0 and abs(step - dq[ax]) < 0.1 * dq[ax]:
            refined[ax] = step
    return refined


# Post-LSQ snap deviation ceiling shared by both lattice builders: truly
# on-lattice q (exact float64 products) refines to <=1e-9 of a step, while
# NEAR-lattice data (e.g. a cell sheared by ~1e-7 relative) sits around 1e-6.
# Snapping the latter silently corrupts the reconstructed field beyond NUFFT
# eps, so it must fall back to type-3 instead.
_LATTICE_SNAP_MAX_DEV = 1e-7

# Sentinel distinguishing "no safe storage right now" (RAM tight AND scratch
# full: retry later, do not cache) from "geometrically lattice-ineligible"
# (None: cache the negative marker for the shard).
_LATTICE_RESOURCE_DENIED = "lattice-resource-denied"


def _build_lattice_groups_streaming(
    ordered_groups: "list[list[IntervalTask]]",
    *,
    snap_tol: float = _LATTICE_SNAP_MAX_DEV,
) -> list | None:
    """Per-role lattice grids from already-loaded interval tasks.

    Thin wrapper over ``_build_lattice_entry_from_inputs`` so that BOTH lattice
    builders share one allocation policy: total-across-roles accounting, the
    availability-aware RAM admission, and the disk-spill fallback. (This
    builder previously allocated plain in-RAM grids with only a per-role
    static-budget check, which is how an hkl40-scale work unit could pass the
    gate and drive the host into the OOM-killer.) Returns ``[(role, meta,
    grids)]`` or ``None`` when the data is lattice-ineligible or no safe
    storage exists (caller falls back to type-3)."""
    tasks = [task for group in ordered_groups for task in group]
    if not tasks:
        return None
    entry = _build_lattice_entry_from_inputs(tasks, snap_tol=snap_tol)
    if entry is None or entry == _LATTICE_RESOURCE_DENIED:
        return None
    return entry["groups"]


_POINT_ID_CACHE: dict[tuple[int, int], np.ndarray] = {}
_POINT_ID_CACHE_LOCK = threading.Lock()


def _shared_point_ids(point_start: int, count: int) -> np.ndarray:
    """One shared, read-only id vector per (start, count) per process.

    Every streaming accumulator of a chunk stores the SAME
    ``start + arange(count)`` — building a fresh 8-byte-per-point array per
    accumulator held num_slots x num_chunks x 1.69 GB of anonymous RAM at
    hkl40 scale. Large vectors live in a scratch-backed memmap (page cache,
    not RSS); everything downstream treats them read-only."""
    key = (int(point_start), int(count))
    with _POINT_ID_CACHE_LOCK:
        cached = _POINT_ID_CACHE.get(key)
        if cached is not None:
            return cached
    nbytes = int(count) * 8
    if nbytes <= (64 << 20):
        ids = np.arange(key[0], key[0] + key[1], dtype=np.int64)
        ids.setflags(write=False)
    else:
        root = Path(_lattice_scratch_dir()) / "mosaic-point-ids"
        root.mkdir(parents=True, exist_ok=True)
        final = root / f"ids-{key[0]}-{key[1]}.npy"
        if not final.exists():
            tmp = root / f"ids-{key[0]}-{key[1]}.{uuid4().hex}.tmp.npy"
            mm = np.lib.format.open_memmap(
                str(tmp), mode="w+", dtype=np.int64, shape=(key[1],)
            )
            step = 16_777_216
            for offset in range(0, key[1], step):
                stop = min(offset + step, key[1])
                mm[offset:stop] = np.arange(
                    key[0] + offset, key[0] + stop, dtype=np.int64
                )
            mm.flush()
            del mm
            os.replace(tmp, final)
        ids = np.lib.format.open_memmap(str(final), mode="r")
    with _POINT_ID_CACHE_LOCK:
        _POINT_ID_CACHE.setdefault(key, ids)
        return _POINT_ID_CACHE[key]


def _stage1_prefetch_window() -> int:
    """In-builder stage-1 prefetch depth (payloads computed ahead, bounded).

    ``MOSAIC_STREAMING_STAGE1_PARALLEL`` keeps its historical meaning as THE
    stage-1 parallelism knob; default 4 (~230 MB of in-flight payloads)."""
    raw = os.getenv("MOSAIC_STREAMING_STAGE1_PARALLEL")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    return 4


def _result_memmap_min_bytes() -> int:
    raw = os.getenv("MOSAIC_RESIDUAL_RESULT_MEMMAP_MIN_BYTES")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    return 1 << 30


def _allocate_result_arrays(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-initialised (delta, average) result rows for one work unit.

    Small results stay plain ndarrays. Large ones (hkl40: 2 x 3.39 GB per
    chunk) become anonymous disk-backed memmaps under the lattice scratch dir
    -- their pages are evictable file cache instead of anonymous RSS, which is
    what lets several work units run concurrently inside one Dask worker
    memory budget. Files are unlinked immediately after creation, so the
    space can never leak past process exit."""
    total = 2 * int(n_points) * 16
    if total <= _result_memmap_min_bytes():
        return (
            np.zeros(int(n_points), dtype=np.complex128),
            np.zeros(int(n_points), dtype=np.complex128),
        )
    root = Path(_lattice_scratch_dir()) / "mosaic-residual-results"
    root.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    for tag in ("delta", "average"):
        path = root / f"result-{uuid4().hex}-{tag}.npy"
        mm = np.lib.format.open_memmap(
            str(path), mode="w+", dtype=np.complex128, shape=(int(n_points),)
        )
        try:
            os.unlink(path)
        except OSError:
            pass
        arrays.append(mm)
    return arrays[0], arrays[1]


def _execute_lattice_groups(
    groups: list,
    *,
    rifft_grid: np.ndarray,
    nufft_eps: float,
    nufft_prefer_cpu: bool,
    nufft_gpu_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every half-space role's type-2 and fold the tiles straight into one
    (delta, average) result pair.

    The transform streams target tiles through ``tile_consumer``, so the full
    ``(n_trans, n_targets)`` output of a role is never materialized; the
    conjugate half-space reconstruction (``v + conj(v)`` == ``2*Re(v)``) is
    applied per tile during the fold. Host cost is O(tile) plus the result
    pair itself, which spills to a scratch memmap when large."""
    n_tgt = int(np.asarray(rifft_grid).shape[0])
    amplitudes_delta, amplitudes_average = _allocate_result_arrays(n_tgt)
    for role, meta, grids in groups:
        conjugate_double = half_space_conjugate_reconstruction_required(
            _LATTICE_EMPTY_Q, role
        )

        def _fold_tile(t0: int, t1: int, tile_out, _double=conjugate_double):
            if _double:
                amplitudes_delta[t0:t1] += 2.0 * tile_out[0].real
                amplitudes_average[t0:t1] += 2.0 * tile_out[1].real
            else:
                amplitudes_delta[t0:t1] += tile_out[0]
                amplitudes_average[t0:t1] += tile_out[1]

        execute_type2_on_lattice(
            meta,
            grids,
            rifft_grid,
            eps=nufft_eps,
            prefer_cpu=nufft_prefer_cpu,
            gpu_only=nufft_gpu_only,
            tile_consumer=_fold_tile,
        )
    return amplitudes_delta, amplitudes_average


_SCATTER_INVALID = object()  # sentinel: interval failed lattice validation


def _lattice_scatter_workers(total_grid_bytes: int) -> int:
    """Stripe-parallel scatter width. Runtime-only knob — never part of
    any digest or work-unit identity. Small grids stay serial (thread
    fan-out overhead beats the win below ~256 MiB)."""
    raw = os.getenv("MOSAIC_RESIDUAL_LATTICE_SCATTER_THREADS")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if total_grid_bytes < (256 << 20):
        return 1
    from core.runtime.cpu_resources import available_cpu_count

    return max(1, min(8, available_cpu_count() // 8))


def _prepare_interval_scatter(task, state, snap_tol):
    """Index-prep for one interval (cast, snap, ravel, stable sort) —
    pure with respect to ``state`` (reads dq/dims/qmin only), so it can
    ride the prefetch pool instead of the single builder thread. Returns
    None for a mask-empty task, _SCATTER_INVALID on validation failure,
    else the sorted scatter arrays."""
    if task is None:
        return None
    role = str(task.half_space_role)
    q = np.asarray(task.q_grid, dtype=np.float64)
    dq = state["dq"]
    dims_t = state["dims"]
    idx = np.zeros(q.shape, dtype=np.int64)
    for ax in range(len(dq)):
        if dq[ax] <= 0:
            if np.abs(q[:, ax] - state["qmin"][ax]).max() > 1e-9:
                return _SCATTER_INVALID
            continue
        ratio = (q[:, ax] - state["qmin"][ax]) / dq[ax]
        ax_idx = np.round(ratio).astype(np.int64)
        if np.abs(ratio - ax_idx).max() > snap_tol:
            return _SCATTER_INVALID
        if ax_idx.min() < 0 or ax_idx.max() >= dims_t[ax]:
            return _SCATTER_INVALID
        idx[:, ax] = ax_idx
    flat = np.ravel_multi_index(tuple(idx.T), dims_t)
    amp = np.asarray(task.q_amp, dtype=np.complex128).reshape(-1)
    av = np.asarray(task.q_amp_av, dtype=np.complex128).reshape(-1)
    # Stable sort by cell: equal-cell entries keep their original relative
    # order, so the duplicate-summing order (np.add.at branch) is exactly
    # the serial code's; distinct cells are independent accumulators.
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    has_duplicates = flat_sorted.size > 1 and bool(
        np.any(flat_sorted[1:] == flat_sorted[:-1])
    )
    return {
        "role": role,
        "flat": flat_sorted,
        "delta": (amp - av)[order],
        "av": av[order],
        "has_duplicates": has_duplicates,
    }


def _apply_scatter_stripe(flat_rows, prepared, cell_lo, cell_hi):
    """Apply one interval's contributions for cells in [cell_lo, cell_hi).
    Cell-disjoint across stripe workers -> no write overlap; per-cell FP64
    accumulation order is identical to the serial loop."""
    flat = prepared["flat"]
    lo = int(np.searchsorted(flat, cell_lo, side="left"))
    hi = int(np.searchsorted(flat, cell_hi, side="left"))
    if lo >= hi:
        return
    seg = slice(lo, hi)
    if prepared["has_duplicates"]:
        np.add.at(flat_rows[0], flat[seg], prepared["delta"][seg])
        np.add.at(flat_rows[1], flat[seg], prepared["av"][seg])
    else:
        flat_rows[0][flat[seg]] += prepared["delta"][seg]
        flat_rows[1][flat[seg]] += prepared["av"][seg]


def _build_lattice_entry_from_inputs(
    loaded_interval_inputs,
    *,
    snap_tol: float = _LATTICE_SNAP_MAX_DEV,
) -> dict | None:
    """Streaming (two-pass over files) lattice build straight from interval
    inputs: load -> extract bounds/role -> discard, then load -> scatter ->
    discard. Peak memory is the grids plus ONE interval payload, never the whole
    payload list (hkl32: ~17 GiB instead of ~55 GiB). Returns the cache entry or
    ``None`` when the data is lattice-ineligible."""

    def _load(interval_input):
        if isinstance(interval_input, IntervalTask):
            return interval_input
        if callable(interval_input):
            # Streaming lazy loader: computes one payload on demand and may
            # return None for a mask-empty interval (an exact zero
            # contribution; the caller's ledger still records the id).
            return interval_input()
        return load_interval_task_payload(interval_input)

    def _iter_loaded(inputs):
        """Yield loaded payloads in order, computing up to K ahead.

        Stage-1 loads were strictly sequential here, so a shard's prologue ran
        on ONE of the machine's cores while every GPU idled (measured: 25+ min
        for a sparse-mask hkl40 shard). A bounded sliding window keeps peak
        memory at K payloads (~57 MB each) while overlapping the mask/q-grid/
        forward work across threads — safe now that ``set_cpu_only`` no longer
        nulls the shared CuPy handle under running threads."""
        window = _stage1_prefetch_window()
        if window <= 1:
            for interval_input in inputs:
                yield _load(interval_input)
            return
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=window) as pool:
            pending: deque = deque()
            iterator = iter(inputs)
            for interval_input in iterator:
                pending.append(pool.submit(_load, interval_input))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    per_role: dict[str, dict] = {}
    contribution = 0
    for task in _iter_loaded(loaded_interval_inputs):
        if task is None:
            continue
        q = np.asarray(task.q_grid, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] not in (1, 2, 3) or len(q) == 0:
            return None
        n_q = len(q)
        if (
            int(np.asarray(task.q_amp).reshape(-1).shape[0]) != n_q
            or int(np.asarray(task.q_amp_av).reshape(-1).shape[0]) != n_q
        ):
            return None
        contribution += scattering_contribution_point_count(task)
        role = str(task.half_space_role)
        steps = _infer_axis_steps(q)
        lo, hi = q.min(0), q.max(0)
        state = per_role.get(role)
        if state is None:
            per_role[role] = {"dq": steps, "qmin": lo, "qmax": hi, "sample_q": q.copy()}
        else:
            dq = state["dq"]
            both = (dq > 0) & (steps > 0)
            state["dq"] = np.where(both, np.minimum(dq, steps), np.maximum(dq, steps))
            state["qmin"] = np.minimum(state["qmin"], lo)
            state["qmax"] = np.maximum(state["qmax"], hi)
        del task, q
    if not per_role:
        return None
    total_grid_bytes = 0
    for role, state in per_role.items():
        # LSQ-refine the step against one representative interval so index
        # errors do not accumulate across the lattice (see _refine_axis_steps).
        state["dq"] = _refine_axis_steps(
            state.pop("sample_q"), state["dq"], state["qmin"]
        )
        dq = state["dq"]
        dims = np.ones(len(dq), dtype=np.int64)
        active = dq > 0
        dims[active] = (
            np.round((state["qmax"][active] - state["qmin"][active]) / dq[active]).astype(np.int64)
            + 1
        )
        if np.any(dims > (1 << 20)):
            return None
        state["dims"] = tuple(int(v) for v in dims)
        total_grid_bytes += int(np.prod(dims)) * 16 * 2
    in_ram = _lattice_grids_fit_in_ram(total_grid_bytes)
    if not in_ram and not _lattice_disk_spill_allowed(total_grid_bytes):
        # Resource denial (RAM tight AND scratch full) is TRANSIENT: it must
        # not be cached as a lattice-ineligible negative marker, or one bad
        # moment degrades every later work unit of the shard to type-3.
        return _LATTICE_RESOURCE_DENIED
    for state in per_role.values():
        # Allocate with the final shape so a disk-spilled grid stays an
        # np.memmap instance (reshaping later would demote it to a plain
        # ndarray view and the cache would mis-count its bytes as RAM).
        state["grids"] = _allocate_lattice_grids(
            (2,) + state["dims"],
            in_ram=in_ram,
        )
    # Pass 2: scatter. The serial form ran cast/snap/ravel/sort AND the
    # ~155M fancy-index adds per hkl40 shard on the ONE builder thread
    # while sibling work units parked on the build event. Now the index
    # prep rides the prefetch pool (_iter_prepared) and the adds fan out
    # over cell-disjoint stripes — bitwise-identical accumulation order.
    scatter_workers = _lattice_scatter_workers(total_grid_bytes)

    def _iter_prepared(inputs):
        window = _stage1_prefetch_window()

        def _prep(interval_input):
            task = _load(interval_input)
            if task is None:
                return None
            return _prepare_interval_scatter(
                task, per_role[str(task.half_space_role)], snap_tol
            )

        if window <= 1:
            for interval_input in inputs:
                yield _prep(interval_input)
            return
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=window) as pool:
            pending: deque = deque()
            for interval_input in inputs:
                pending.append(pool.submit(_prep, interval_input))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    if scatter_workers <= 1:
        for prepared in _iter_prepared(loaded_interval_inputs):
            if prepared is None:
                continue
            if prepared is _SCATTER_INVALID:
                return None
            state = per_role[prepared["role"]]
            flat_rows = state["grids"].reshape(2, -1)  # view; keeps grid type
            _apply_scatter_stripe(
                flat_rows, prepared, 0, int(np.prod(state["dims"])) + 1
            )
            del prepared, flat_rows
    else:
        from concurrent.futures import ThreadPoolExecutor

        invalid = False
        with ThreadPoolExecutor(max_workers=scatter_workers) as scatter_pool:
            for prepared in _iter_prepared(loaded_interval_inputs):
                if prepared is None:
                    continue
                if prepared is _SCATTER_INVALID:
                    # Pool context exit joins in-flight stripe jobs before
                    # the caller can cache (and thus leak) a partial entry.
                    invalid = True
                    break
                state = per_role[prepared["role"]]
                flat_rows = state["grids"].reshape(2, -1)
                n_cells = int(np.prod(state["dims"]))
                bounds = np.linspace(
                    0, n_cells + 1, scatter_workers + 1
                ).astype(np.int64)
                futures = [
                    scatter_pool.submit(
                        _apply_scatter_stripe,
                        flat_rows,
                        prepared,
                        int(bounds[w]),
                        int(bounds[w + 1]),
                    )
                    for w in range(scatter_workers)
                ]
                for future in futures:
                    future.result()
                del prepared, flat_rows, futures
        if invalid:
            return None
    groups = []
    for role in sorted(per_role):
        state = per_role[role]
        meta = {
            "origin": state["qmin"].copy(),
            "dq": state["dq"].copy(),
            "dims": state["dims"],
            "snap_dev": 0.0,
        }
        groups.append((role, meta, state["grids"]))
    return {"groups": groups, "contribution": int(contribution)}


def _compute_from_lattice_entry(
    entry: dict,
    *,
    rifft_grid: np.ndarray,
    grid_shape_nd: np.ndarray,
    point_start: int,
    nufft_eps: float,
    nufft_prefer_cpu: bool,
    nufft_gpu_only: bool,
) -> ResidualChunkComputeResult:
    """Run the residual transform from cached lattice grids (no interval loads)."""
    amplitudes_delta, amplitudes_average = _execute_lattice_groups(
        entry["groups"],
        rifft_grid=rifft_grid,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    point_ids = _shared_point_ids(int(point_start or 0), amplitudes_delta.shape[0])
    return ResidualChunkComputeResult(
        grid_shape_nd=grid_shape_nd,
        contribution_reciprocal_points=int(entry["contribution"]),
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
        point_ids=point_ids,
    )


def _residual_local_window_enabled() -> bool:
    """Evaluate the residual field in LOCAL window coordinates.

    Each target is ``center_atom + delta`` where the window offsets ``delta`` are
    identical across atoms (small, ~1 Angstrom extent). Factoring the per-atom
    centre phase out of ``exp(-i (center+delta).q)`` turns the transform into one
    batched inverse over the shared window grid, whose type-3 fine grid depends on
    the *window* extent, not the supercell extent. That collapses the
    ``(supercell x hkl)^3`` fine-grid blow-up of large 3D cells (hkl32: 0.2 TiB ->
    ~8 MiB) while giving an identical field to NUFFT eps. Off by default until the
    end-to-end result is validated against the global path on the target run."""
    raw = os.getenv("MOSAIC_RESIDUAL_LOCAL_WINDOW")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _local_window_phase_budget_bytes() -> int:
    """Cap on the per-tile ``(n_atoms x n_q)`` centre-phase matrix (bytes)."""
    raw = os.getenv("MOSAIC_RESIDUAL_LOCAL_PHASE_BUDGET")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1 << 20, int(raw))
        except (TypeError, ValueError):
            pass
    return 512 << 20


def _reconstruct_window_centers_offsets(
    rifft_grid: np.ndarray,
    grid_shape_nd: np.ndarray,
):
    """Recover ``(centers, shared_offsets)`` from an atom-major rifft grid.

    Returns ``None`` when the windows are not uniform across atoms (different
    shape or non-shared offsets), in which case the caller keeps the global
    transform. Relies on ``_generate_grid`` building ``pts = offsets + center``
    with ``offsets`` independent of the centre, and ``_process_chunk`` stacking
    per-atom windows in order (both verified in core.scattering.grid)."""
    grid = np.asarray(rifft_grid, dtype=np.float64)
    shapes = np.asarray(grid_shape_nd)
    if grid.ndim != 2 or shapes.ndim != 2 or len(shapes) == 0:
        return None
    if not bool(np.all(shapes == shapes[0])):
        return None
    window = int(np.prod(shapes[0]))
    n_atoms = int(len(shapes))
    if window <= 0 or len(grid) != n_atoms * window:
        return None
    blocks = grid.reshape(n_atoms, window, grid.shape[1])
    centers = blocks.mean(axis=1)
    offsets = blocks[0] - centers[0]
    # Offsets must genuinely be shared across atoms for the factorisation to hold.
    if not np.allclose(blocks, centers[:, None, :] + offsets[None, :, :], rtol=0, atol=1e-6):
        return None
    return centers, offsets


def _local_window_inverse(
    q_coords: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    *,
    eps: float,
    prefer_cpu: bool,
    gpu_only: bool,
) -> np.ndarray:
    """Local-window inverse type-3, batched over atoms (see
    :func:`core.adapters.cunufft_wrapper.execute_local_window_inverse`). Returns
    ``(n_rows, n_atoms*n_win)`` in atom-major order, identical to the global
    ``_call_inverse_super_batch`` output to NUFFT eps."""
    return execute_local_window_inverse(
        q_coords,
        weights,
        offsets,
        centers,
        eps=eps,
        prefer_cpu=prefer_cpu,
        gpu_only=gpu_only,
    )


def compute_residual_field_interval_chunk_arrays(
    interval_tasks: Sequence[IntervalTask],
    *,
    rifft_grid: np.ndarray,
    grid_shape_nd: np.ndarray,
    point_start: int = 0,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
    lattice_sink: dict | None = None,
    lattice_enabled: bool | None = None,
) -> ResidualChunkComputeResult:
    ordered_interval_tasks = sorted(interval_tasks, key=_interval_task_sort_key)
    concat_sources = _concat_cross_grid_sources_enabled()
    if lattice_enabled is None:
        lattice_enabled = _residual_lattice_fft_enabled()
    if concat_sources and not lattice_enabled:
        # Auto-disable concat for large-cell / wide-hkl work units, where the
        # whole-work-unit type-3 fine grid is enormous and the concat path
        # drives host memory into the OOM-killer. Only relevant when the
        # lattice (scatter+type-2) path is off -- the lattice path has no
        # type-3 fine grid, and when it turns out ineligible at build time the
        # same guard is re-applied before the type-3 fallback below.
        global_fine = _concat_global_fine_grid_bytes(rifft_grid, ordered_interval_tasks)
        if global_fine > _concat_global_fine_grid_max_bytes():
            concat_sources = False
            logger.info(
                "Residual-field: disabling source concat for this work unit "
                "(whole-unit fine grid ~%.1f GiB > %.1f GiB cap); using the "
                "memory-safe per-interval path.",
                global_fine / (1 << 30),
                _concat_global_fine_grid_max_bytes() / (1 << 30),
            )
    grouped_interval_tasks: dict[tuple, list[IntervalTask]] = {}
    contribution_reciprocal_points = 0
    for interval_task in ordered_interval_tasks:
        # When concatenating, all intervals sharing a half-space role fold into ONE
        # inverse transform regardless of their (distinct) q_grids; otherwise keep the
        # legacy per-q_grid grouping that the pre-sum / same-grid stacking requires.
        group_key = (
            (str(interval_task.half_space_role),)
            if concat_sources
            else (
                _q_grid_signature(
                    interval_task.q_grid,
                    getattr(interval_task, "q_grid_digest", None),
                ),
                interval_task.half_space_role,
            )
        )
        grouped_interval_tasks.setdefault(group_key, []).append(interval_task)
        contribution_reciprocal_points += scattering_contribution_point_count(interval_task)
    if not grouped_interval_tasks:
        raise ValueError("Residual-field batch task produced no interval contributions.")

    amplitudes_delta = None
    amplitudes_average = None
    use_presum = _same_q_grid_presum_enabled() and not concat_sources
    ordered_groups = [
        grouped_interval_tasks[key]
        for key in sorted(grouped_interval_tasks, key=str)
    ]
    # When enabled, evaluate the field in local window coordinates (identical
    # result, tiny fine grid). Reconstruct the shared centres/offsets once;
    # None -> non-uniform windows -> keep the global transform.
    local_window = (
        _reconstruct_window_centers_offsets(rifft_grid, grid_shape_nd)
        if _residual_local_window_enabled()
        else None
    )
    if concat_sources and local_window is None and lattice_enabled:
        # Lattice path: scatter each role's (masked) q-values onto the dense
        # reciprocal lattice and run ONE type-2 per role. Identical to the
        # summed type-3 to NUFFT eps; no type-3 fine grid, so no budget split.
        lattice_groups = _build_lattice_groups_streaming(
            [sorted(group, key=_interval_task_sort_key) for group in ordered_groups]
        )
        if lattice_groups is not None:
            amplitudes_delta, amplitudes_average = _execute_lattice_groups(
                lattice_groups,
                rifft_grid=rifft_grid,
                nufft_eps=nufft_eps,
                nufft_prefer_cpu=nufft_prefer_cpu,
                nufft_gpu_only=nufft_gpu_only,
            )
            if lattice_sink is not None:
                lattice_sink["groups"] = lattice_groups
                lattice_sink["contribution"] = contribution_reciprocal_points
            ordered_groups = []          # transform done; skip the type-3 loop
        else:
            logger.info(
                "Residual-field lattice path ineligible for this work unit; "
                "falling back to type-3."
            )
            # Re-apply the large-problem concat guard for the type-3 fallback.
            global_fine = _concat_global_fine_grid_bytes(rifft_grid, ordered_interval_tasks)
            if global_fine > _concat_global_fine_grid_max_bytes():
                concat_sources = False
                use_presum = _same_q_grid_presum_enabled()
                regrouped: dict[tuple, list[IntervalTask]] = {}
                for interval_task in ordered_interval_tasks:
                    key = (
                        _q_grid_signature(
                            interval_task.q_grid,
                            getattr(interval_task, "q_grid_digest", None),
                        ),
                        interval_task.half_space_role,
                    )
                    regrouped.setdefault(key, []).append(interval_task)
                ordered_groups = [regrouped[key] for key in sorted(regrouped, key=str)]
    if concat_sources and local_window is None and ordered_groups:
        # Global path only: split each half-space role's concatenation into
        # fine-grid-bounded sub-groups so no single transform blows past a
        # GPU-sized fine grid on large-cell / wide-hkl runs (summed across
        # sub-groups -> identical to one giant concat to NUFFT eps). The local
        # path's fine grid is set by the window extent, not the supercell, so it
        # needs no such split -- concatenating every interval minimises the number
        # of batched transforms.
        budget = _residual_concat_fine_grid_budget_bytes()
        ordered_groups = [
            subgroup
            for group in ordered_groups
            for subgroup in _budget_bounded_concat_subgroups(
                group, rifft_grid=rifft_grid, budget_bytes=budget
            )
        ]
    for grouped_tasks in ordered_groups:
        grouped_tasks = sorted(grouped_tasks, key=_interval_task_sort_key)
        if concat_sources:
            # One type-3 over the concatenation of every interval's q-points. The
            # transform is linear so this equals summing per-interval transforms
            # (to NUFFT eps) but pays the target-side cost only once.
            for task in grouped_tasks:
                q_len = int(np.asarray(task.q_grid).shape[0])
                if int(np.asarray(task.q_amp).reshape(-1).shape[0]) != q_len:
                    raise ValueError(
                        "Residual-field interval q_amp length does not match q_grid: "
                        f"interval={int(task.irecip_id)} "
                        f"q_amp={int(np.asarray(task.q_amp).reshape(-1).shape[0])} q_grid={q_len}"
                    )
                if int(np.asarray(task.q_amp_av).reshape(-1).shape[0]) != q_len:
                    raise ValueError(
                        "Residual-field interval q_amp_av length does not match q_grid: "
                        f"interval={int(task.irecip_id)} "
                        f"q_amp_av={int(np.asarray(task.q_amp_av).reshape(-1).shape[0])} q_grid={q_len}"
                    )
            reference_q_grid = np.concatenate(
                [np.asarray(task.q_grid) for task in grouped_tasks], axis=0
            )
            delta = np.concatenate(
                [
                    np.asarray(task.q_amp, dtype=np.complex128).reshape(-1)
                    - np.asarray(task.q_amp_av, dtype=np.complex128).reshape(-1)
                    for task in grouped_tasks
                ]
            )
            average = np.concatenate(
                [
                    np.asarray(task.q_amp_av, dtype=np.complex128).reshape(-1)
                    for task in grouped_tasks
                ]
            )
            inverse_weights = np.stack([delta, average], axis=0)
            del delta, average
        else:
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
        if local_window is not None:
            inverse_outputs = _local_window_inverse(
                reference_q_grid,
                inverse_weights,
                local_window[0],
                local_window[1],
                eps=nufft_eps,
                prefer_cpu=nufft_prefer_cpu,
                gpu_only=nufft_gpu_only,
            )
        else:
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
        if concat_sources or use_presum:
            if inverse_outputs.shape[0] != 2:
                raise ValueError(
                    "Residual-field inverse expected two output transforms; "
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
    point_ids = _shared_point_ids(int(point_start or 0), amplitudes_delta.shape[0])
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
    streaming_compute_context=None,
) -> ResidualFieldShardManifest | ResidualFieldAccumulatorStatus | None:
    _ensure_worker_logging()
    interval_ids = work_unit.interval_ids or ((work_unit.interval_id,) if work_unit.interval_id is not None else ())
    try:
        show_progress = _task_progress_enabled(quiet_logs)
        resolved_backend = reducer_backend or build_residual_field_reducer_backend(
            "local_restartable"
        )
        if streaming_compute_context is not None:
            # Fused stage-1: this task computes its batch's scattering
            # payloads itself (after the durable-skip check below), so no
            # interval inputs arrive from the caller.
            if not owner_local_reducer:
                raise ValueError(
                    "Streaming residual work units require the owner-local reducer."
                )
            loaded_interval_inputs = ()
        else:
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
        streamed_inputs_ready = streaming_compute_context is None

        def _resolve_interval_inputs():
            """This batch's stage-1 inputs, as LAZY per-interval loaders.

            Two layers of laziness, both load-bearing for memory:
            (1) nothing is built at all unless a consumer actually needs the
                payloads (a lattice-cache hit consumes only the cached entry —
                eager compute here built ~3.7 GB per unit and freed it unread
                for every fold after the first, num_chunks times per shard);
            (2) consumers receive zero-arg loaders, so the two-pass lattice
                builder holds ONE ~57 MB payload at a time instead of the
                whole shard's multi-GB list.
            """
            nonlocal loaded_interval_inputs, streamed_inputs_ready
            if not streamed_inputs_ready:
                from core.scattering.streaming import lazy_streamed_interval_loaders

                loaded_interval_inputs = lazy_streamed_interval_loaders(
                    interval_ids,
                    streaming_compute_context,
                    nufft_eps=nufft_eps,
                    nufft_prefer_cpu=nufft_prefer_cpu,
                    nufft_gpu_only=nufft_gpu_only,
                )
                streamed_inputs_ready = True
            return loaded_interval_inputs

        def _record_mask_empty_batch():
            """Every interval in this batch is mask-empty: its exact
            contribution is zero, but the accumulator must still record the
            batch as incorporated or finalize's coverage check would
            (correctly) refuse to publish the chunk."""
            n_points = int(rifft_grid.shape[0])
            worker_backend.accept_local_contribution(
                work_unit,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                contribution_reciprocal_points=0,
                amplitudes_delta=np.zeros(n_points, dtype=np.complex128),
                amplitudes_average=np.zeros(n_points, dtype=np.complex128),
                point_ids=_shared_point_ids(
                    int(work_unit.point_start or 0), n_points
                ),
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
                contribution_reciprocal_point_count=0,
                total_reciprocal_points=total_reciprocal_points,
            )
        if show_progress:
            logger.debug(
                "Residual batch start | chunk=%d | partition=%s | intervals=%s | rifft_points=%d",
                int(work_unit.chunk_id),
                work_unit.partition_id,
                ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a",
                int(rifft_grid.shape[0]),
            )
        # Lattice grid cache: the scattered coefficient grids depend only on the
        # interval data, not on the chunk/partition, so every work unit of the
        # same interval shard reuses one scatter and SKIPS re-loading the
        # interval payloads (hkl32: 140 work units x ~38 GB of interval reads
        # otherwise). Concurrent threads wait on the first build via an event.
        lattice_key = (
            (str(work_unit.parameter_digest), tuple(int(i) for i in interval_ids))
            if interval_ids and _residual_lattice_fft_enabled()
            else None
        )
        cached_entry = None
        lattice_builder = False
        build_event = None
        if lattice_key is not None:
            cached_entry = _lattice_cache_get(lattice_key)
            if cached_entry is None:
                with _LATTICE_CACHE_LOCK:
                    build_event = _LATTICE_CACHE_BUILDING.get(lattice_key)
                    if build_event is None:
                        build_event = threading.Event()
                        _LATTICE_CACHE_BUILDING[lattice_key] = build_event
                        lattice_builder = True
                if not lattice_builder:
                    build_event.wait(timeout=3600)
                    cached_entry = _lattice_cache_get(lattice_key)
        try:
            if cached_entry is None and lattice_builder:
                # Streaming build straight from the interval files: peak memory
                # is the grids + ONE payload, never the full payload list. The
                # result -- or a negative marker for lattice-ineligible data --
                # is cached for every later work unit of this shard.
                streamed_inputs = _resolve_interval_inputs()
                if streaming_compute_context is not None and not streamed_inputs:
                    return _record_mask_empty_batch()
                built_entry = _build_lattice_entry_from_inputs(streamed_inputs)
                if built_entry == _LATTICE_RESOURCE_DENIED:
                    # Transient storage denial: fall back to type-3 for THIS
                    # work unit only; later work units retry the build.
                    built_entry = None
                else:
                    _lattice_cache_store(
                        lattice_key,
                        built_entry
                        if built_entry is not None
                        else {"groups": None, "contribution": 0},
                    )
                cached_entry = built_entry
            if cached_entry is not None and cached_entry.get("groups"):
                compute_result = _compute_from_lattice_entry(
                    cached_entry,
                    rifft_grid=rifft_grid,
                    grid_shape_nd=grid_shape_nd,
                    point_start=int(work_unit.point_start or 0),
                    nufft_eps=nufft_eps,
                    nufft_prefer_cpu=nufft_prefer_cpu,
                    nufft_gpu_only=nufft_gpu_only,
                )
            else:
                streamed_inputs = _resolve_interval_inputs()
                materialized: list[IntervalTask] = []
                for interval_input in streamed_inputs:
                    if isinstance(interval_input, IntervalTask):
                        materialized.append(interval_input)
                        continue
                    if callable(interval_input):
                        loaded = interval_input()
                        if loaded is None:
                            continue
                        materialized.append(loaded)
                        continue
                    materialized.append(load_interval_task_payload(interval_input))
                if streaming_compute_context is not None and not materialized:
                    # Every interval in the batch is mask-empty: exact zero
                    # contribution, recorded so coverage still closes.
                    return _record_mask_empty_batch()
                interval_tasks = sorted(materialized, key=_interval_task_sort_key)
                compute_result = compute_residual_field_interval_chunk_arrays(
                    interval_tasks,
                    rifft_grid=rifft_grid,
                    grid_shape_nd=grid_shape_nd,
                    point_start=int(work_unit.point_start or 0),
                    nufft_eps=nufft_eps,
                    nufft_prefer_cpu=nufft_prefer_cpu,
                    nufft_gpu_only=nufft_gpu_only,
                    # A cached negative marker means this data already proved
                    # lattice-ineligible -- skip re-attempting it in compute.
                    lattice_enabled=False if lattice_key is not None else None,
                )
        finally:
            if lattice_builder:
                with _LATTICE_CACHE_LOCK:
                    _LATTICE_CACHE_BUILDING.pop(lattice_key, None)
                build_event.set()
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
