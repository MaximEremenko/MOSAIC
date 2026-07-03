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
        while _LATTICE_CACHE_BYTES > _lattice_cache_max_bytes() and len(_LATTICE_CACHE) > 1:
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
    """Zero-initialised grid storage: plain ndarray within the RAM budget, else
    an anonymous disk-backed memmap (file unlinked immediately, so it can never
    leak; space is reclaimed when the cache entry is garbage-collected)."""
    if in_ram:
        return np.zeros(shape, dtype=np.complex128)
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


def _build_lattice_groups_streaming(
    ordered_groups: "list[list[IntervalTask]]",
    *,
    snap_tol: float = 0.05,
) -> list | None:
    """Two-pass streaming build of per-role lattice grids.

    Pass 1 infers per-axis steps and global bounds; pass 2 scatter-adds each
    interval's [delta, average] weights into the dense grid. Never concatenates
    the raw points (hkl32: 539M points would need ~30 GB transient). Returns
    ``[(role, meta, grids)]`` or ``None`` if any point fails the lattice snap or
    the grid exceeds the host budget (caller falls back to type-3)."""
    host_budget = lattice_host_budget_bytes()
    result = []
    for grouped_tasks in ordered_groups:
        role = str(grouped_tasks[0].half_space_role)
        dq = None
        qmin = None
        qmax = None
        for task in grouped_tasks:
            q = np.asarray(task.q_grid, dtype=np.float64)
            if q.ndim != 2 or q.shape[1] not in (1, 2, 3) or len(q) == 0:
                return None
            steps = _infer_axis_steps(q)
            if dq is None:
                dq = steps
            else:
                # take the finest nonzero step seen on each axis
                both = (dq > 0) & (steps > 0)
                dq = np.where(both, np.minimum(dq, steps), np.maximum(dq, steps))
            lo, hi = q.min(0), q.max(0)
            qmin = lo if qmin is None else np.minimum(qmin, lo)
            qmax = hi if qmax is None else np.maximum(qmax, hi)
        dq = _refine_axis_steps(
            np.asarray(grouped_tasks[0].q_grid, dtype=np.float64), dq, qmin
        )
        dims = np.ones(len(dq), dtype=np.int64)
        active = dq > 0
        dims[active] = np.round((qmax[active] - qmin[active]) / dq[active]).astype(np.int64) + 1
        if np.any(dims > (1 << 20)):
            return None
        grid_points = int(np.prod(dims))
        if grid_points * 16 * 2 > host_budget:
            return None
        dims_t = tuple(int(v) for v in dims)
        grids = np.zeros((2, grid_points), dtype=np.complex128)
        for task in grouped_tasks:
            q = np.asarray(task.q_grid, dtype=np.float64)
            amp = np.asarray(task.q_amp, dtype=np.complex128).reshape(-1)
            av = np.asarray(task.q_amp_av, dtype=np.complex128).reshape(-1)
            if amp.shape[0] != len(q) or av.shape[0] != len(q):
                return None
            idx = np.zeros(q.shape, dtype=np.int64)
            for ax in range(len(dq)):
                if dq[ax] <= 0:
                    if np.abs(q[:, ax] - qmin[ax]).max() > 1e-9:
                        return None
                    continue
                ratio = (q[:, ax] - qmin[ax]) / dq[ax]
                ax_idx = np.round(ratio).astype(np.int64)
                if np.abs(ratio - ax_idx).max() > snap_tol:
                    return None
                if ax_idx.min() < 0 or ax_idx.max() >= dims_t[ax]:
                    return None
                idx[:, ax] = ax_idx
            flat = np.ravel_multi_index(tuple(idx.T), dims_t)
            np.add.at(grids[0], flat, amp - av)
            np.add.at(grids[1], flat, av)
        meta = {"origin": qmin.copy(), "dq": dq.copy(), "dims": dims_t, "snap_dev": 0.0}
        result.append((role, meta, grids.reshape((2,) + dims_t)))
    return result


def _execute_lattice_groups(
    groups: list,
    *,
    rifft_grid: np.ndarray,
    nufft_eps: float,
    nufft_prefer_cpu: bool,
    nufft_gpu_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes_delta = None
    amplitudes_average = None
    for role, meta, grids in groups:
        outs = execute_type2_on_lattice(
            meta,
            grids,
            rifft_grid,
            eps=nufft_eps,
            prefer_cpu=nufft_prefer_cpu,
            gpu_only=nufft_gpu_only,
        )
        d = apply_half_space_conjugate_reconstruction(outs[0], _LATTICE_EMPTY_Q, role)
        a = apply_half_space_conjugate_reconstruction(outs[1], _LATTICE_EMPTY_Q, role)
        amplitudes_delta = d if amplitudes_delta is None else amplitudes_delta + d
        amplitudes_average = a if amplitudes_average is None else amplitudes_average + a
    return amplitudes_delta, amplitudes_average


def _build_lattice_entry_from_inputs(
    loaded_interval_inputs,
    *,
    snap_tol: float = 0.05,
) -> dict | None:
    """Streaming (two-pass over files) lattice build straight from interval
    inputs: load -> extract bounds/role -> discard, then load -> scatter ->
    discard. Peak memory is the grids plus ONE interval payload, never the whole
    payload list (hkl32: ~17 GiB instead of ~55 GiB). Returns the cache entry or
    ``None`` when the data is lattice-ineligible."""
    host_budget = lattice_host_budget_bytes()

    def _load(interval_input):
        if isinstance(interval_input, IntervalTask):
            return interval_input
        return load_interval_task_payload(interval_input)

    per_role: dict[str, dict] = {}
    contribution = 0
    for interval_input in loaded_interval_inputs:
        task = _load(interval_input)
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
    if total_grid_bytes > host_budget and not _lattice_disk_spill_allowed(total_grid_bytes):
        return None
    for state in per_role.values():
        # Allocate with the final shape so a disk-spilled grid stays an
        # np.memmap instance (reshaping later would demote it to a plain
        # ndarray view and the cache would mis-count its bytes as RAM).
        state["grids"] = _allocate_lattice_grids(
            (2,) + state["dims"],
            in_ram=total_grid_bytes <= host_budget,
        )
    for interval_input in loaded_interval_inputs:
        task = _load(interval_input)
        role = str(task.half_space_role)
        state = per_role[role]
        q = np.asarray(task.q_grid, dtype=np.float64)
        dq = state["dq"]
        dims_t = state["dims"]
        idx = np.zeros(q.shape, dtype=np.int64)
        for ax in range(len(dq)):
            if dq[ax] <= 0:
                if np.abs(q[:, ax] - state["qmin"][ax]).max() > 1e-9:
                    return None
                continue
            ratio = (q[:, ax] - state["qmin"][ax]) / dq[ax]
            ax_idx = np.round(ratio).astype(np.int64)
            if np.abs(ratio - ax_idx).max() > snap_tol:
                return None
            if ax_idx.min() < 0 or ax_idx.max() >= dims_t[ax]:
                return None
            idx[:, ax] = ax_idx
        flat = np.ravel_multi_index(tuple(idx.T), dims_t)
        amp = np.asarray(task.q_amp, dtype=np.complex128).reshape(-1)
        av = np.asarray(task.q_amp_av, dtype=np.complex128).reshape(-1)
        flat_rows = state["grids"].reshape(2, -1)     # view; grids object keeps its type
        np.add.at(flat_rows[0], flat, amp - av)
        np.add.at(flat_rows[1], flat, av)
        del task, q, idx, flat, amp, av, flat_rows
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
    point_ids = int(point_start or 0) + np.arange(
        amplitudes_delta.shape[0], dtype=np.int64
    )
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
                built_entry = _build_lattice_entry_from_inputs(loaded_interval_inputs)
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
