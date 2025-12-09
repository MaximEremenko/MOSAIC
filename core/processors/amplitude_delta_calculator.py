"""
Refactored amplitude-delta calculator
------------------------------------

Drop-in replacement for processors/amplitude_delta_calculator.py that keeps
ALL original behaviour but structures the work as a clean two-stage pipeline.
"""

from __future__ import annotations

import atexit
import gc
import inspect
import logging
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

import cupy as cp
import numpy as np
from dask import delayed
from dask.distributed import (
    Client,
    Lock,
    WorkerPlugin,
    as_completed,
)

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from managers.database_manager import DatabaseManager
from processors.point_data_processor import PointDataProcessor
from utilities.cunufft_wrapper import (
    execute_cunufft,
    execute_inverse_cunufft,
    set_cpu_only,
)

# ──────────────────────────────────────────────────────────────────────────────
# Globals & helpers
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
TIMER = time.perf_counter  # monotonic clock alias
DEFAULT_TASK_RETRIES = 4

def _is_sync_client(client) -> bool:
    """True when dask.distributed SyncClient or anything without an asyncio loop."""
    try:
        if client is None:
            return True
        cls = type(client).__name__.lower()
        loop = getattr(client, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)
        return ("syncclient" in cls) or (not has_loop)
    except Exception:
        return True

class _NoopLock:
    """Context manager that does nothing (used in single-threaded debug mode)."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False

def _safe_chunk_lock(name: str):
    """
    Return a distributed Lock when we have a real Dask client with an asyncio loop;
    otherwise return a no-op lock to avoid SyncClient crashes.
    """
    try:
        # Import lazily so this file also works without distributed installed
        from distributed import get_client
        from dask.distributed import Lock as _DaskLock

        try:
            c = get_client()  # works on workers and on driver
        except Exception:
            # No active client: single-threaded or debug
            return _NoopLock()

        # SyncClient has no asyncio loop; some local clients may also have loop=None
        loop = getattr(c, "loop", None)
        has_loop = (loop is not None) and (getattr(loop, "asyncio_loop", None) is not None)

        cls = type(c).__name__.lower()
        if "syncclient" in cls or not has_loop:
            return _NoopLock()

        # Real distributed client with event loop: use cluster-wide lock
        return _DaskLock(name, client=c)

    except Exception:
        # Any import/lookup problem → safest fallback in debug/sync mode
        return _NoopLock()

def chunk_mutex(chunk_id: int):
    return _safe_chunk_lock(f"chunk-{chunk_id}")

def _yield_futures_with_results(futs: Iterable, client: Client | None):
    """Yield (future, ok_bool) for each completed future, bound to a given client loop."""
    loop = getattr(client, "loop", None)
    for f, res in as_completed(futs, with_results=True, loop=loop):
        ok = False
        try:
            ok = bool(res)
        except Exception:
            ok = False
        yield f, ok

@contextmanager
def _timed(label: str):
    t0 = TIMER()
    try:
        yield
    finally:
        logger.info("%s took %.3f s", label, TIMER() - t0)

def _tqdm(total: int, *, desc: str, unit: str):
    """tqdm configured to avoid ‘0interval’ and to auto-disable when useless."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0,
        miniters=1,
        mininterval=0.1,
        leave=True,
        disable=(total <= 0 or not sys.stderr.isatty()),
    )

@contextmanager
def _quiet_db_info():
    """
    Temporarily raise DatabaseManager log level during tqdm progress to prevent
    INFO lines ('schema ready', 'connection closed') from tearing the bar.
    """
    names = ("managers.database_manager", "DatabaseManager")
    logs = [logging.getLogger(n) for n in names]
    prev = [lg.level for lg in logs]
    try:
        for lg in logs:
            lg.setLevel(max(logging.WARNING, lg.level))
        yield
    finally:
        for lg, lvl in zip(logs, prev):
            lg.setLevel(lvl)

# ──────────────────────────────────────────────────────────────────────────────
#  Worker-side clean-up (silences leaked-shared_memory warnings)
# ──────────────────────────────────────────────────────────────────────────────

def _final_cleanup():
    """Free CuPy pools *and* unlink any remaining shared-memory blocks."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass  # CPU-only worker or CuPy not imported

    try:
        from multiprocessing import resource_tracker, shared_memory

        for shm_name in list(shared_memory._SHARED_MEMORY_BLOCKS):
            try:
                shared_memory.SharedMemory(name=shm_name).unlink()
            except FileNotFoundError:
                pass
            resource_tracker.unregister(shm_name, "shared_memory")
    except Exception:
        pass


class CuPyCleanup(WorkerPlugin):
    """Run `_final_cleanup` when a worker process shuts down."""
    name = "cupy-cleanup"

    def teardown(self, worker):  # noqa: D401
        _final_cleanup()


atexit.register(_final_cleanup)


# ──────────────────────────────────────────────────────────────────────────────
#  Small helper structures & utilities
# ──────────────────────────────────────────────────────────────────────────────

class IntervalTask(NamedTuple):
    """Cached data for a reciprocal-space interval after the heavy NUFFT pass."""
    irecip_id: int
    element: str
    q_grid: np.ndarray
    q_amp: np.ndarray
    q_amp_av: np.ndarray


def _to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    """Translate flexible JSON/HDF5 interval to flat {h_start, h_end, …}."""
    out: Dict[str, float] = {}
    for ax in ("h", "k", "l"):
        rng = iv.get(f"{ax}_range")
        if rng is not None:
            out[f"{ax}_start"], out[f"{ax}_end"] = rng
    return out


def reciprocal_space_points_counter(interval: Dict[str, float], supercell: np.ndarray) -> int:
    """
    Count how many integer HKL grid points are in *interval* for the given
    supercell.  Includes the “×2 if l≠0” symmetry used in the legacy code.
    """
    supercell = np.asarray(supercell, dtype=float)
    step = 1.0 / supercell
    dim = len(supercell)

    def npts(start: float, end: float, st: float) -> int:
        return int(np.floor((end - start) / st + 0.5)) + 1

    h_n = npts(interval["h_start"], interval["h_end"], step[0])
    k_n = npts(interval.get("k_start", 0.0), interval.get("k_end", 0.0), step[1]) if dim > 1 else 1
    l_n = npts(interval.get("l_start", 0.0), interval.get("l_end", 0.0), step[2]) if dim > 2 else 1

    total = h_n * k_n * l_n
    if dim > 2 and not (interval["l_start"] == 0 and interval["l_end"] == 0):
        total *= 2
    return total


# ════════════════════════════════════════════════════════════════════════════
#  Q-space grid generator (mask-aware, matches original output)
# ════════════════════════════════════════════════════════════════════════════

def _call_generate_mask(mask_strategy, hkl: np.ndarray, mask_params: Dict[str, Any]):
    """Dispatch correctly across MaskStrategy variants."""
    sig = inspect.signature(mask_strategy.generate_mask)
    if len(sig.parameters) == 1:
        return mask_strategy.generate_mask(hkl)
    return mask_strategy.generate_mask(hkl, mask_params)

def generate_q_space_grid(
    interval: Dict[str, float],
    B_: np.ndarray,
    mask_parameters: Dict[str, Any],
    mask_strategy,
    supercell: np.ndarray,
) -> np.ndarray:
    """Build masked q-points in Cartesian reciprocal coordinates."""
    supercell = np.asarray(supercell, dtype=float)
    int_supercell = np.round(supercell).astype(int)

    def get_axis_vals(ax: str, i: int):
        start = interval.get(f"{ax}_start", 0.0)
        end   = interval.get(f"{ax}_end",   0.0)
        N = int_supercell[i]
        idx0 = int(np.ceil(start * N))
        idx1 = int(np.floor(end   * N))
        vals = np.arange(idx0, idx1 + 1) / N
        return vals

    h_vals = get_axis_vals("h", 0)
    k_vals = get_axis_vals("k", 1) if supercell.size > 1 else np.array([0.0])
    l_vals = get_axis_vals("l", 2) if supercell.size > 2 else np.array([0.0])

    mesh = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
    hkl = np.stack([m.ravel() for m in mesh], axis=1)

    mask = _call_generate_mask(mask_strategy, hkl, mask_parameters) if mask_strategy is not None else np.ones(len(hkl), dtype=bool)
    hkl_masked = hkl[mask]
    q_coords = 2 * np.pi * (hkl_masked[:, : supercell.size] @ B_)
    return q_coords

def generate_q_space_grid_sync(*args, **kwargs):
    return generate_q_space_grid(*args, **kwargs)

# ════════════════════════════════════════════════════════════════════════════
#  RIFFT grid helpers (ported 1-to-1)
# ════════════════════════════════════════════════════════════════════════════

def _generate_grid(
    dimensionality: int,
    step_sizes: np.ndarray,
    central_point: np.ndarray,
    dist_from_atom_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dense rectangular grid around *central_point*."""
    eps = 1e-8
    axes = []
    for i in range(dimensionality):
        dist = dist_from_atom_center[i]
        step = step_sizes[i]
        if step <= 0 or dist <= step:
            axis = np.array([0.0])
        else:
            axis = np.arange(-dist, dist + step - eps, step)
            if axis.size == 0:
                axis = np.array([0.0])
        axes.append(axis)

    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.vstack([m.ravel() for m in mesh]).T + central_point
    shape_nd = np.array(mesh[0].shape)
    return pts, shape_nd

def _process_chunk(chunk_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.array([pd["coordinates"] for pd in chunk_data])
    dist_vec = np.array([pd["dist_from_atom_center"] for pd in chunk_data])
    step_vec = np.array([pd["step_in_frac"] for pd in chunk_data])

    grids, shapes = [], []
    for cp_, dv, sv in zip(coords, dist_vec, step_vec):
        g, s = _generate_grid(coords.shape[1], sv, cp_, dv)
        grids.append(g)
        shapes.append(s)

    return np.vstack(grids), np.vstack(shapes)

def generate_rifft_grid(chunk_data: List[dict]):
    return _process_chunk(chunk_data)

def _build_rifft_grid_locally(chunk_data: List[dict]):
    with _timed("RIFFT grid build"):
        return _process_chunk(chunk_data)

# ════════════════════════════════════════════════════════════════════════════
#  Stage-1: heavy NUFFT pass — one output file per interval
# ════════════════════════════════════════════════════════════════════════════

def _process_interval_element(
    iv: dict,
    q_grid: np.ndarray,
    el: str,
    orig_coords: np.ndarray,
    cell_orig: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
) -> Tuple:
    ff = ff_factory.calculate(q_grid, el, charge=charge)
    mask = elements_arr == el
    if not np.any(mask):
        return None
    q_amp = ff * execute_cunufft(orig_coords[mask], np.ones(mask.sum()), q_grid, eps=1e-12)
    q_av = execute_cunufft(cell_orig, np.ones(orig_coords.shape[0]), q_grid, eps=1e-12)
    q_del = execute_cunufft(
        orig_coords[mask] - cell_orig[mask], np.ones(mask.sum()), q_grid, eps=1e-12
    )
    q_av_final = ff * q_av * q_del / orig_coords.shape[0]
    return (iv["id"], el, q_grid, q_amp, q_av_final)

def _process_interval_coeff(
    iv: dict,
    q_grid: np.ndarray,
    coeff: np.ndarray,
    orig_coords: np.ndarray,
    cell_orig: np.ndarray,
) -> Tuple:
    """Legacy-exact implementation (no behavioural change)."""
    M = orig_coords.shape[0]
    c_ = coeff * (np.ones(M) + 1j * np.zeros(M))
    q_amplitudes       = execute_cunufft(orig_coords, c_, q_grid, eps=1e-12)
    q_amplitudes_av    = execute_cunufft(cell_orig, c_ * 0.0 + 1.0, q_grid, eps=1e-12)
    q_amplitudes_delta = execute_cunufft(orig_coords - cell_orig, c_, q_grid, eps=1e-12)
    q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / M
    return (iv["id"], "All", q_grid, q_amplitudes, q_amplitudes_av_final)

def aggregate_interval_tasks(tasks: List[tuple], use_coeff: bool) -> IntervalTask:
    """Merge results for different elements (or a single coeff run)."""
    if use_coeff:
        irecip_id, element, qg, qa, qav = tasks[0]
        return IntervalTask(irecip_id, element, qg, qa, qav)

    irecip_id = tasks[0][0]
    q_grid = tasks[0][2]
    q_amp = np.sum([t[3] for t in tasks], axis=0)
    q_av = np.sum([t[4] for t in tasks], axis=0)
    return IntervalTask(irecip_id, "All", q_grid, q_amp, q_av)

DEFAULT_INTERVAL_RETRIES = 2

def handle_interval_worker(
        iv: dict,
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
        out_dir: str,
        db_path: str,
) -> str | None:
    """
    Heavy NUFFT for one interval.
    Returns the final .npz *path* (as a string) or None if skipped.
    """
    from managers.database_manager import create_db_manager_for_thread
    db = create_db_manager_for_thread(db_path)

    if db.is_interval_precomputed(iv["id"]):
        p = Path(out_dir) / f"interval_{iv['id']}.npz"
        if p.exists():
            db.close()
            return str(p)

    q_grid = generate_q_space_grid_sync(iv, B_, mask_params, MaskStrategy, supercell)
    if q_grid.size == 0:
        db.close()
        return None                                # fully masked

    tasks: list[tuple] = []
    if use_coeff:
        tasks.append(
            _process_interval_coeff(iv, q_grid, coeff_val,
                                    original_coords, cells_origin)
        )
    else:
        for el in unique_elements:
            t = _process_interval_element(
                iv, q_grid, el,
                original_coords, cells_origin,
                elements_arr, charge, ff_factory,
            )
            if t is not None:
                tasks.append(t)

    if not tasks:
        db.close()
        return None

    task = aggregate_interval_tasks(tasks, use_coeff)

    out_p = Path(out_dir) / f"interval_{task.irecip_id}.npz"
    with tempfile.NamedTemporaryFile(dir=out_dir,
                                     prefix=f"interval_{task.irecip_id}_",
                                     suffix=".npz",
                                     delete=False) as tf:
        np.savez_compressed(
            tf,
            irecip_id=task.irecip_id,
            element=task.element,
            q_grid=task.q_grid,
            q_amp=task.q_amp,
            q_amp_av=task.q_amp_av,
        )
    Path(tf.name).replace(out_p)

    db.mark_interval_precomputed(task.irecip_id, True)
    db.close()
    return str(out_p)

def precompute_intervals(
    reciprocal_space_intervals: Iterable[dict],
    *,
    B_: np.ndarray,
    parameters: Dict[str, Any],
    unique_elements: Iterable[str],
    mask_params: Dict[str, Any],
    MaskStrategy,
    supercell: np.ndarray,
    out_dir: Path,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
    db: DatabaseManager,
    client: Client | None,
) -> List[Path]:
    """
    Heavy NUFFT stage. Generates one .npz per interval.
    Works with Dask client or in a local synchronous fallback.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    use_coeff = "coeff" in parameters
    coeff_val = parameters.get("coeff")

    todo, cached = [], []
    for iv in reciprocal_space_intervals:
        iv_id = iv["id"]
        path  = out_dir / f"interval_{iv_id}.npz"
        if db.is_interval_precomputed(iv_id) and path.exists():
            cached.append(path)
        else:
            todo.append(iv)

    if not todo:
        logger.info("Stage-1 complete: %d written, %d cached, %d skipped", 0, len(cached), 0)
        return cached

    written_files: List[Path] = []

    # ── Dask path ─────────────────────────────────────────────────────────
    if (client is not None) and (not _is_sync_client(client)):
        futures = [
            client.submit(
                handle_interval_worker,
                iv,
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
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                out_dir=str(out_dir),
                db_path=db.db_path,
                pure=False,
                resources={"nufft": 1},
            )
            for iv in todo
        ]

        with logging_redirect_tqdm():
            with _tqdm(len(futures), desc="Precompute intervals", unit="intervals") as pbar:
                for fut, _ok in _yield_futures_with_results(futures, client):
                    try:
                        p = fut.result()
                    except Exception:
                        p = None
                    if p:
                        written_files.append(Path(p))
                    pbar.update(1); pbar.refresh()

        logger.info(
            "Stage-1 complete: %d written, %d cached, %d skipped",
            len(written_files), len(cached), len(todo) - len(written_files),
        )
        return cached + written_files

    # ── Synchronous (no Dask client) ─────────────────────────────────────
    with _tqdm(len(todo), desc="Precompute intervals", unit="intervals") as pbar:
        for iv in todo:
            p = handle_interval_worker(
                iv,
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
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                out_dir=str(out_dir),
                db_path=db.db_path,
            )
            if p:
                written_files.append(Path(p))
            pbar.update(1); pbar.refresh()

    logger.info(
        "Stage-1 complete: %d written, %d cached, %d skipped",
        len(written_files), len(cached), len(todo) - len(written_files),
    )
    return cached + written_files


# ════════════════════════════════════════════════════════════════════════════
#  Stage-2: read cached intervals, apply to every chunk_id
# ════════════════════════════════════════════════════════════════════════════

@delayed
def _interval_path_delayed(path: Path):
    return path

def _save_amplitudes_and_meta(
    *,
    chunk_id: int,
    task: IntervalTask,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    point_data_processor: PointDataProcessor,
    db_path: str,
    quiet_logs: bool = False,
) -> None:
    """
    Idempotent accumulation of ΔF into on-disk HDF5 for this chunk and DB flagging.
    """
    t0 = TIMER()
    lock = chunk_mutex(chunk_id)

    # Filenames (stable across runs)
    fn_amp   = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes")
    fn_amp_av   = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes_av")
    fn_shape = point_data_processor.data_saver.generate_filename(chunk_id, "_shapeNd")
    fn_tot   = point_data_processor.data_saver.generate_filename(
        chunk_id, "_amplitudes_ntotal_reciprocal_space_points")
    fn_nrec  = point_data_processor.data_saver.generate_filename(
        chunk_id, "_amplitudes_nreciprocal_space_points")
    fn_applied = point_data_processor.data_saver.generate_filename(
        chunk_id, "_applied_interval_ids")

    already_applied = False

    with lock:
        # one-off metadata
        try:
            point_data_processor.data_saver.load_data(fn_shape)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data({"shapeNd": grid_shape_nd}, fn_shape)

        # total reciprocal points (write once; update if sentinel -1)
        val = int(total_reciprocal_points)
        try:
            d = point_data_processor.data_saver.load_data(fn_tot)
        
            def _needs_update(store: dict, key: str) -> bool:
                arr = store.get(key, None)
                if arr is None:
                    return True
                try:
                    v = int(np.array(arr).ravel()[0])
                except Exception:
                    return True
                return v == -1  # sentinel → rewrite
        
            if _needs_update(d, "ntotal_reciprocal_space_points") or _needs_update(d, "ntotal_reciprocal_points"):
                d["ntotal_reciprocal_space_points"] = np.array([val], dtype=np.int64)
                d["ntotal_reciprocal_points"]       = np.array([val], dtype=np.int64)  # legacy
                point_data_processor.data_saver.save_data(d, fn_tot)
        
        except FileNotFoundError:
            point_data_processor.data_saver.save_data(
                {
                    "ntotal_reciprocal_space_points": np.array([val], dtype=np.int64),
                    "ntotal_reciprocal_points":       np.array([val], dtype=np.int64),
                },
                fn_tot,
            )


        # idempotency ledger
        try:
            applied_arr = point_data_processor.data_saver.load_data(fn_applied)["ids"]
            applied_set = set(int(x) for x in np.array(applied_arr).ravel().tolist())
        except FileNotFoundError:
            applied_set = set()

        if int(task.irecip_id) in applied_set:
            already_applied = True
        else:
            # load current
            try:
                current = point_data_processor.data_saver.load_data(fn_amp)["amplitudes"]
                current_av = point_data_processor.data_saver.load_data(fn_amp_av)["amplitudes_av"]
                nrec    = point_data_processor.data_saver.load_data(fn_nrec)["nreciprocal_space_points"]
            except FileNotFoundError:
                current, current_av, nrec = None, None, 0

            # accumulate amplitudes
            if current is None:
                current = amplitudes_delta
                nrec    = task.q_grid.shape[0]
            else:
                # 3-D symmetry: double count the non-zero-l plane
                if task.q_grid.shape[1] > 2 and np.max(np.abs(task.q_grid[:, 2])) > 1e-7:
                    current[:, 1] += amplitudes_delta + np.conj(amplitudes_delta)
                    nrec += task.q_grid.shape[0] * 2
                else:
                    current[:, 1] += amplitudes_delta
                    nrec += task.q_grid.shape[0]
            if current_av is None:
                current_av = amplitudes_average
            else:
                # 3-D symmetry: double count the non-zero-l plane
                if task.q_grid.shape[1] > 2 and np.max(np.abs(task.q_grid[:, 2])) > 1e-7:
                    current_av[:, 1] += amplitudes_average + np.conj(amplitudes_average)
                else:
                    current_av[:, 1] += amplitudes_average
            # persist (atomic via existing saver)
            point_data_processor._save_chunk_data(chunk_id, None, current, current_av, nrec)

            # update ledger
            applied_set.add(int(task.irecip_id))
            applied_sorted = np.array(sorted(applied_set), dtype=np.int64)
            point_data_processor.data_saver.save_data({"ids": applied_sorted}, fn_applied)

    # DB flag (thread-local connection)
    from managers.database_manager import create_db_manager_for_thread
    db = create_db_manager_for_thread(db_path)
    try:
        db.update_interval_chunk_status(task.irecip_id, chunk_id, saved=True)
    finally:
        db.close()

    # quiet INFO to avoid tearing the tqdm on the driver
    if quiet_logs:
        logger.debug(
            "write-HDF5 | chunk %d | iv %d %s | %.3f s",
            chunk_id,
            task.irecip_id,
            "already applied (idempotent skip)" if already_applied else "applied",
            TIMER() - t0,
        )
    else:
        if already_applied:
            logger.info(
                "write-HDF5 | chunk %d | iv %d already applied (idempotent skip) | %.3f s",
                chunk_id, task.irecip_id, TIMER() - t0
            )
        else:
            logger.info(
                "write-HDF5 | chunk %d | iv %d applied | %.3f s",
                chunk_id, task.irecip_id, TIMER() - t0
            )

###############################################################################
#  _process_chunk_id  (returns True/False and can silence worker logs)       #
###############################################################################
def _process_chunk_id(
    chunk_id: int,
    iv_path: Path,
    atoms: np.recarray,
    total_reciprocal_points: int,
    point_data_processor: PointDataProcessor,
    db_path: str,
    quiet_logs: bool = False,
) -> bool:
    """Compute Δ-amplitudes for one (interval, chunk) pair – executes on worker."""
    # Reduce worker log noise while tqdm is active on the driver.
    if quiet_logs:
        for name in (
            __name__,
            "managers.database_manager",
            "DatabaseManager",
            "processors.point_data_processor",
            "PointDataProcessor",
            "RIFFTInDataSaver",
        ):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

    recip_id = None
    try:
        # load cached interval
        with np.load(iv_path, mmap_mode="r") as dat:
            task = IntervalTask(
                int(dat["irecip_id"]),
                str(dat["element"]),
                dat["q_grid"],
                dat["q_amp"],
                dat["q_amp_av"],
            )
        recip_id = task.irecip_id

        # RIFFT grid
        chunk_data = [
            {
                "coordinates": atoms["coordinates"][i],
                "dist_from_atom_center": atoms["dist_from_atom_center"][i],
                "step_in_frac": atoms["step_in_frac"][i],
            }
            for i in range(atoms.shape[0])
        ]
        rifft_grid, grid_shape = _process_chunk(chunk_data)

        # inverse NUFFT
        amplitudes_delta = execute_inverse_cunufft(
            q_coords=task.q_grid,
            c=task.q_amp - task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        amplitudes_average = execute_inverse_cunufft(
            q_coords=task.q_grid,
            c=task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        
        # persist & DB
        _save_amplitudes_and_meta(
            chunk_id=chunk_id,
            task=task,
            grid_shape_nd=grid_shape,
            total_reciprocal_points=total_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_data_processor=point_data_processor,
            db_path=db_path,
            quiet_logs=quiet_logs,
        )

        return True
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            chunk_id,
            recip_id if recip_id is not None else "n/a",
            err,
            exc_info=True,
        )
        # GPU quarantine on the worker itself
        try:
            _msg = str(err).lower()
            is_gpu_err = any(
                kw in _msg for kw in (
                    "cuda", "cudart", "cufft", "cufinufft", "cupy",
                    "device-side assert", "illegal memory access",
                    "out of memory", "driver shutting down"
                )
            )
            if is_gpu_err:
                from distributed import get_worker
                try:
                    set_cpu_only(True)
                    w = get_worker()
                    logger.warning("Worker %s set to CPU-only after GPU error", w.address)
                except Exception:
                    pass
                try:
                    cnt = getattr(w, "gpu_fail_count", 0)
                    setattr(w, "gpu_fail_count", int(cnt) + 1)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return False

# ————————————————————————————————————————————————————————————————
#  Core streaming submission function (Stage-2)
# ————————————————————————————————————————————————————————————————

def _task_key(iv_id: int, chunk_id: int) -> str:
    return f"proc-{iv_id}-{chunk_id}"

def _parse_task_key(key: str) -> tuple[int, int]:
    _, iv_id, cid = key.split("-", 2)
    return int(iv_id), int(cid)

def process_chunks_with_intervals(
    interval_files: Iterable[Path],
    *,
    chunk_ids: Iterable[int],
    total_reciprocal_points: int,
    point_data_list: list[dict],
    point_data_processor: PointDataProcessor,
    db_manager: DatabaseManager,
    client: Client | None,
    max_inflight: int = 5_000,
) -> None:
    """Stage-2 streaming loop with retry logic. Falls back to sync when client is None."""
    # If nothing to do, bail early
    unsaved = set(db_manager.get_unsaved_interval_chunks())  # (iv_id, chunk_id)
    total_tasks = len(unsaved)
    if total_tasks == 0:
        logger.info("Stage-2 skipped – no unsaved (interval, chunk) pairs.")
        return

    # ── Synchronous (debug mode, no Dask) ─────────────────────────────────
    if client is None:
        rec = _point_list_to_recarray(point_data_list)
        with _tqdm(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:
            for p in interval_files:
                iv_id = int(p.stem.split("_")[1])
                for cid in chunk_ids:
                    if (iv_id, cid) not in unsaved:
                        continue
                    atoms = rec[rec.chunk_id == cid]
                    ok = _process_chunk_id(
                        cid, p, atoms, total_reciprocal_points,
                        point_data_processor, db_manager.db_path, False
                    )
                    pbar.update(1); pbar.refresh()
                    if not ok:
                        logger.error("GAVE UP after retries | iv %d | chunk %d (sync)", iv_id, cid)
        logger.info("Stage-2 finished (sync).")
        return

    # ── Dask path ─────────────────────────────────────────────────────────
    FAIL_STREAK, FAIL_THRESHOLD = 0, 3
    GPU_TRIPPED = False

    def _trip_to_cpu_only():
        nonlocal GPU_TRIPPED, max_inflight
        if GPU_TRIPPED:
            return
        if hasattr(client, "run"):
            try:
                client.run(set_cpu_only, True)
            except Exception:
                pass
        max_inflight = min(max_inflight, 256)
        GPU_TRIPPED = True
        logger.warning("Circuit-breaker: switching Stage-2 to CPU-only & throttling.")

    rec = _point_list_to_recarray(point_data_list)
    chunk_futs = {
        cid: client.scatter(rec[rec.chunk_id == cid], broadcast=False, hash=False)
        for cid in chunk_ids
    }
    pd_future = client.scatter(point_data_processor, broadcast=True)
    db_path   = db_manager.db_path

    retries_left  = {key: DEFAULT_TASK_RETRIES for key in unsaved}
    flying: set   = set()
    fut_meta: dict = {}  # future -> (iv_id, chunk_id, iv_path_future)
    submitted     = 0

    def _submit(iv_path_future, iv_id, cid):
        nonlocal submitted
        fut = client.submit(
            _process_chunk_id,
            cid,
            iv_path_future,
            chunk_futs[cid],
            total_reciprocal_points,
            pd_future,
            db_path,
            True,                       # quiet_logs=True
            key=f"proc-{iv_id}-{cid}",
            pure=False,
            resources={"nufft": 1},
            retries=DEFAULT_TASK_RETRIES,
        )
        flying.add(fut)
        fut_meta[fut] = (iv_id, cid, iv_path_future)
        submitted += 1

    def _harvest_finished_nonblocking(bump):
        nonlocal FAIL_STREAK
        done_now = [f for f in list(flying) if f.done()]
        for f in done_now:
            try:
                ok = bool(f.result())
            except Exception:
                ok = False

            flying.discard(f)
            iv, ch, ivpf = fut_meta.pop(f, (None, None, None))
            bump()

            if not ok and iv is not None:
                FAIL_STREAK += 1
                if FAIL_STREAK >= FAIL_THRESHOLD:
                    _trip_to_cpu_only()
                if retries_left.get((iv, ch), 0) > 0:
                    retries_left[(iv, ch)] -= 1
                    _submit(ivpf, iv, ch)
            else:
                FAIL_STREAK = 0

    with logging_redirect_tqdm():
        with _tqdm(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:
            def bump():
                pbar.update(1); pbar.refresh()

            for p in interval_files:
                iv_id = int(p.stem.split("_")[1])
                iv_path_future = client.scatter(p, broadcast=False)
                for cid in chunk_ids:
                    if (iv_id, cid) not in unsaved:
                        continue
                    _submit(iv_path_future, iv_id, cid)
                    _harvest_finished_nonblocking(bump)
                    while len(flying) >= max_inflight:
                        for fut, ok in _yield_futures_with_results(list(flying), client):
                            flying.discard(fut)
                            iv, ch, ivpf = fut_meta.pop(fut, (None, None, None))
                            bump()
                            if not ok and iv is not None:
                                FAIL_STREAK += 1
                                if FAIL_STREAK >= FAIL_THRESHOLD:
                                    _trip_to_cpu_only()
                                if retries_left.get((iv, ch), 0) > 0:
                                    retries_left[(iv, ch)] -= 1
                                    _submit(ivpf, iv, ch)
                            else:
                                FAIL_STREAK = 0

            for fut, ok in _yield_futures_with_results(list(flying), client):
                iv, ch, _ = fut_meta.pop(fut, (None, None, None))
                bump()
                if not ok and iv is not None:
                    logger.error("GAVE UP after retries | iv %d | chunk %d", iv, ch)

    logger.info("Stage-2 finished – %d tasks submitted", submitted)


# ════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════════════════════

def _point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    """Compact list[dict] → contiguous NumPy recarray (runtime-dimension)."""
    if not point_data_list:
        raise ValueError("point_data_list is empty")

    dim = len(point_data_list[0]["coordinates"])
    for pd in point_data_list:
        if len(pd["coordinates"]) != dim:
            raise ValueError("Mixed dimensionalities in point_data_list")

    vect = (dim,)
    dtype = np.dtype([
        ("chunk_id", "<i4"),
        ("coordinates", "<f8", vect),
        ("dist_from_atom_center", "<f8", vect),
        ("step_in_frac", "<f8", vect),
    ])

    out = np.empty(len(point_data_list), dtype=dtype)
    for i, pd in enumerate(point_data_list):
        out[i] = (
            pd["chunk_id"],
            pd["coordinates"],
            pd["dist_from_atom_center"],
            pd["step_in_frac"],
        )
    return out.view(np.recarray)

def compute_amplitudes_delta(
    parameters: Dict[str, Any],
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters: Dict[str, Any],
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor: PointDataProcessor,
    client: Client,
):
    """API entry identical to legacy version; registers WorkerPlugin & runs."""
    try:
        if client is not None and not _is_sync_client(client):
            client.register_worker_plugin(CuPyCleanup(), name="cupy-cleanup")
    except ValueError:
        pass

    # unpack parameters (verbatim)
    reciprocal_space_intervals_all = parameters["reciprocal_space_intervals_all"]
    reciprocal_space_intervals = parameters["reciprocal_space_intervals"]
    point_data_list = parameters["point_data_list"]
    original_coords = parameters["original_coords"]
    cells_origin = parameters["cells_origin"]
    elements_arr = parameters["elements"]
    vectors = parameters["vectors"]
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0.0)

    B_ = np.linalg.inv(vectors / supercell)
    unique_elements = np.unique(elements_arr)

    # Stage-0: total HKL count
    total_pts = sum(
        reciprocal_space_points_counter(_to_interval_dict(iv), supercell)
        for iv in reciprocal_space_intervals_all
    )
    logger.info("Total reciprocal-space integer points: %s", total_pts)

    # Stage-1: heavy NUFFT / interval caching  (quiet DB INFO around tqdm)
    interval_dir = Path(output_dir) / "precomputed_intervals"
    with _quiet_db_info():
        interval_files = precompute_intervals(
            reciprocal_space_intervals,
            B_=B_,
            parameters=parameters,
            unique_elements=unique_elements,
            mask_params=MaskStrategyParameters,
            MaskStrategy=MaskStrategy,
            supercell=supercell,
            out_dir=interval_dir,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            charge=charge,
            ff_factory=FormFactorFactoryProducer,
            db=db_manager,
            client=client,
        )

    # Stage-2: accumulate ΔF chunk-wise (streaming)  (quiet DB INFO around tqdm)
    chunk_ids = db_manager.get_pending_chunk_ids()
    with _quiet_db_info():
        process_chunks_with_intervals(
            interval_files,
            chunk_ids=chunk_ids,
            total_reciprocal_points=total_pts,
            point_data_list=point_data_list,
            point_data_processor=point_data_processor,
            db_manager=db_manager,
            client=client,
        )

    logger.info("Completed compute_amplitudes_delta (refactored + fixed)")
