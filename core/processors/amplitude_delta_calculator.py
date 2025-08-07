"""
Refactored amplitude-delta calculator
------------------------------------

Drop-in replacement for processors/amplitude_delta_calculator.py that keeps
ALL original behaviour but structures the work as a clean two-stage pipeline.

Author: ChatGPT (refactor for Maksim), 2025-06-25
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple
import time
import tempfile
import gc
from contextlib import contextmanager
import atexit 
import numpy as np
import cupy as cp
from dask.distributed import (
    Client,
    Lock,
    WorkerPlugin,
    as_completed,
)
from dask import delayed
from utilities.cunufft_wrapper import execute_cunufft, execute_inverse_cunufft
from managers.database_manager import DatabaseManager
from processors.point_data_processor import PointDataProcessor
from tqdm import tqdm
# ──────────────────────────────────────────────────────────────────────────────
# Globals & helpers
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
TIMER = time.perf_counter  # monotonic clock alias
DEFAULT_TASK_RETRIES = 2

@contextmanager
def _timed(label: str):
    t0 = TIMER()
    try:
        yield
    finally:
        logger.info("%s took %.3f s", label, TIMER() - t0)


# ──────────────────────────────────────────────────────────────────────────────
#  Worker‑side clean‑up (silences leaked‑shared_memory warnings)  ##############
# ──────────────────────────────────────────────────────────────────────────────

def _final_cleanup():
    """Free CuPy pools *and* unlink any remaining shared‑memory blocks."""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass  # CPU‑only worker or CuPy not imported

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

    def teardown(self, worker):  # noqa: D401  (imperative mood required)
        _final_cleanup()


# ──────────────────────────────────────────────────────────────────────────────
#  Small helper structures & utilities
# ──────────────────────────────────────────────────────────────────────────────


class IntervalTask(NamedTuple):
    """Cached data for a reciprocal‑space interval after the heavy NUFFT pass."""

    irecip_id: int
    element: str
    q_grid: np.ndarray
    q_amp: np.ndarray
    q_amp_av: np.ndarray


# ————————————————————————————————————————————————————————————————
# Misc. utility functions (unchanged logic – only formatting edits)
# ————————————————————————————————————————————————————————————————

def _to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    """Translate flexible JSON/HDF5 interval to flat {h_start, h_end, …}."""

    out: Dict[str, float] = {}
    for ax in ("h", "k", "l"):
        rng = iv.get(f"{ax}_range")
        if rng is not None:
            out[f"{ax}_start"], out[f"{ax}_end"] = rng
    return out



# ------------------------------------------------------------------------- #
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
        total *= 2  # original “mirror” rule
    return total

atexit.register(_final_cleanup)

# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
TIMER = time.perf_counter          # monotonic and fast

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

    # already done?
    if db.is_interval_precomputed(iv["id"]):
        p = Path(out_dir) / f"interval_{iv['id']}.npz"
        if p.exists():
            return str(p)

    q_grid = generate_q_space_grid_sync(
        iv, B_, mask_params, MaskStrategy, supercell
    )
    if q_grid.size == 0:
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
    logger.debug("Saved interval %s → %s", task.irecip_id, out_p)
    db.close()
    return str(out_p)
 
def chunk_mutex(chunk_id: int) -> Lock:
    """
    Return a cluster-wide mutex for this chunk_id.
    Call ensure_dask_client() once before tasks are built.
    """
    return Lock(f"chunk-{chunk_id}")

# ════════════════════════════════════════════════════════════════════════════
#  Q-space grid generator (mask-aware, matches original output)
# ════════════════════════════════════════════════════════════════════════════
def _call_generate_mask(mask_strategy, hkl: np.ndarray, mask_params: Dict[str, Any]):
    """
    Some MaskStrategy subclasses expect only one arg (HKL), others (HKL, params).
    Dispatch correctly using introspection.
    """
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
    """
    Build masked q-points (in Cartesian reciprocal coordinates), using integer-index
    based HKL grid generation to ensure all intervals perfectly match symmetry
    requirements and there are no overlaps or misses due to floating-point error.
    """
    # Convert supercell size to integer per axis
    supercell = np.asarray(supercell, dtype=float)
    int_supercell = np.round(supercell).astype(int)

    # Helper: find start/end index for each axis
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

    # Apply mask if mask_strategy is not None
    if mask_strategy is not None:
        mask = _call_generate_mask(mask_strategy, hkl, mask_parameters)
    else:
        mask = np.ones(len(hkl), dtype=bool)

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
    """
    Create a dense rectangular grid around *central_point* with given step sizes.
    """
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
    """Synchronous helper, runs on the worker."""
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
###############################################################################
# 1)  _process_chunk_id  (now returns True/False instead of raising)          #
###############################################################################
def _process_chunk_id(
    chunk_id: int,
    iv_path: Path,
    atoms: np.recarray,
    total_reciprocal_points: int,
    point_data_processor: PointDataProcessor,
    db_path: str,
) -> bool:                                  # ← return success flag
    """Compute Δ-amplitudes for one (interval, chunk) pair – executes on worker.

    Returns
    -------
    bool
        True  → written & DB updated
        False → failed (caller may retry)
    """
    t_total  = TIMER()
    recip_id = None

    try:
        # ---------- load cached interval -----------------------------------
        with np.load(iv_path, mmap_mode="r") as dat:
            task = IntervalTask(
                int(dat["irecip_id"]),
                str(dat["element"]),
                dat["q_grid"],
                dat["q_amp"],
                dat["q_amp_av"],
            )
        recip_id = task.irecip_id

        # ---------- RIFFT grid ---------------------------------------------
        chunk_data = [
            {
                "coordinates": atoms["coordinates"][i],
                "dist_from_atom_center": atoms["dist_from_atom_center"][i],
                "step_in_frac": atoms["step_in_frac"][i],
            }
            for i in range(atoms.shape[0])
        ]
        rifft_grid, grid_shape = _process_chunk(chunk_data)

        # ---------- inverse NUFFT ------------------------------------------
        amplitudes_delta = execute_inverse_cunufft(
            q_coords=task.q_grid,
            c=task.q_amp - task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )

        # ---------- persist results & DB flag ------------------------------
        _save_amplitudes_and_meta(
            chunk_id=chunk_id,
            task=task,
            grid_shape_nd=grid_shape,
            total_reciprocal_points=total_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            point_data_processor=point_data_processor,
            db_path=db_path,
        )

        return True                       # ← SUCCESS
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            chunk_id,
            recip_id if recip_id is not None else "n/a",
            err,
            exc_info=True,
        )
        # Optional: reset the device so the next try starts clean
        try:
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return False                      # ← FAILURE – caller may retry
    finally:
        try:
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        gc.collect()
        logger.info(
            "TOTAL task | chunk %d | iv %s took %.3f s",
            chunk_id,
            recip_id if recip_id is not None else "n/a",
            TIMER() - t_total,
        )


def _process_interval_coeff(
    iv: dict,
    q_grid: np.ndarray,
    coeff: np.ndarray,
    orig_coords: np.ndarray,
    cell_orig: np.ndarray,
) -> Tuple:
    """
    Legacy-exact implementation (no behavioural change allowed).
    """
    M = orig_coords.shape[0]

    # --- exact original construction --------------------------------------
    c_ = coeff * (np.ones(M) + 1j * np.zeros(M))

    # NUFFT calculations (unchanged order / eps)
    q_amplitudes       = execute_cunufft(orig_coords, c_, q_grid, eps=1e-12)
    q_amplitudes_av    = execute_cunufft(cell_orig, c_ * 0.0 + 1.0, q_grid, eps=1e-12)
    q_amplitudes_delta = execute_cunufft(
        orig_coords - cell_orig, c_, q_grid, eps=1e-12
    )

    # final combination exactly as before
    q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / M

    return (iv["id"], "All", q_grid, q_amplitudes, q_amplitudes_av_final)


def aggregate_interval_tasks(tasks: List[tuple], use_coeff: bool) -> IntervalTask:
    """
    Merge results for different elements (or a single coeff run) into a single object.
    """
    if use_coeff:
        irecip_id, element, qg, qa, qav = tasks[0]
        return IntervalTask(irecip_id, element, qg, qa, qav)

    irecip_id = tasks[0][0]
    q_grid = tasks[0][2]
    q_amp = np.sum([t[3] for t in tasks], axis=0)
    q_av = np.sum([t[4] for t in tasks], axis=0)
    return IntervalTask(irecip_id, "All", q_grid, q_amp, q_av)

DEFAULT_INTERVAL_RETRIES = 2  
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
    ff_factory, db: DatabaseManager,
    client
) -> List[Path]:
    """
    Heavy NUFFT stage.  Generates **one compressed .npz per interval** and
    returns the list of files that were actually written.

    The work is dispatched through `dask.distributed.Client`, so it runs on a
    local in-process cluster when testing and on remote workers under
    dask-mpi / PBS / Slurm just the same.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    #client   = ensure_dask_client(2)
    use_coeff = "coeff" in parameters
    coeff_val = parameters.get("coeff")
    todo, cached = [], []
    for iv in reciprocal_space_intervals:
        iv_id = iv["id"]
        path  = out_dir / f"interval_{iv_id}.npz"
        if db.is_interval_precomputed(iv_id) and path.exists():
            cached.append(path)          # already done last run
        else:
            todo.append(iv)              # still to compute
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

    from tqdm import tqdm
    written_files = []
    with tqdm(total=len(futures), desc="Precompute intervals", unit="interval") as pbar:
        for future, result in zip(as_completed(futures), futures):
            p = future.result()
            if p:
                written_files.append(Path(p))
            pbar.update(1)
    # In case you want to keep the Dask's built-in progress as well:
    # progress(futures)
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

# ---------------------------------------------------------------------------

def _save_amplitudes_and_meta(
    *,
    chunk_id: int,
    task: IntervalTask,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    point_data_processor: PointDataProcessor,
    db_path: str,
) -> None:
    """
    Accumulate ΔF into the on-disk HDF-5 for this chunk and update DB flags.
    Serialised with a cluster-wide `Lock("chunk-<id>")` to avoid races.
    """
    t0 = TIMER()
    lock: Lock = Lock(f"chunk-{chunk_id}")

    with lock:
        # ---------- filenames ------------------------------------------------
        fn_amp   = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes")
        fn_shape = point_data_processor.data_saver.generate_filename(chunk_id, "_shapeNd")
        fn_tot   = point_data_processor.data_saver.generate_filename(
            chunk_id, "_amplitudes_ntotal_reciprocal_space_points")
        fn_nrec  = point_data_processor.data_saver.generate_filename(
            chunk_id, "_amplitudes_nreciprocal_space_points")

        # ---------- one-off metadata -----------------------------------------
        try:
            point_data_processor.data_saver.load_data(fn_shape)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data({"shapeNd": grid_shape_nd}, fn_shape)

        try:
            point_data_processor.data_saver.load_data(fn_tot)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data(
                {"ntotal_reciprocal_points": total_reciprocal_points}, fn_tot)

        # ---------- accumulate amplitudes ------------------------------------
        try:
            current = point_data_processor.data_saver.load_data(fn_amp)["amplitudes"]
            nrec    = point_data_processor.data_saver.load_data(fn_nrec)["nreciprocal_space_points"]
        except FileNotFoundError:
            current, nrec = None, 0

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

        point_data_processor._save_chunk_data(chunk_id, None, current, nrec)

    # ---------- DB flag outside the lock (thread-local connection) ----------
    from managers.database_manager import create_db_manager_for_thread
    db = create_db_manager_for_thread(db_path)
    db.update_interval_chunk_status(task.irecip_id, chunk_id, saved=True)
    db.close()

    logger.info("write-HDF5 | chunk %d | iv %d took %.3f s",
                chunk_id, task.irecip_id, TIMER() - t0)

def _point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    """Compact list[dict] → contiguous NumPy recarray (runtime‑dimension)."""

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


# ————————————————————————————————————————————————————————————————
#  Core streaming submission function
# ————————————————————————————————————————————————————————————————
###############################################################################
# 2)  process_chunks_with_intervals                                           #
###############################################################################

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
    client: Client,
    max_inflight: int = 5_000,
) -> None:
    """Stage-2 streaming loop with retry logic (tuple-safe)."""

    rec = _point_list_to_recarray(point_data_list)
    chunk_futs = {
        cid: client.scatter(rec[rec.chunk_id == cid], broadcast=False, hash=False)
        for cid in chunk_ids
    }

    pd_future = client.scatter(point_data_processor, broadcast=True)
    db_path   = db_manager.db_path

    unsaved       = set(db_manager.get_unsaved_interval_chunks())
    retries_left  = {key: DEFAULT_TASK_RETRIES for key in unsaved}
    flying: set   = set()
    submitted     = 0
    total_tasks   = len(unsaved)

    pbar = tqdm(total=total_tasks, desc="Stage 2 (chunks × intervals)", unit="tasks")

    # helper --------------------------------------------------------------
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
            key=_task_key(iv_id, cid),
            pure=False,
            resources={"nufft": 1},
        )
        flying.add(fut)
        submitted += 1

    # main loop -----------------------------------------------------------
    for p in interval_files:
        iv_id = int(p.stem.split("_")[1])
        iv_path_future = client.scatter(p, broadcast=False)

        for cid in chunk_ids:
            if (iv_id, cid) not in unsaved:
                continue
            _submit(iv_path_future, iv_id, cid)

            # throttle ----------------------------------------------------
            if len(flying) >= max_inflight:
                for fut, ok in as_completed(flying, with_results=True):
                    pbar.update(1)
                    flying.discard(fut)

                    if not ok:
                        iv, ch = _parse_task_key(fut.key)
                        if retries_left[(iv, ch)] > 0:
                            retries_left[(iv, ch)] -= 1
                            _submit(iv_path_future, iv, ch)
                    break   # processed one finished task

    # drain remaining -----------------------------------------------------
    for fut, ok in as_completed(flying, with_results=True):
        pbar.update(1)
        if not ok:
            iv, ch = _parse_task_key(fut.key)
            logger.error("GAVE UP after retries | iv %d | chunk %d", iv, ch)

    pbar.close()
    logger.info("Stage-2 finished – %d tasks submitted", submitted)

# ###########################################################################
#  Public entry point                                                       #
# ###########################################################################

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

    # Ensure the cleanup plugin is active (idempotent)
    try:
        client.register_worker_plugin(CuPyCleanup(), name="cupy-cleanup")
    except ValueError:  # already registered in this session
        pass

    # ——— unpack parameters (verbatim) ————————————————————————
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

    # ——— Stage‑0: total HKL count ——————————————————————————
    total_pts = sum(
        reciprocal_space_points_counter(_to_interval_dict(iv), supercell)
        for iv in reciprocal_space_intervals_all
    )
    logger.info("Total reciprocal‑space integer points: %s", total_pts)

    # ——— Stage‑1: heavy NUFFT / interval caching ————————————————
    interval_dir = Path(output_dir) / "precomputed_intervals"

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

    # ——— Stage‑2: accumulate ΔF chunk‑wise (streaming) ———————————
    chunk_ids = db_manager.get_pending_chunk_ids()
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
