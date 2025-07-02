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
import numpy as np
from utilities.cunufft_wrapper import execute_cunufft, execute_inverse_cunufft
from managers.database_manager import DatabaseManager
from processors.point_data_processor import PointDataProcessor
from dask.distributed import Client, LocalCluster, Lock, get_client
from dask import delayed
import time

from time import perf_counter
from contextlib import contextmanager
from utilities.dask_helpres import ensure_dask_client


# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #
TIMER = time.perf_counter          # monotonic and fast
@contextmanager
def _timed(label: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.3f s", label, perf_counter() - t0)

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
#  Small helper structures & utilities
# ════════════════════════════════════════════════════════════════════════════
class IntervalTask(NamedTuple):
    """
    What we cache per reciprocal-space interval after the heavy NUFFT pass.
    """

    irecip_id: int
    element: str
    q_grid: np.ndarray
    q_amp: np.ndarray
    q_amp_av: np.ndarray


# ------------------------------------------------------------------------- #
def _to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    """
    Translate flexible JSON/HDF5 description into flat {h_start, h_end, …}.
    """

    res: Dict[str, float] = {}
    for axis in ("h", "k", "l"):
        rng = iv.get(f"{axis}_range")
        if rng is not None:
            res[f"{axis}_start"], res[f"{axis}_end"] = rng
    return res


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
    Build masked q-points (in Cartesian reciprocal coordinates).
    """
    supercell = np.asarray(supercell, dtype=float)
    step = 1.0 / supercell

    h_vals = (
        np.arange(interval["h_start"], interval["h_end"] + step[0], step[0])
        if interval["h_end"] > interval["h_start"]
        else np.array([interval["h_start"]])
    )
    k_vals = (
        np.arange(interval["k_start"], interval["k_end"] + step[1], step[1])
        if "k_start" in interval and interval["k_end"] > interval["k_start"]
        else np.array([interval.get("k_start", 0.0)])
    )
    l_vals = (
        np.arange(interval["l_start"], interval["l_end"] + step[2], step[2])
        if "l_start" in interval and interval["l_end"] > interval["l_start"]
        else np.array([interval.get("l_start", 0.0)])
    )

    mesh = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
    hkl = np.stack([m.ravel() for m in mesh], axis=1)

    if mask_strategy is not None:
        mask = _call_generate_mask(mask_strategy, hkl, mask_parameters)
    else:
        mask = np.ones(len(hkl), dtype=bool)

    hkl_masked = hkl[mask]
    q_coords = 2 * np.pi * (hkl_masked[:, : len(supercell)] @ B_)
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
    for cp, dv, sv in zip(coords, dist_vec, step_vec):
        g, s = _generate_grid(coords.shape[1], sv, cp, dv)
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

import tempfile
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
         )
         for iv in todo
     ]
    # gather results back to driver
    written_files = [Path(p) for p in client.gather(futures) if p]

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
            if task.q_grid.shape[1] > 2 and np.all(np.round(task.q_grid[:, 2], 8) != 0):
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


# ---------------------------------------------------------------------------
# Build one compact np.recarray from the original Python list
# ---------------------------------------------------------------------------
def _point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    """
    Convert the slow-to-pickle   list[dict]   into one contiguous NumPy
    structured array.  Each field is a *fixed-size* NumPy column, so
    serialisation is 5-10 × faster and the Dask scheduler no longer chokes.
    """
    # ---- describe the structure ------------------------------------------
    dtype = np.dtype([
        ("chunk_id",             "<i4"),        # int32
        ("coordinates",          "<f8",  (3,)), # 3×float64
        ("dist_from_atom_center","<f8",  (3,)),
        ("step_in_frac",         "<f8",  (3,)),
    ])

    out = np.empty(len(point_data_list), dtype=dtype)

    # ---- fill it ----------------------------------------------------------
    for i, pd in enumerate(point_data_list):
        out["chunk_id"][i]              = pd["chunk_id"]
        out["coordinates"][i]           = pd["coordinates"]
        out["dist_from_atom_center"][i] = pd["dist_from_atom_center"]
        out["step_in_frac"][i]          = pd["step_in_frac"]

    return out.view(np.recarray)   # convenient attribute-style access

def _process_chunk_id(
    chunk_id: int,
    iv_path: Path,
    atoms: np.recarray,              # **slice already filtered by the caller**
    total_reciprocal_points: int,
    point_data_processor: PointDataProcessor,
    db_path: str,
) -> None:

    t_total = TIMER()

    # ---------- load interval (.npz – tiny) -------------------------------
    dat  = np.load(iv_path, mmap_mode="r")
    task = IntervalTask(int(dat["irecip_id"]), str(dat["element"]),
                        dat["q_grid"], dat["q_amp"], dat["q_amp_av"])

    # ---------- build RIFFT grid -----------------------------------------
    t0 = TIMER()
    chunk_data = [
        dict(coordinates          = atoms["coordinates"][i],
             dist_from_atom_center= atoms["dist_from_atom_center"][i],
             step_in_frac         = atoms["step_in_frac"][i])
        for i in range(atoms.shape[0])
    ]
    rifft_grid, grid_shape_nd = _process_chunk(chunk_data)
    logger.info("RIFFT-grid   | chunk %d | iv %d took %.3f s",
                chunk_id, task.irecip_id, TIMER() - t0)

    # ---------- inverse NUFFT --------------------------------------------
    t1 = TIMER()
    amplitudes_delta = execute_inverse_cunufft(
        q_coords = task.q_grid,
        c        = task.q_amp - task.q_amp_av,
        real_coords = rifft_grid,
        eps = 1e-12,
    )
    logger.info("inv-NUFFT    | chunk %d | iv %d took %.3f s",
                chunk_id, task.irecip_id, TIMER() - t1)

    # ---------- save / update --------------------------------------------
    _save_amplitudes_and_meta(
        chunk_id               = chunk_id,
        task                   = task,
        grid_shape_nd          = grid_shape_nd,
        total_reciprocal_points= total_reciprocal_points,
        amplitudes_delta       = amplitudes_delta,
        point_data_processor   = point_data_processor,
        db_path                = db_path,
    )

    logger.debug("TOTAL task   | chunk %d | iv %d took %.3f s",
                 chunk_id, task.irecip_id, TIMER() - t_total)


# ---------------------------------------------------------------------------
# Build & launch stage-2 graph  –  uses the recarray helper above
# ---------------------------------------------------------------------------
def process_chunks_with_intervals(
    interval_files          : Iterable[Path],
    *,                                    # keyword-only
    chunk_ids               : Iterable[int],
    total_reciprocal_points : int,
    point_data_list         : list[dict],
    point_data_processor    : PointDataProcessor,
    db_manager              : DatabaseManager,
    client
) -> None:

    # 1️⃣  pack everything into *one* compact block
    rec = _point_list_to_recarray(point_data_list)

    # 2️⃣  scatter only the slice that belongs to each chunk
    chunk_futs = {
        cid: client.scatter(rec[rec.chunk_id == cid], broadcast=False, hash=False)
        for cid in chunk_ids
    }

    # 3️⃣  scatter paths (trivial) & create tasks
    iv_futs = {p: client.scatter(p, broadcast=False) for p in interval_files}
    unsaved = set(db_manager.get_unsaved_interval_chunks())
    tasks = [
        client.submit(
            _process_chunk_id,
            cid,
            iv_futs[p],
            chunk_futs[cid],               # ← tiny: just that chunk’s atoms
            total_reciprocal_points,
            point_data_processor,
            db_manager.db_path,
            pure=False                     # every call is unique
        )
        for cid in chunk_ids
        for p   in interval_files
        if (int(p.stem.split("_")[1]), cid) in unsaved 
    ]

    logger.info("Submitting %d interval×chunk tasks …", len(tasks))
    client.gather(tasks)                   # block until everything finishes
    logger.info("Stage-2 finished")

# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def compute_amplitudes_delta(
    parameters: Dict[str, Any],
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters: Dict[str, Any],
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor: PointDataProcessor,
    client
):
    """
    Entry-point function with the same signature the rest of MOSAIC expects.
    """

    # ------------------------ unpack parameters ----------------------------- #
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

    # ------------------ Stage-0: total point count -------------------------- #
    total_pts = sum(
        reciprocal_space_points_counter(_to_interval_dict(iv), supercell)
        for iv in reciprocal_space_intervals_all
    )
    logger.info("Total reciprocal-space integer points: %s", total_pts)

    # ------------------ Stage-1: precompute intervals ----------------------- #
    interval_dir = Path(output_dir) / "precomputed_intervals"

    #with dask.config.set(scheduler="threads", num_workers=max_threads):
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
        client = client
    )

    # ------------------ Stage-2: per-chunk accumulation -------------------- #
    chunk_ids = db_manager.get_pending_chunk_ids()
    process_chunks_with_intervals(
        interval_files,
        chunk_ids=chunk_ids,
        total_reciprocal_points=total_pts,
        point_data_list=point_data_list,
        point_data_processor=point_data_processor,
        db_manager=db_manager, 
        client = client
    )

    logger.info("Completed compute_amplitudes_delta (refactored)")
