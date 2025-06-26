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
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

import numpy as np
from dask import compute, delayed

from utilities.cunufft_wrapper import execute_cunufft, execute_inverse_cunufft
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from managers.database_manager import DatabaseManager
from processors.rifft_grid_generator import (
    GridGenerator1D,
    GridGenerator2D,
    GridGenerator3D,
)
from processors.point_data_processor import PointDataProcessor

# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------- #

from dask.distributed import Client, LocalCluster, Lock, get_client
import platform
import sys


from time import perf_counter
from contextlib import contextmanager
@contextmanager
def _timed(label: str):
    t0 = perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.3f s", label, perf_counter() - t0)

def ensure_dask_client(max_threads: int = 8) -> Client:
    """
    Re-use the existing dask client or spin up a local one.
    On Windows interactive sessions we fall back to threads
    (processes=True would hit the spawn RuntimeError).
    """
    try:
        return get_client()
    except ValueError:
        use_processes = (
            platform.system() != "Windows"            # everything except Windows
            or (not hasattr(sys, "ps1")               # Windows but *not* interactive
                and __name__ == "__main__")           # i.e., run from a script
        )

        cluster = LocalCluster(
            n_workers=max_threads,
            threads_per_worker=1,
            processes=use_processes,
            silence_logs=True,
        )
        return Client(cluster)

client = ensure_dask_client(2)
 
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
# try:
#     from numba import njit, prange           # optional dependency
#     NUMBA_OK = True
# except ImportError:
#     NUMBA_OK = False


# def _generate_grid_py(
#     dimensionality: int,
#     step_sizes: np.ndarray,
#     centre: np.ndarray,
#     dists: np.ndarray,
# ):
#     """Pure-NumPy reference implementation (unchanged semantics)."""
#     eps = 1e-8
#     axes = []
#     for i in range(dimensionality):
#         step = step_sizes[i]
#         dist = dists[i]
#         if step <= 0 or dist <= step:
#             axis = np.array([0.0], dtype=float)
#         else:
#             axis = np.arange(-dist, dist + step - eps, step, dtype=float)
#             if axis.size == 0:
#                 axis = np.array([0.0], dtype=float)
#         axes.append(axis)

#     if all(ax.size == 1 for ax in axes):          # single-point shortcut
#         return centre.reshape(1, dimensionality), np.ones(dimensionality, dtype=int)

#     mesh = np.meshgrid(*axes, indexing="ij")
#     pts  = np.stack(mesh, axis=-1).reshape(-1, dimensionality) + centre
#     return pts, np.array(mesh[0].shape, dtype=int)


# if NUMBA_OK:
#     @njit(cache=True, fastmath=True)
#     def _generate_grid_numba(
#         dimensionality: int,
#         step_sizes: np.ndarray,
#         centre: np.ndarray,
#         dists: np.ndarray,
#     ):
#         """Numba-compatible version (no meshgrid, no Python lists)."""
#         eps = 1e-8

#         # ------------------------------------------------------------------
#         # build axes & total size
#         # ------------------------------------------------------------------
#         lens = np.empty(dimensionality, dtype=np.int64)

#         for i in range(dimensionality):
#             step = step_sizes[i]
#             dist = dists[i]
#             if step <= 0 or dist <= step:
#                 lens[i] = 1
#             else:
#                 n = int(((dist * 2.0) / step) + 1.5)      # round-safe
#                 lens[i] = n if n > 0 else 1

#         total_pts = 1
#         for i in range(dimensionality):
#             total_pts *= lens[i]

#         # ------------------------------------------------------------------
#         # allocate result & fill with nested loops
#         # ------------------------------------------------------------------
#         pts = np.empty((total_pts, dimensionality), dtype=np.float64)

#         # pre-compute axis values in contiguous arrays
#         ax0 = np.zeros(lens[0], dtype=np.float64)
#         if lens[0] > 1:
#             start = -dists[0]
#             step  = step_sizes[0]
#             for k in range(lens[0]):
#                 ax0[k] = start + k * step

#         if dimensionality >= 2:
#             ax1 = np.zeros(lens[1], dtype=np.float64)
#             if lens[1] > 1:
#                 start = -dists[1]
#                 step  = step_sizes[1]
#                 for k in range(lens[1]):
#                     ax1[k] = start + k * step

#         if dimensionality == 3:
#             ax2 = np.zeros(lens[2], dtype=np.float64)
#             if lens[2] > 1:
#                 start = -dists[2]
#                 step  = step_sizes[2]
#                 for k in range(lens[2]):
#                     ax2[k] = start + k * step

#         # nested loops – unrolled for 1-, 2- and 3-D
#         idx = 0
#         if dimensionality == 1:
#             for i0 in range(lens[0]):
#                 pts[idx, 0] = centre[0] + ax0[i0]
#                 idx += 1

#         elif dimensionality == 2:
#             for i0 in range(lens[0]):
#                 for i1 in range(lens[1]):
#                     pts[idx, 0] = centre[0] + ax0[i0]
#                     pts[idx, 1] = centre[1] + ax1[i1]
#                     idx += 1

#         else:   # 3-D
#             for i0 in range(lens[0]):
#                 for i1 in range(lens[1]):
#                     for i2 in range(lens[2]):
#                         pts[idx, 0] = centre[0] + ax0[i0]
#                         pts[idx, 1] = centre[1] + ax1[i1]
#                         pts[idx, 2] = centre[2] + ax2[i2]
#                         idx += 1

#         return pts, lens.astype(np.int64)

#     # expose the jit’d version
#     _generate_grid = _generate_grid_numba      # type: ignore[misc]

# else:
#     # fallback – still works, just slower
#     _generate_grid = _generate_grid_py         # type: ignore[misc]


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

# import time, itertools
# t0 = time.time()
# _ = generate_rifft_grid_sync(chunk_data)
# logger.debug("RIFFT grid gen took %.3f s", time.time() - t0)

#def generate_rifft_grid_sync(chunk_data: List[dict]):
#    return generate_rifft_grid(chunk_data)#)[0]
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
    ff_factory,
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
    retries  = int(parameters.get("interval_retries", DEFAULT_INTERVAL_RETRIES))
    use_coeff = "coeff" in parameters
    coeff_val = parameters.get("coeff")

    # ------------------------------------------------------------------ task body
    def _handle_interval(iv: dict) -> Path | None:
        """Executed on a worker; returns final .npz path or None if skipped."""
        for attempt in range(retries + 1):
            try:
                # ---- build q-grid with masking --------------------------------
                q_grid = generate_q_space_grid_sync(iv, B_, mask_params, MaskStrategy, supercell)
                if q_grid.size == 0:
                    logger.debug("Interval %s masked out – skipped", iv.get("id", "?"))
                    return None

                # ---- element/coeff NUFFT --------------------------------------
                results: List[tuple] = []
                if use_coeff:
                    results.append(
                        _process_interval_coeff(iv, q_grid, coeff_val, original_coords, cells_origin)
                    )
                else:
                    for el in unique_elements:
                        t = _process_interval_element(
                            iv, q_grid, el,
                            original_coords, cells_origin,
                            elements_arr, charge, ff_factory,
                        )
                        if t is not None:
                            results.append(t)

                if not results:
                    return None

                task = aggregate_interval_tasks(results, use_coeff)

                # ---- atomic save ----------------------------------------------
                final_path = out_dir / f"interval_{task.irecip_id}.npz"
                with tempfile.NamedTemporaryFile(
                    dir=out_dir,
                    prefix=f"interval_{task.irecip_id}_",
                    suffix=".npz",
                    delete=False,
                ) as tf:
                    tmp_path = Path(tf.name)
                    np.savez_compressed(
                        tf,
                        irecip_id=task.irecip_id,
                        element=task.element,
                        q_grid=task.q_grid,
                        q_amp=task.q_amp,
                        q_amp_av=task.q_amp_av,
                    )
                tmp_path.replace(final_path)
                logger.debug("Saved interval %s → %s", task.irecip_id, final_path)
                return final_path

            except Exception as exc:
                logger.warning(
                    "Interval %s failed (attempt %d/%d): %s",
                    iv.get("id", "?"), attempt + 1, retries + 1, exc,
                )
                if attempt == retries:
                    logger.error("Interval %s exhausted retries – skipping.", iv.get("id", "?"))
                    return None
                # else retry

    # ------------------------------------------------------------------ dispatch
    futures = [client.submit(_handle_interval, iv, pure=False)  # pure=False → keep retries independent
               for iv in reciprocal_space_intervals]

    # gather results back to driver
    paths = client.gather(futures)

    written_files = [p for p in paths if p is not None]
    logger.info(
        "Stage-1 complete: %d interval files written (%d skipped)",
        len(written_files), len(paths) - len(written_files),
    )
    return written_files



# ════════════════════════════════════════════════════════════════════════════
#  Stage-2: read cached intervals, apply to every chunk_id
# ════════════════════════════════════════════════════════════════════════════
@delayed
def _interval_path_delayed(path: Path):
    return path 

# def _load_interval_delayed(path: Path) -> IntervalTask:
#     """Read an .npz on demand (mmap to keep memory down)."""
#     dat = np.load(path, mmap_mode="r")
#     return IntervalTask(
#         int(dat["irecip_id"]),
#         str(dat["element"]),
#         dat["q_grid"],
#         dat["q_amp"],
#         dat["q_amp_av"],
#     )
from dask.distributed import Lock       # cluster-wide mutex
# ---------------------------------------------------------------------------


# @delayed
# def _process_chunk_id_delayed(
#     chunk_id: int,
#     iv_path: Path,
#     rifft_saver: RIFFTInDataSaver,
#     point_data_list: List[dict],
#     total_reciprocal_points: int,
#     point_data_processor: PointDataProcessor,
#     db_path: str,
# ):
#     dat = np.load(iv_path, mmap_mode="r")
#     task = IntervalTask(
#         int(dat["irecip_id"]), str(dat["element"]),
#         dat["q_grid"], dat["q_amp"], dat["q_amp_av"]
#     )
#     # ------------------------------------------------------------------ heavy NUFFT (unchanged)
#     chunk_data = [pd for pd in point_data_list if pd["chunk_id"] == chunk_id]
#     if not chunk_data:
#         logger.warning("Chunk %s has no point data; skipping", chunk_id)
#         return

#     rifft_grid, grid_shape_nd = _build_rifft_grid_locally(chunk_data)
#     if rifft_grid.size == 0:
#         logger.warning("Chunk %s produced empty RIFFT grid", chunk_id)
#         return
#     with _timed(f"inv-NUFFT | chunk {chunk_id} | iv {task.irecip_id}"):
#         r_partial = execute_inverse_cunufft(
#             q_coords=task.q_grid,
#             c=task.q_amp - task.q_amp_av,
#             real_coords=rifft_grid,
#             eps=1e-12,
#         )

#     # ------------------------------------------------------------------ serialised I/O section
#     # ONE distributed lock per chunk; same name → same lock across the cluster
#     #with chunk_mutex(chunk_id):
#     with _timed(f"write HDF5 | chunk {chunk_id}"), chunk_mutex(chunk_id):
#         fname_amp   = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes")
#         fname_shape = point_data_processor.data_saver.generate_filename(chunk_id, "_shapeNd")
#         fname_tot   = point_data_processor.data_saver.generate_filename(
#             chunk_id, "_amplitudes_ntotal_reciprocal_space_points"
#         )
#         fname_nrec  = point_data_processor.data_saver.generate_filename(
#             chunk_id, "_amplitudes_nreciprocal_space_points"
#         )

#         # write shape once
#         try:
#             point_data_processor.data_saver.load_data(fname_shape)
#         except FileNotFoundError:
#             point_data_processor.data_saver.save_data({"shapeNd": grid_shape_nd}, fname_shape)

#         # write total-points once
#         try:
#             point_data_processor.data_saver.load_data(fname_tot)
#         except FileNotFoundError:
#             point_data_processor.data_saver.save_data(
#                 {"ntotal_reciprocal_points": total_reciprocal_points}, fname_tot
#             )

#         # load existing amplitudes (if any)
#         try:
#             current = point_data_processor.data_saver.load_data(fname_amp).get("amplitudes", None)
#             nrec    = point_data_processor.data_saver.load_data(fname_nrec).get(
#                 "nreciprocal_space_points", 0
#             )
#         except FileNotFoundError:
#             current, nrec = None, 0

#         # accumulate
#         if current is None:
#             current = r_partial
#             nrec    = task.q_grid.shape[0]
#         else:
#             if task.q_grid.shape[1] > 2 and np.all(np.round(task.q_grid[:, 2], 8) != 0):
#                 current[:, 1] += r_partial + np.conj(r_partial)
#                 nrec += task.q_grid.shape[0] * 2
#             else:
#                 current[:, 1] += r_partial
#                 nrec += task.q_grid.shape[0]

#         point_data_processor._save_chunk_data(chunk_id, None, current, nrec)

#     # ------------------------------------------------------------------ DB update (thread/process local)
#     from managers.database_manager import create_db_manager_for_thread
#     local_db = create_db_manager_for_thread(db_path)
#     local_db.update_saved_status_for_chunk_or_point(task.irecip_id, None, chunk_id, 1)
#     local_db.close()

#     logger.info("Chunk %s updated for interval %s", chunk_id, task.irecip_id)

import time
TIMER = time.perf_counter          # monotonic and fast

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
    db.update_saved_status_for_chunk_or_point(task.irecip_id, None, chunk_id, 1)
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
    rifft_grid, grid_shape_nd = _process_chunk([
        dict(coordinates          = atoms.coordinates[i],
             dist_from_atom_center= atoms.dist_from_atom_center[i],
             step_in_frac         = atoms.step_in_frac[i])
        for i in range(atoms.shape[0])
    ])
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
) -> None:

    client = ensure_dask_client()

    # 1️⃣  pack everything into *one* compact block
    rec = _point_list_to_recarray(point_data_list)

    # 2️⃣  scatter only the slice that belongs to each chunk
    chunk_futs = {
        cid: client.scatter(rec[rec.chunk_id == cid], broadcast=False, hash=False)
        for cid in chunk_ids
    }

    # 3️⃣  scatter paths (trivial) & create tasks
    iv_futs = {p: client.scatter(p, broadcast=False) for p in interval_files}

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
    ]

    logger.info("Submitting %d interval×chunk tasks …", len(tasks))
    client.gather(tasks)                   # block until everything finishes
    logger.info("Stage-2 finished")


# def process_chunks_with_intervals(
#     interval_files: Iterable[Path],
#     *,
#     chunk_ids: Iterable[int],
#     total_reciprocal_points: int,
#     rifft_saver: RIFFTInDataSaver,
#     point_data_list: List[dict],
#     point_data_processor: PointDataProcessor,
#     db_manager: DatabaseManager,
#     dask_max_threads: int = 8,
# ):
#     #client = ensure_dask_client(dask_max_threads)   # local or external

#     # ------------------------------------------------------------------
#     # 2️⃣  one _load_interval_delayed node per .npz (reuse across chunks)
#     # ------------------------------------------------------------------
#     interval_nodes = {p: _interval_path_delayed(p) for p in interval_files}

#     graph = [
#             _process_chunk_id_delayed(
#                 cid,
#                 iv_node,                       # Path, not IntervalTask
#                 rifft_saver,
#                 point_data_list,
#                 total_reciprocal_points,
#                 point_data_processor,
#                 db_path=db_manager.db_path,
#             )
#             for cid in chunk_ids
#             for iv_node in interval_nodes.values()
#         ]

#     if not graph:
#         logger.warning("No (interval × chunk) work to do.")
#         return

#     # ------------------------------------------------------------------
#     #   let the distributed scheduler handle parallelism & Locks
#     # ------------------------------------------------------------------
#     client.compute(graph, sync=True)   # blocks until all done
#     logger.info("Finished %d interval×chunk tasks", len(graph))
# def process_chunks_with_intervals(
#     interval_files          : Iterable[Path],
#     *,                                         # keyword-only from here
#     chunk_ids               : Iterable[int],
#     total_reciprocal_points : int,
#     point_data_list         : List[dict],
#     point_data_processor    : PointDataProcessor,
#     db_manager              : DatabaseManager,
# ):
#     """
#     Builds a **tiny** task graph: one Future for every (chunk, interval) pair.
#     No giant literals are embedded any more.
#     """
#     client = ensure_dask_client()

#     # 1️⃣  scatter point-data once per chunk
#     chunk_futs = {
#         cid: client.scatter(
#             [pd for pd in point_data_list if pd["chunk_id"] == cid],
#             broadcast=False,
#             hash=False,              # skip expensive hashing of big lists
#         )
#         for cid in chunk_ids
#     }

#     # 2️⃣  scatter every .npz path (almost free – just a Path object)
#     iv_futs = {p: client.scatter(p, broadcast=False) for p in interval_files}

#     # 3️⃣  fire off the computation
#     tasks = [
#         client.submit(
#             _process_chunk_id,
#             cid,
#             iv_futs[p],
#             chunk_futs[cid],
#             total_reciprocal_points,
#             point_data_processor,
#             db_manager.db_path,
#             pure=False                # → each call is unique
#         )
#         for cid in chunk_ids
#         for p   in interval_files
#     ]

#     client.gather(tasks)              # block until all done
#     logger.info("Finished %d interval×chunk tasks", len(tasks))

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
    max_threads = int(2)

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
    )

    # ------------------ Stage-2: per-chunk accumulation -------------------- #
    chunk_ids = db_manager.get_pending_chunk_ids()
    rifft_saver = RIFFTInDataSaver(output_dir=output_dir, file_extension="hdf5")

    process_chunks_with_intervals(
        interval_files,
        chunk_ids=chunk_ids,
        total_reciprocal_points=total_pts,
        #rifft_saver=rifft_saver,
        point_data_list=point_data_list,
        point_data_processor=point_data_processor,
        db_manager=db_manager, 
        #dask_max_threads = max_threads,
    )

    logger.info("Completed compute_amplitudes_delta (refactored)")
