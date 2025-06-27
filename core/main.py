# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:40:08 2025

@author: Maksim Eremenko
"""

# unified_main.py  – works for 1-D, 2-D and 3-D
# -----------------------------------------------------------------------------
import os, json, logging, numpy as np, copy, contextlib, itertools
#os.chdir("../core")                       # repo root
#os.environ["OMP_NUM_THREADS"] = "32"

# ─── common imports ----------------------------------------------------------
from utilities.logger_config import setup_logging
from utilities.rmc_neutron_scl import rmc_neutron_scl_
from utilities.utils import determine_configuration_file_type

from factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from factories.parameters_processor_factory import ParametersProcessorFactoryProvider
from factories.point_processor_factory import PointProcessorFactory

from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from processors.point_data_processor import PointDataProcessor
from processors.point_data_reciprocal_space_manager import ReciprocalSpaceIntervalManager
from managers.database_manager import DatabaseManager
from processors.point_data_postprocessing_processor import PointDataPostprocessingProcessor
from processors.amplitude_delta_calculator import compute_amplitudes_delta

# shape / mask per dimension
from strategies.shape_strategies import IntervalShapeStrategy, CircleShapeStrategy
from strategies.mask_strategies import EqBasedStrategy
from form_factors.form_factor_factory_producer import FormFactorFactoryProducer
# -----------------------------------------------------------------------------

# ─── logging -----------------------------------------------------------------
from multiprocessing import freeze_support


from dask.distributed import Client, LocalCluster, get_client

def ensure_dask_client(max_workers: int = 8,
                   *,
                   processes: bool = True,
                   threads_per_worker: int = 4) -> Client:
    """
    Get (or create) a Dask Client with at least `max_workers` *process* workers
    unless `processes=False` is explicitly requested.
    """
    try:
        client = get_client()
        cluster = client.cluster
        if isinstance(cluster, LocalCluster) and len(cluster.workers) < max_workers:
            cluster.scale(max_workers)
        return client
    except ValueError:        # no running client
        cluster = LocalCluster(
            n_workers=max_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            protocol="tcp",   
            #silence_logs="error",
        )
        return Client(cluster)
    
    
def shutdown_dask():
    """
    Close the current Dask `Client` (and its underlying cluster) **if** one
    exists.  Safe to call multiple times – it becomes a no-op when no client
    is active.
    """
    try:
        client = get_client()      # raises ValueError when there is no client
        # First close the client (disconnects workers & scheduler) …
        client.close()             # <- frees sockets, tasks, futures
        # … then the cluster itself (optional, but frees ports & temp dirs)
        if hasattr(client, "cluster"):
            client.cluster.close()
        print("✅ Dask client closed.")
    except ValueError:
        # Nothing to do – no client was running
        print("ℹ️  No active Dask client.")
    
    #shutdown_dask()
    
def main():   
    #shutdown_dask()
    client = ensure_dask_client(8)              # <-- ONLY HERE
    #shutdown_dask()
    setup_logging()
    log = logging.getLogger("app")    
    # ─── helpers -----------------------------------------------------------------
    def build_mask_strategy(dim, peak_info):
        if dim == 1:
            return IntervalShapeStrategy(peak_info)
        if dim == 2:
            return CircleShapeStrategy(peak_info)
        else: dim == 3
        condition = (
            "(cos(pi*h)+cos(pi*k)+cos(pi*l) > -0.5025 and "
            "cos(pi*h)+cos(pi*k)+cos(pi*l) < 0.5025)"
        )
        return EqBasedStrategy(condition)
    
    def pad_interval(d, dim):
        """Return dict with BOTH styles of keys for any dimension."""
        base = {}
        if "h_range" in d:
            base["h_range"] = d["h_range"]
            base["h_start"], base["h_end"] = d["h_range"]
        if dim >= 2:
            base["k_range"] = d.get("k_range", (0, 0))
            base["k_start"], base["k_end"] = base["k_range"]
        else:
            base["k_range"] = (0, 0); base["k_start"] = base["k_end"] = 0
        if dim == 3:
            base["l_range"] = d.get("l_range", (0, 0))
            base["l_start"], base["l_end"] = base["l_range"]
        else:
            base["l_range"] = (0, 0); base["l_start"] = base["l_end"] = 0
        return base
    
    # ─── 1. run-level paths ------------------------------------------------------
    with open("run_parameters.json") as fh:
        working_path = os.path.abspath(json.load(fh)["working_path"]) + os.sep
    
    # ─── 2. parameters -----------------------------------------------------------
    p_json = os.path.join(working_path, "input_parameters.json")
    p_h5   = os.path.join(working_path, "parameters.hdf5")
    pfactory = ParametersProcessorFactoryProvider().get_factory()
    pproc = (pfactory.create_processor(p_h5,  source_type="hdf5", hdf5_file_path=p_h5)
             if False #os.path.exists(p_h5)
             else pfactory.create_processor(p_json, source_type="file",  hdf5_file_path=p_h5))
    pproc.process()
    parameters = pproc.get_parameters()
    
    dim = int(parameters["structInfo"]["dimension"])  # 1 / 2 / 3
    log.info("Running %d-D workflow", dim)
    
    # ─── 3. structure ------------------------------------------------------------
    struct = parameters["structInfo"]
    cfg_path = os.path.join(working_path, struct["filename"])
    cfg_type = determine_configuration_file_type(struct["filename"])
    cfg_proc = ConfigurationProcessorFactoryProvider.get_factory(cfg_type)\
              .create_processor(cfg_path, "calculate")
    cfg_proc.process()
    
    vectors      = cfg_proc.get_vectors()
    metric       = cfg_proc.get_metric()
    supercell    = cfg_proc.get_supercell()
    orig_coords  = cfg_proc.get_coordinates()
    avg_coords   = cfg_proc.get_average_coordinates()
    elements     = cfg_proc.get_elements()
    refnumbers   = cfg_proc.get_refnumbers()
    cells_origin = cfg_proc.get_cells_origin()
    coeff        = elements.apply(lambda el: rmc_neutron_scl_(el)[0])
    
    # ─── 4. point grid -----------------------------------------------------------
    work_dir = struct["working_directory"]
    out_dir  = os.path.join(working_path, work_dir, "processed_point_data")
    os.makedirs(out_dir, exist_ok=True)
    
    rspace = parameters["rspace_info"]
    parameters["hdf5_file_path"] = os.path.join(out_dir, "point_data.hdf5")
    
    pt_proc = PointProcessorFactory.create_processor(
        rspace["method"], parameters,
        average_structure=dict(average_coords=avg_coords, elements=elements,
                               refnumbers=refnumbers, vectors=vectors,
                               metric=metric, supercell=supercell))
    pt_proc.process_parameters()
    pgrid = pt_proc.get_point_data()
    
    saver = RIFFTInDataSaver(out_dir, "hdf5")
    pdp   = PointDataProcessor(data_saver=saver,
                               save_rifft_coordinates=rspace.get("save_rifft_coordinates", False))
    pdp.process_point_data(pgrid)
    
    # ─── 5. database -------------------------------------------------------------
    db = DatabaseManager(os.path.join(out_dir, "point_reciprocal_space_associations.db"), dim)
    
    point_ids = db.insert_point_data_batch([{
        "central_point_id": int(pgrid.central_point_ids[i]),
        "coordinates":      pgrid.coordinates[i].tolist(),
        "dist_from_atom_center": pgrid.dist_from_atom_center[i].tolist(),
        "step_in_frac":     pgrid.step_in_frac[i].tolist(),
        "chunk_id":         int(pgrid.chunk_ids[i]),
        "grid_amplitude_initialized": int(pgrid.grid_amplitude_initialized[i])
    } for i in range(pgrid.central_point_ids.size)])
    
    # ─── 6. reciprocal-space boxes ----------------------------------------------
    recip_h5 = os.path.join(out_dir, "point_reciprocal_space_data.hdf5")
    r_mgr = ReciprocalSpaceIntervalManager(recip_h5, parameters, supercell)
    r_mgr.process_reciprocal_space_intervals()
    
    
    
    compact_rs = []
    for d in r_mgr.reciprocal_space_intervals:
        entry = {"h_range": d["h_range"]}
        if dim >= 2: entry["k_range"] = d.get("k_range", (0, 0))
        if dim == 3: entry["l_range"] = d.get("l_range", (0, 0))
        compact_rs.append(entry)
    
    rs_ids = db.insert_reciprocal_space_interval_batch(compact_rs)
    db.associate_point_reciprocal_space_batch([(pid, rid) for pid in point_ids for rid in rs_ids])
    
    unique_chunks = np.unique(pgrid.chunk_ids)
    db.insert_interval_chunk_status_batch(
        [(rs_id, int(chunk), 0)                 # saved = 0
         for rs_id in rs_ids
         for chunk  in unique_chunks]
    )
    
    
    padded_rs = [pad_interval(d, dim) for d in compact_rs]
    
    # ─── 7. Δ-amplitude stage ----------------------------------------------------
    unsaved = db.get_unsaved_associations()
    
    pt_need = list({p for p, _ in unsaved})
    rs_need = list({r for _, r in unsaved})
    pt_rows = db.get_point_data_for_point_ids(pt_need)
    
    placeholders = ",".join("?" * len(rs_need))
    db.cursor.execute(f"SELECT * FROM ReciprocalSpaceInterval WHERE id IN ({placeholders})",
                      rs_need)
    rs_full = [pad_interval({
        "h_range": (row[1], row[2]),
        "k_range": (row[3], row[4]) if dim >= 2 else (0, 0),
        "l_range": (row[5], row[6]) if dim == 3 else (0, 0)
    }, dim) | {"id": row[0]}            # merge dicts (py3.9+)
               for row in db.cursor.fetchall()]
    
    params = {
        "reciprocal_space_intervals"     : rs_full,
        "reciprocal_space_intervals_all" : padded_rs,
        "point_data_list": [{
            "central_point_id": pd["central_point_id"],
            "coordinates":      pd["coordinates"],
            "dist_from_atom_center": pd["dist_from_atom_center"],
            "step_in_frac":     pd["step_in_frac"],
            "chunk_id":         pd["chunk_id"],
            "grid_amplitude_initialized": pd["grid_amplitude_initialized"],
            "id":               pd["central_point_id"],
        } for pd in pt_rows],
        "original_coords": orig_coords.to_numpy(),
        "cells_origin"   : cells_origin.to_numpy(),
        "elements"       : elements.to_numpy(),
        "vectors"        : vectors,
        "supercell"      : supercell,
    }
    if coeff is not None:
        params["coeff"] = coeff.to_numpy()
    
    
    
    if unsaved:
        mask_strategy = build_mask_strategy(dim, parameters["peakInfo"])
        ff_calc = FormFactorFactoryProducer.get_factory("neutron").create_calculator("default")
        compute_amplitudes_delta(
            parameters=params, FormFactorFactoryProducer=ff_calc,
            MaskStrategy=mask_strategy, MaskStrategyParameters=parameters["peakInfo"],
            db_manager=db, output_dir=out_dir, point_data_processor=pdp, client = client
        )

    # %%
    post = PointDataPostprocessingProcessor(db, pdp, params)
    
    #os.environ.setdefault("OMP_NUM_THREADS",  "1")
    #os.environ.setdefault("MKL_NUM_THREADS",  "1")
    #os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    #os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    # import platform, sys
    
    # def ensure_dask_client(max_workers: int = os.cpu_count() or 8,
    #                        *,
    #                        processes: bool | None = None,
    #                        threads_per_worker: int = 1,
    #                        dashboard: bool = False) -> Client:
    #     """
    #     Get an existing distributed.Client or spin up a LocalCluster that
    #     has *process* workers (so every worker can occupy its own CPU core).
    #     """
    #     try:
    #         client = get_client()
    #         clus   = client.cluster
    #         if isinstance(clus, LocalCluster) and len(clus.workers) < max_workers:
    #             clus.scale(max_workers)
    #         return client
    
    #     except ValueError:                                               # no client
    #         if processes is None:
    #             # use process workers unless the user runs *interactively* on Windows
    #             processes = not (platform.system() == "Windows" and hasattr(sys, "ps1"))
    
    #         cluster = LocalCluster(
    #             n_workers=max_workers,
    #             threads_per_worker=threads_per_worker,
    #             processes=processes,
    #             silence_logs="error",
    #             protocol="tcp",              # ←■■ force real processes
    #             dashboard_address=":8787" if dashboard else None,
    #             memory_limit="auto",
    #         )
    #         return Client(cluster)
    
    # # ─────────────────────────────────────────────────────────────────────────────
    # # 3. helper executed on each worker – unchanged except for imports inside
    # # ─────────────────────────────────────────────────────────────────────────────
    # def _run_process_chunk(db_path: str,
    #                        params_blob: bytes,
    #                        chunk_id: int,
    #                        saver_blob: bytes,
    #                        out_dir: str) -> str:
    #     """
    #     Re-create light objects on the worker and run
    #     PointDataPostprocessingProcessor.process_chunk for one chunk.
    #     """
    #     import pickle
    #     from managers.database_manager import DatabaseManager
    #     from processors.point_data_postprocessing_processor import \
    #          PointDataPostprocessingProcessor
    #     from processors.point_data_processor import PointDataProcessor
    
    #     params = pickle.loads(params_blob)
    #     saver  = pickle.loads(saver_blob)
    
    #     dim = int(params["vectors"].shape[1])
    #     db  = DatabaseManager(db_path, dim)           # read-only open
    
    #     pd_proc = PointDataProcessor(saver, False)
    #     post    = PointDataPostprocessingProcessor(db, pd_proc, params)
    
    #     post.process_chunk(chunk_id, saver, output_dir=out_dir)
    #     db.close()
    #     return f"chunk {chunk_id} ✔"
    
    # # ─────────────────────────────────────────────────────────────────────────────
    # # 4. driver – build and submit one task per chunk
    # # ─────────────────────────────────────────────────────────────────────────────
    # import pickle, dask
    # from dask.distributed import wait
    
    # n_chunks   = parameters["rspace_info"]["num_chunks"]
    # db_path    = os.path.join(out_dir, "point_reciprocal_space_associations.db")
    
    # params_blob = pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)
    # saver_blob  = pickle.dumps(saver,  protocol=pickle.HIGHEST_PROTOCOL)
    
    #client = ensure_dask_client(max_workers=n_chunks, processes=True)

    
    
    client = ensure_dask_client(max_workers=8, processes=True)
    #client.run(lambda: __import__("numba").set_num_threads(32))
    
    for c in range(parameters["rspace_info"]["num_chunks"]):
        post.process_chunk(c, saver, client, output_dir=out_dir)
    
    db.close()
    log.info("âœ“ %d-D workflow finished", dim)
if __name__ == "__main__":
    freeze_support()          # makes .exe builds happy
    main()
# client = ensure_dask_client(max_workers=n_chunks, processes=True)

# # ─── NEW: wait until all workers are up (with a safety timeout) ──────────
# client.wait_for_workers(n_workers=n_chunks, timeout="60s")

# # ---- quick sanity checks ------------------------------------------------
# info   = client.scheduler_info()
# pids   = client.run(lambda: os.getpid())   # {worker-addr: PID, …}
# print(" Workers live:", list(info["workers"]))
# print(" Distinct PIDs:", set(pids.values()))
# print(" nthreads per worker:", client.nthreads())     # {addr: 1, …}

# print("Workers online →", list(client.scheduler_info()["workers"]))

# print("Workers:", client.scheduler_info()["workers"].keys())

# futs = [
#     client.submit(
#         _run_process_chunk,
#         db_path,
#         params_blob,
#         cid,
#         saver_blob,
#         out_dir,
#         pure=False,          # ensure every chunk is executed
#     )
#     for cid in range(n_chunks)
# ]

# wait(futs)                      # block until all tasks finished
# for f in futs:
#     log.info(f.result())        # e.g. “chunk 5 ✔”

# db.close()
# log.info("✓ %d-D workflow finished", dim)