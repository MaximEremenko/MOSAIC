# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:40:08 2025

@author: Maksim Eremenko
"""

# unified_main.py  – works for 1-D, 2-D and 3-D
# -----------------------------------------------------------------------------
import os, json, logging
import numpy as np

# ─── common imports ----------------------------------------------------------
from utilities.logger_config import setup_logging
from utilities.dask_helpres import ensure_dask_client, shutdown_dask
from utilities.dask_client import get_client
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
from strategies.shape_strategies import IntervalShapeStrategy, CircleShapeStrategy, SphereShapeStrategy
from strategies.mask_strategies import EqBasedStrategy
from form_factors.form_factor_factory_producer import FormFactorFactoryProducer
# ─── logging -----------------------------------------------------------------
from multiprocessing import freeze_support


#from dask.distributed import Client, LocalCluster, get_client
    
def main():   
    # LOG_DIR = "/data/mve/MOSAIC_wsl/tests/config_3D"
    # job_extra = [
    # "-cwd",
    # "-V",
    # os.environ["DASK_GPU"],
    # os.environ["DASK_PE"],
    # os.environ["DASK_HOST"],
    # f"-o {LOG_DIR}/worker.o.$JOB_ID.$TASK_ID",
    # f"-e {LOG_DIR}/worker.e.$JOB_ID.$TASK_ID",
    # ]
    
    # client = ensure_dask_client(
    # backend=os.getenv("DASK_BACKEND", "sge"),
    # max_workers=int(os.getenv("DASK_MAX_WORKERS", 4)),
    # threads_per_worker=int(os.getenv("DASK_THREADS_PER_WORKER", 4)),
    # gpu=int(os.getenv("GPUS_PER_JOB", 1)),
    # worker_dashboard=False,
    # job_extra_directives=job_extra,
    # python="/data/mve/venvs/mosaic/bin/python",
    # scheduler_options={"host": "0.0.0.0"},
    # )
    os.environ["DASK_WORKER_DASHBOARD"] = "0"
    os.environ["DASK_BACKEND"] =  "local"
    #os.environ["DASK_BACKEND"]  = "single-threaded"
    os.environ["DASK_MAX_WORKERS"] =  "2"
    os.environ["DASK_THREADS_PER_WORKER"] =  "14" 
    os.environ["DASK_PROCESSES"] = "0"   # or make sure ensure_dask_client uses processes=False
    
    client = get_client()
    if client is not None:
        client.wait_for_workers(int(os.getenv("DASK_MAX_WORKERS", 1)), timeout="120s")
    setup_logging()
    log = logging.getLogger("app")  
    #client.config.set(scheduler="synchronous")
    # ─── helpers -----------------------------------------------------------------
    def build_mask_strategy(dim, peak_info):
        if dim == 1:
            return IntervalShapeStrategy(peak_info)
        if dim == 2:
            return CircleShapeStrategy(peak_info)
        else: dim == 3
        # r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1876)))
        # r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.202)))
        # condition = """
        # ( Min(Abs(Mod(Abs(h),2) - 1.5), 2 - Abs(Mod(Abs(h),2) - 1.5))**2
        # + Min(Abs(Mod(Abs(k),2) - 0.5), 2 - Abs(Mod(Abs(k),2) - 0.5))**2 >=  ({r})**2 )
        # |
        # ( Min(Abs(Mod(Abs(h),2) - 0.5), 2 - Abs(Mod(Abs(h),2) - 0.5))**2
        # + Min(Abs(Mod(Abs(k),2) - 1.5), 2 - Abs(Mod(Abs(k),2) - 1.5))**2 >=  ({r})**2 )
        # """.strip().format(r=r_val)
        # condition = """(((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= ({r1})**2)
        #  """.strip().format(r1=r1_val)
            #0    
        #rods(0.5h,0.5k,l) - spheres(0.5h,0.5k,0.5l)
        # condition = """
        #     (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= ({r1})**2) &
        #     (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= ({r2})**2)
        # """.strip().format(r1=r1_val, r2=r2_val)
        #     #1    
        #spheres(0.5h,0.5k,0.5l) input_parameters_LiFeO2_404040_out_antisite_swap_30_70_sph
        # r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1876)))
        # r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.202)))
        # condition = """
        #     (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) < ({r2})**2)
        # """.strip().format(r1=r1_val, r2=r2_val)        
        #spheres(0.5h,0.5k,0.5l)
        r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1576)))
        r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.1576)))
        condition = """
            (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) < ({r2})**2)
        """.strip().format(r1=r1_val, r2=r2_val)    
        # #     #2    
        # #>rods(0.5h,0.5k,l) and spheres(0.5h,0.5k,0.5l)
        # condition = """
        #      (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) > ({r1})**2) &
        #      (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) > ({r2})**2)
        # """.strip().format(r1=r1_val, r2=r2_val)
        # r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.0)))
        # #     #3    
        # #all points
        # r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.0)))
        # condition = """
        #      (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= ({r2})**2)
        # """.strip().format(r2=r2_val)

        #     #4   
        #>rods(0.5h,0.5k,l) and spheres(0.5h,0.5k,0.5l)
        # r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1876)))
        # r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.2501)))
        # condition = """
        #      (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) > ({r1})**2) |
        #      (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) > ({r2})**2)
        # """.strip().format(r1=r1_val, r2=r2_val)
        
        
        # condition = """
        # (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= ({r_1})**2) and
        # (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= ({r_2})**2)
        # """.strip().format(r1=r1_val)
        
        # condition = (
        #     "(cos(pi*h)+cos(pi*k)+cos(pi*l) > -0.5025 and "
        #     "cos(pi*h)+cos(pi*k)+cos(pi*l) < 0.5025)"
        # )
        return EqBasedStrategy(condition)
        #return SphereShapeStrategy(peak_info) #EqBasedStrategy(condition)
    
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
    #pproc = (pfactory.create_processor(p_h5,  source_type="hdf5", hdf5_file_path=p_h5)
    #         if False #os.path.exists(p_h5)
    #         else 
    pproc = pfactory.create_processor(p_json, source_type="file",  hdf5_file_path=p_h5)
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
    cell_ids = cfg_proc.get_cell_ids()
    # ─── 4. point grid -----------------------------------------------------------
    work_dir = struct["working_directory"]
    out_dir  = os.path.join(working_path, work_dir, "processed_point_data")
    os.makedirs(out_dir, exist_ok=True)
    
    rspace = parameters["rspace_info"]
    post_mode = (
        rspace.get("mode")
        or rspace.get("postprocess_mode")
        or rspace.get("postprocessing_mode")
        or "displacement"
    )
    parameters["hdf5_file_path"] = os.path.join(out_dir, "point_data.hdf5")
    
    pt_proc = PointProcessorFactory.create_processor(
        rspace["method"], parameters,
        average_structure=dict(average_coords=avg_coords, elements=elements,
                               refnumbers=refnumbers, vectors=vectors,
                               metric=metric, supercell=supercell, cell_ids=cell_ids))
    pt_proc.process_parameters()
    pgrid = pt_proc.get_point_data()
    
    saver = RIFFTInDataSaver(out_dir, "hdf5")
    pdp   = PointDataProcessor(data_saver=saver,
                               save_rifft_coordinates=rspace.get("save_rifft_coordinates", False))
    pdp.process_point_data(pgrid)
    
    # ─── 5. database -------------------------------------------------------------
    db = DatabaseManager(os.path.join(out_dir, "point_reciprocal_space_associations.db"), dim)
    
    _ = db.insert_point_data_batch([{
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
    #db.associate_point_reciprocal_space_batch([(pid, rid) for pid in point_ids for rid in rs_ids])
    
    unique_chunks = np.unique(pgrid.chunk_ids)
    db.insert_interval_chunk_status_batch(
        [(rs_id, int(chunk), 0)                 # saved = 0
         for rs_id in rs_ids
         for chunk  in unique_chunks]
    )
    
    
    padded_rs = [pad_interval(d, dim) for d in compact_rs]
    
    # ─── 7. Δ-amplitude stage ----------------------------------------------------
    # chunk-level bookkeeping
    unsaved = db.get_unsaved_interval_chunks()          # [(interval_id, chunk_id), …]
    ch_need = sorted({c for _, c in unsaved})            # chunks to process
    rs_need = sorted({r for r, _ in unsaved})            # intervals to process

    pt_rows = []
    for c in ch_need:
        pt_rows.extend(db.get_point_data_for_chunk(c))
    
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
        "average_coords" : avg_coords.to_numpy(),
        "cells_origin"   : cells_origin.to_numpy(),
        "elements"       : elements.to_numpy(),
        "refnumbers"     : refnumbers.to_numpy(),
        "vectors"        : vectors,
        "supercell"      : supercell,
        "postprocessing_mode": "chem",
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

    #client = ensure_dask_client(max_workers=2, processes=True)
   
    for c in range(parameters["rspace_info"]["num_chunks"]):
        post.process_chunk(c, saver, client, output_dir=out_dir)
    db.close()
    shutdown_dask()
    log.info("âœ“ %d-D workflow finished", dim)
if __name__ == "__main__":
    freeze_support()          # makes .exe builds happy
    main()
