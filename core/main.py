# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:40:08 2025

@author: Maksim Eremenko
"""

# unified_main.py  – works for 1-D, 2-D and 3-D
# -----------------------------------------------------------------------------
import os, json, logging
from pathlib import Path
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
    os.environ["DASK_THREADS_PER_WORKER"] =  "16" 
    os.environ["DASK_PROCESSES"] = "0"   # or make sure ensure_dask_client uses processes=False
    
    client = get_client()
    if client is not None:
        client.wait_for_workers(int(os.getenv("DASK_MAX_WORKERS", 1)), timeout="120s")
    setup_logging()
    log = logging.getLogger("app")  
    #client.config.set(scheduler="synchronous")
    # ─── helpers -----------------------------------------------------------------
    def _get_mask_equation(peak_info):
        # Keep this permissive so older configs can work without edits.
        for key in ("mask_equation", "maskEquation", "equation", "condition"):
            val = peak_info.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None

    def _normalize_post_mode(mode):
        mode_norm = str(mode or "").strip().lower()
        if mode_norm in (
            "chemical",
            "chem",
            "checmical",
            "occupational",
            "occupancy",
            "occupantioal",
        ):
            return "chemical"
        return "displacement"

    def build_mask_strategy(dim, peak_info, post_mode="displacement"):
        eq = _get_mask_equation(peak_info)
        if eq is not None:
            return EqBasedStrategy(eq)
        if dim == 1:
            return IntervalShapeStrategy(peak_info)
        if dim == 2:
            return CircleShapeStrategy(peak_info)

        r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1876)))
        r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.2501)))
        if _normalize_post_mode(post_mode) == "displacement":
            condition = """
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= ({r1})**2) &
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= ({r2})**2)
            """.strip().format(r1=r1_val, r2=r2_val)
        else:
            condition = """
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) > ({r1})**2) &
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) > ({r2})**2)
            """.strip().format(r1=r1_val, r2=r2_val)
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

    def _first_present(d, keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                continue
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
        return None

    def _load_coefficients_from_file(
        *,
        coeff_path: str,
        dim: int,
        n_atoms: int,
        supercell: np.ndarray,
        vectors: np.ndarray,
        cells_origin: np.ndarray,
        log: logging.Logger,
    ) -> np.ndarray:
        arr = np.loadtxt(coeff_path, dtype=float)
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 0:
            return np.full((n_atoms,), float(arr))

        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)

        if arr.ndim == 1:
            if arr.size != n_atoms:
                raise ValueError(
                    f"Coefficient file '{coeff_path}' has {arr.size} values, "
                    f"but configuration has {n_atoms} atoms."
                )
            return arr

        if arr.ndim != 2:
            raise ValueError(
                f"Coefficient file '{coeff_path}' must be 1D or 2D, got shape {arr.shape}."
            )

        if dim != 2:
            raise ValueError(
                f"2D coefficient-matrix mapping is only supported for dim=2; got dim={dim}."
            )

        nx, ny = int(supercell[0]), int(supercell[1])
        if arr.shape not in {(ny, nx), (nx, ny)}:
            raise ValueError(
                f"Coefficient matrix shape {arr.shape} does not match supercell "
                f"(ny,nx)=({ny},{nx}) or (nx,ny)=({nx},{ny})."
            )

        invV = np.linalg.inv(np.asarray(vectors, float))
        frac = np.asarray(cells_origin, float) @ invV
        fx = np.asarray(frac[:, 0], float)
        fy = np.asarray(frac[:, 1], float)

        ix = np.round(fx * nx).astype(int)
        iy = np.round(fy * ny).astype(int)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        if arr.shape == (ny, nx):
            coeff = arr[iy, ix]
        else:
            coeff = arr[ix, iy]

        log.info(
            "Loaded coefficient matrix '%s' (shape %s) mapped onto %d atoms.",
            coeff_path, tuple(arr.shape), n_atoms,
        )
        return np.asarray(coeff, float)

    def _center_coefficients(coeff: np.ndarray, refnumbers: np.ndarray | None, mode) -> np.ndarray:
        coeff = np.asarray(coeff, float)
        if mode in (None, "", "none", False):
            return coeff
        mode_s = str(mode).strip().lower()
        if mode_s in ("global", "mean", "avg"):
            return coeff - float(np.mean(coeff))
        if mode_s in ("refnumber", "refnumbers", "site", "sites"):
            if refnumbers is None:
                return coeff - float(np.mean(coeff))
            ref = np.asarray(refnumbers)
            out = coeff.copy()
            for r in np.unique(ref):
                m = ref == r
                if np.any(m):
                    out[m] -= float(np.mean(coeff[m]))
            return out
        raise ValueError(f"Unknown coeff centering mode: {mode!r}")
    
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
    cell_ids = cfg_proc.get_cell_ids()

    # Default: neutron scattering length from internal table (one value per element).
    coeff = elements.apply(lambda el: rmc_neutron_scl_(el)[0])

    # Optional: per-atom coefficients from the config file itself (e.g. "Coeff"/"coeff").
    coeff_from_cfg = None
    if hasattr(cfg_proc, "get_coeff"):
        try:
            coeff_from_cfg = cfg_proc.get_coeff()
        except Exception:
            coeff_from_cfg = None
    coeff_source = _first_present(struct, ("coeff_source", "coeffSource")) or "auto"
    coeff_source = str(coeff_source).strip().lower()
    if coeff_source not in ("auto", "config", "file"):
        raise ValueError(f"Unsupported coeff_source={coeff_source!r} (use 'auto'|'config'|'file').")
    if coeff_source in ("auto", "config") and coeff_from_cfg is not None:
        coeff = coeff_from_cfg

    # Optional: per-atom coefficients from an external file (vector or 2D matrix).
    coeff_file = _first_present(
        struct,
        (
            "coeff_file",
            "coeff_filename",
            "coeff_path",
            "coefficients_file",
            "intensity_coeff_file",
            "intensity_coeff_filename",
        ),
    )
    if coeff_file is not None and coeff_source in ("auto", "file") and (coeff_from_cfg is None or coeff_source == "file"):
        coeff_path = Path(coeff_file)
        if not coeff_path.is_absolute():
            coeff_path = Path(working_path) / coeff_path
        coeff = _load_coefficients_from_file(
            coeff_path=str(coeff_path),
            dim=dim,
            n_atoms=int(len(elements)),
            supercell=np.asarray(supercell, int),
            vectors=np.asarray(vectors, float),
            cells_origin=np.asarray(cells_origin.to_numpy(), float),
            log=log,
        )

    coeff_raw = np.asarray(coeff.to_numpy() if hasattr(coeff, "to_numpy") else coeff, float)
    # ─── 4. point grid -----------------------------------------------------------
    work_dir = struct["working_directory"]
    out_dir  = os.path.join(working_path, work_dir, "processed_point_data")
    os.makedirs(out_dir, exist_ok=True)
    
    rspace = parameters["rspace_info"]
    post_mode = _first_present(rspace, ("mode", "postprocess_mode", "postprocessing_mode")) or "displacement"
    post_mode_norm = _normalize_post_mode(post_mode)
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
    
    chemical_filtered = bool(
        _first_present(rspace, ("chemical_filtered_ordering", "chemical_filtered"))
        or _first_present(struct, ("chemical_filtered_ordering", "chemical_filtered"))
    )
    use_coeff = bool(rspace.get("use_coeff", True)) or chemical_filtered

    coeff_center_mode = _first_present(
        rspace, ("coeff_center_by", "coeff_center_mode", "chemical_coeff_center_by")
    ) or ("global" if chemical_filtered else "none")
    coeff_arr = _center_coefficients(
        np.asarray(coeff_raw, float),
        refnumbers.to_numpy() if hasattr(refnumbers, "to_numpy") else np.asarray(refnumbers),
        coeff_center_mode,
    )

    base_params = {
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
        "rspace_info"    : rspace,
        "vectors"        : vectors,
        "supercell"      : supercell,
        "postprocessing_mode": post_mode_norm,
    }
    amp_params = dict(base_params)
    if use_coeff:
        amp_params["coeff"] = np.asarray(coeff_arr, float)
    if chemical_filtered:
        amp_params["original_coords"] = cells_origin.to_numpy()
        log.info("Chemical-filtered ordering enabled: using cells_origin as original_coords.")
    
    
    if unsaved:
        mask_strategy = build_mask_strategy(dim, parameters["peakInfo"], post_mode=post_mode_norm)
        ff_calc = FormFactorFactoryProducer.get_factory("neutron").create_calculator("default")
        compute_amplitudes_delta(
            parameters=amp_params, FormFactorFactoryProducer=ff_calc,
            MaskStrategy=mask_strategy, MaskStrategyParameters=parameters["peakInfo"],
            db_manager=db, output_dir=out_dir, point_data_processor=pdp, client = client
        )

    # %%
    run_post = bool(rspace.get("run_postprocessing", True))
    if run_post:
        post = PointDataPostprocessingProcessor(db, pdp, base_params)
        #client = ensure_dask_client(max_workers=2, processes=True)
        for c in range(parameters["rspace_info"]["num_chunks"]):
            post.process_chunk(c, saver, client, output_dir=out_dir)
    db.close()
    shutdown_dask()
    log.info("âœ“ %d-D workflow finished", dim)
if __name__ == "__main__":
    freeze_support()          # makes .exe builds happy
    main()
