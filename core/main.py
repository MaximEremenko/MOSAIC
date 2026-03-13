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
    # Runtime settings are loaded from input_parameters.json later in startup.
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
    def _resolve_path_from(base_dir, raw_path):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        return path

    def _unique_paths(paths):
        unique = []
        seen = set()
        for path in paths:
            if path is None:
                continue
            path = Path(path).resolve()
            key = str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _load_run_settings(run_file="run_parameters.json"):
        run_path = Path(run_file)
        if run_path.is_absolute():
            candidates = [run_path]
        else:
            script_dir = Path(__file__).resolve().parent
            candidates = [script_dir / run_path, Path.cwd() / run_path]

        resolved_candidates = _unique_paths(candidates)
        for candidate in resolved_candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as fh:
                    return candidate, json.load(fh) or {}

        return resolved_candidates[0], {}

    def _resolve_input_parameters_path(run_file="run_parameters.json"):
        run_path, run_settings = _load_run_settings(run_file)

        raw_input = _first_present(
            run_settings,
            (
                "input_parameters_path",
                "input_parameters",
                "input_parameters_json",
                "input_json",
            ),
        )
        if raw_input is not None:
            candidate = _resolve_path_from(run_path.parent, raw_input)
            if candidate.is_dir():
                candidate = candidate / "input_parameters.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured input parameters file '{candidate}' does not exist."
                )
            return candidate, run_settings, None

        legacy_root = None
        raw_working = _first_present(run_settings, ("working_path", "config_dir", "config_path"))
        if raw_working is not None:
            candidate = _resolve_path_from(run_path.parent, raw_working)
            legacy_root = candidate if candidate.is_dir() else candidate.parent
            if candidate.is_dir():
                candidate = candidate / "input_parameters.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured working path '{candidate}' does not contain input_parameters.json."
                )
            return candidate, run_settings, legacy_root

        search_roots = _unique_paths(
            [
                run_path.parent / "tests",
                run_path.parent.parent / "tests",
                Path.cwd() / "tests",
                Path.cwd().parent / "tests",
            ]
        )
        found = []
        for root in search_roots:
            if root.exists():
                found.extend(root.rglob("input_parameters.json"))
        found = _unique_paths(found)
        if len(found) == 1:
            return found[0], run_settings, found[0].parent
        if not found:
            raise FileNotFoundError(
                "No input_parameters.json found. Set 'input_parameters_path' in run_parameters.json."
            )

        found_msg = "\n".join(f" - {p}" for p in found)
        raise FileNotFoundError(
            "Multiple input_parameters.json files found. Set 'input_parameters_path' in run_parameters.json.\n"
            f"{found_msg}"
        )

    def _resolve_config_root(parameters, input_parameters_path, legacy_root=None):
        json_dir = Path(input_parameters_path).parent.resolve()
        paths = parameters.get("paths", {}) if isinstance(parameters, dict) else {}
        struct = parameters.get("structInfo", {}) if isinstance(parameters, dict) else {}

        explicit_root = _first_present(
            paths,
            (
                "config_root",
                "configRoot",
                "base_directory",
                "baseDirectory",
                "data_root",
                "dataRoot",
            ),
        ) or _first_present(
            struct,
            (
                "config_root",
                "configRoot",
                "base_directory",
                "baseDirectory",
                "data_root",
                "dataRoot",
            ),
        ) or _first_present(
            parameters,
            (
                "config_root",
                "configRoot",
                "base_directory",
                "baseDirectory",
                "data_root",
                "dataRoot",
            ),
        )

        candidates = []
        if explicit_root is not None:
            candidates.append(_resolve_path_from(json_dir, explicit_root))
        if legacy_root is not None:
            candidates.append(Path(legacy_root).resolve())
        candidates.extend((json_dir, json_dir.parent, Path.cwd().resolve()))
        candidates = _unique_paths(candidates)

        cfg_name = _first_present(
            paths,
            ("structure_file", "structureFile", "average_structure_file", "averageStructureFile"),
        ) or _first_present(struct, ("filename", "filename_av"))
        if isinstance(cfg_name, str) and cfg_name.strip():
            cfg_path = Path(cfg_name)
            if cfg_path.is_absolute():
                return cfg_path.parent.resolve()
            for base in candidates:
                if (base / cfg_path).exists():
                    return base

        if explicit_root is not None:
            return candidates[0]
        if legacy_root is not None:
            return Path(legacy_root).resolve()
        return json_dir

    def _normalize_special_points(points):
        normalized = []
        for sp in points or []:
            if not isinstance(sp, dict):
                continue
            entry = {
                "coordinate": _first_present(sp, ("coordinate",)),
                "radius": _first_present(sp, ("radius",)),
            }
            symmetry = _first_present(sp, ("space_group_symmetry", "spaceGroupSymmetry"))
            if symmetry is not None:
                entry["spaceGroupSymmetry"] = symmetry
            shape = _first_present(sp, ("shape",))
            if shape is not None:
                entry["shape"] = shape
            normalized.append(entry)
        return normalized

    def _expand_processing_points(points):
        expanded = []
        for point in points or []:
            if not isinstance(point, dict):
                continue
            selector = point.get("selector") if isinstance(point.get("selector"), dict) else point
            window = point.get("window") if isinstance(point.get("window"), dict) else {}

            dist = _first_present(
                window,
                ("dist_from_atom_center", "distFromAtomCenter"),
            )
            if dist is None:
                dist = _first_present(point, ("dist_from_atom_center", "distFromAtomCenter"))

            step = _first_present(
                window,
                ("step_in_angstrom", "stepInAngstrom"),
            )
            if step is None:
                step = _first_present(point, ("step_in_angstrom", "stepInAngstrom"))

            file_name = _first_present(selector, ("file", "filename"))
            if file_name is not None:
                expanded.append(
                    {
                        "filename": file_name,
                        "distFromAtomCenter": dist,
                        "stepInAngstrom": step,
                    }
                )
                continue

            element = _first_present(selector, ("element", "element_symbol", "elementSymbol"))
            reference_numbers = _first_present(
                selector,
                ("reference_numbers", "referenceNumbers"),
            )
            if reference_numbers is None:
                single_ref = _first_present(selector, ("reference_number", "referenceNumber"))
                if single_ref is not None:
                    reference_numbers = [single_ref]

            if reference_numbers is None:
                reference_numbers = []
            elif not isinstance(reference_numbers, (list, tuple)):
                reference_numbers = [reference_numbers]

            for ref in reference_numbers:
                expanded.append(
                    {
                        "elementSymbol": element,
                        "referenceNumber": ref,
                        "distFromAtomCenter": dist,
                        "stepInAngstrom": step,
                    }
                )
        return expanded

    def _normalize_input_schema(parameters):
        if not isinstance(parameters, dict):
            raise ValueError("Input parameters must be a JSON object.")

        is_universal = any(k in parameters for k in ("paths", "structure", "reciprocal_space", "processing"))
        if not is_universal:
            struct = parameters.setdefault("structInfo", {})
            rspace = parameters.setdefault("rspace_info", {})
            if "cells_limits_min" in struct and "cells_limits_min" not in rspace:
                rspace["cells_limits_min"] = struct["cells_limits_min"]
            if "cells_limits_max" in struct and "cells_limits_max" not in rspace:
                rspace["cells_limits_max"] = struct["cells_limits_max"]
            return parameters

        paths = parameters.get("paths") or {}
        structure = parameters.get("structure") or {}
        reciprocal_space = parameters.get("reciprocal_space") or {}
        processing = parameters.get("processing") or {}
        runtime = parameters.get("runtime") or parameters.get("runtime_info") or {}
        coefficients = structure.get("coefficients") if isinstance(structure.get("coefficients"), dict) else {}
        processing_coeff = processing.get("coefficients") if isinstance(processing.get("coefficients"), dict) else {}
        cell_limits = structure.get("cell_limits") if isinstance(structure.get("cell_limits"), dict) else {}
        mask = reciprocal_space.get("mask") if isinstance(reciprocal_space.get("mask"), dict) else {}

        struct_info = {
            "dimension": int(_first_present(structure, ("dimension",))),
            "working_directory": _first_present(paths, ("output_directory", "working_directory", "outputDirectory")),
            "filename": _first_present(paths, ("structure_file", "filename", "structureFile")),
        }
        config_root = _first_present(paths, ("config_root", "configRoot"))
        if config_root is not None:
            struct_info["config_root"] = config_root
        average_structure_file = _first_present(
            paths,
            ("average_structure_file", "averageStructureFile", "filename_av"),
        )
        if average_structure_file is not None:
            struct_info["filename_av"] = average_structure_file

        coeff_source = _first_present(coefficients, ("source", "coeff_source", "coeffSource"))
        if coeff_source is not None:
            struct_info["coeff_source"] = coeff_source
        coeff_file = _first_present(coefficients, ("file", "path", "filename", "coeff_file", "coeff_path"))
        if coeff_file is not None:
            struct_info["coeff_file"] = coeff_file

        cells_limits_min = _first_present(cell_limits, ("min", "cells_limits_min"))
        cells_limits_max = _first_present(cell_limits, ("max", "cells_limits_max"))
        if cells_limits_min is not None:
            struct_info["cells_limits_min"] = list(cells_limits_min)
        if cells_limits_max is not None:
            struct_info["cells_limits_max"] = list(cells_limits_max)

        peak_info = {
            "reciprocal_space_limits": reciprocal_space.get("intervals")
            or reciprocal_space.get("reciprocal_space_limits")
            or [],
        }
        mask_equation = _first_present(mask, ("equation", "mask_equation", "condition"))
        if mask_equation is not None:
            peak_info["mask_equation"] = mask_equation
        special_points = mask.get("special_points") or mask.get("specialPoints")
        if special_points:
            peak_info["specialPoints"] = _normalize_special_points(special_points)
        shell_radii = mask.get("shell_radii") if isinstance(mask.get("shell_radii"), dict) else {}
        if not shell_radii:
            shell_radii = mask.get("shellRadii") if isinstance(mask.get("shellRadii"), dict) else {}
        r1 = _first_present(shell_radii, ("r1", "inner", "inner_radius", "innerRadius"))
        r2 = _first_present(shell_radii, ("r2", "outer", "outer_radius", "outerRadius"))
        if r1 is not None:
            peak_info["r1"] = r1
        if r2 is not None:
            peak_info["r2"] = r2

        rspace_info = {
            "num_chunks": int(_first_present(processing, ("num_chunks",)) or 1),
            "fresh_start": _as_bool(_first_present(processing, ("fresh_start", "freshStart")), default=False),
            "method": _first_present(processing, ("method",)) or "from_average",
            "filter_type": _first_present(processing, ("filter_type", "filterType")) or "Chebyshev",
            "save_rifft_coordinates": _as_bool(
                _first_present(processing, ("save_rifft_coordinates", "saveRifftCoordinates")),
                default=False,
            ),
            "smooth_intensities": _as_bool(
                _first_present(processing, ("smooth_intensities", "smoothIntensities")),
                default=True,
            ),
            "print_intensities": _as_bool(
                _first_present(processing, ("print_intensities", "printIntensities")),
                default=False,
            ),
            "rspace_parallel_processing": _as_bool(
                _first_present(processing, ("parallel_processing", "rspace_parallel_processing", "parallelProcessing")),
                default=False,
            ),
            "points": _expand_processing_points(processing.get("points", [])),
            "mode": _first_present(processing, ("mode", "postprocessing_mode", "postprocess_mode")) or "displacement",
            "run_postprocessing": _as_bool(
                _first_present(processing, ("run_postprocessing", "run", "runPostprocessing")),
                default=True,
            ),
        }
        if cells_limits_min is not None:
            rspace_info["cells_limits_min"] = list(cells_limits_min)
        if cells_limits_max is not None:
            rspace_info["cells_limits_max"] = list(cells_limits_max)

        chemical_filtered = _first_present(
            processing,
            ("chemical_filtered_ordering", "chemical_filtered", "chemicalFilteredOrdering"),
        )
        if chemical_filtered is not None:
            rspace_info["chemical_filtered_ordering"] = _as_bool(chemical_filtered)

        coeff_center_by = _first_present(
            processing_coeff,
            ("center_by", "coeff_center_by", "coeffCenterBy"),
        ) or _first_present(
            processing,
            ("coeff_center_by", "coeff_center_mode", "chemical_coeff_center_by"),
        ) or _first_present(
            coefficients,
            ("center_by", "coeff_center_by", "coeffCenterBy"),
        )
        if coeff_center_by is not None:
            rspace_info["coeff_center_by"] = coeff_center_by

        use_coeff = _first_present(processing_coeff, ("use", "use_coeff", "useCoeff"))
        if use_coeff is not None:
            rspace_info["use_coeff"] = _as_bool(use_coeff)

        normalized = {
            "schema_version": parameters.get("schema_version", 2),
            "structInfo": struct_info,
            "peakInfo": peak_info,
            "rspace_info": rspace_info,
            "runtime_info": runtime,
        }
        return normalized

    def _normalize_parameter_paths(parameters, config_root, input_parameters_path):
        struct = parameters.setdefault("structInfo", {})
        for key in (
            "filename",
            "filename_av",
            "coeff_file",
            "coeff_filename",
            "coeff_path",
            "coefficients_file",
            "intensity_coeff_file",
            "intensity_coeff_filename",
            "working_directory",
        ):
            raw = struct.get(key)
            if isinstance(raw, str) and raw.strip():
                struct[key] = str(_resolve_path_from(config_root, raw))

        for point in parameters.get("rspace_info", {}).get("points", []):
            raw = point.get("filename")
            if isinstance(raw, str) and raw.strip():
                point["filename"] = str(_resolve_path_from(config_root, raw))

        parameters["input_parameters_path"] = str(Path(input_parameters_path).resolve())
        parameters["config_root"] = str(Path(config_root).resolve())
        return parameters

    def _as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        value_s = str(value).strip().lower()
        if value_s in ("1", "true", "yes", "on"):
            return True
        if value_s in ("0", "false", "no", "off"):
            return False
        return default

    def _resolve_runtime_settings(parameters):
        runtime = _first_present(parameters, ("runtime", "runtime_info", "runtimeInfo")) or {}
        dask = _first_present(runtime, ("dask", "dask_info", "daskInfo")) or {}
        return {
            "worker_dashboard": _as_bool(
                _first_present(dask, ("worker_dashboard", "dashboard", "dask_worker_dashboard")),
                default=False,
            ),
            "backend": str(_first_present(dask, ("backend", "dask_backend")) or "local"),
            "max_workers": int(_first_present(dask, ("max_workers", "dask_max_workers")) or 2),
            "threads_per_worker": int(
                _first_present(dask, ("threads_per_worker", "dask_threads_per_worker")) or 16
            ),
            "processes": _as_bool(
                _first_present(dask, ("processes", "dask_processes")),
                default=False,
            ),
            "wait_timeout": str(
                _first_present(dask, ("worker_wait_timeout", "wait_timeout", "dask_wait_timeout"))
                or "120s"
            ),
        }

    def _apply_runtime_settings(runtime_settings):
        os.environ["DASK_WORKER_DASHBOARD"] = "1" if runtime_settings["worker_dashboard"] else "0"
        os.environ["DASK_BACKEND"] = runtime_settings["backend"]
        os.environ["DASK_MAX_WORKERS"] = str(runtime_settings["max_workers"])
        os.environ["DASK_THREADS_PER_WORKER"] = str(runtime_settings["threads_per_worker"])
        os.environ["DASK_PROCESSES"] = "1" if runtime_settings["processes"] else "0"

    def _resolve_form_factor_settings(parameters):
        runtime = _first_present(parameters, ("runtime", "runtime_info", "runtimeInfo")) or {}
        form_factor = _first_present(runtime, ("form_factor", "formFactor")) or {}
        family = _first_present(form_factor, ("family", "factory", "type", "form_factor_family")) or "neutron"
        calculator = _first_present(
            form_factor,
            ("calculator", "method", "name", "form_factor_calculator"),
        ) or "default"
        return str(family).strip().lower(), str(calculator).strip()

    # --- 1. run-level paths ------------------------------------------------------
    input_parameters_path, _run_settings, legacy_root = _resolve_input_parameters_path()
    with Path(input_parameters_path).open("r", encoding="utf-8") as fh:
        raw_parameters = json.load(fh)

    runtime_settings = _resolve_runtime_settings(raw_parameters)
    _apply_runtime_settings(runtime_settings)
    setup_logging()
    log = logging.getLogger("app")
    client = get_client()
    if client is not None:
        client.wait_for_workers(runtime_settings["max_workers"], timeout=runtime_settings["wait_timeout"])

    config_root = _resolve_config_root(raw_parameters, input_parameters_path, legacy_root=legacy_root)
    working_path = str(config_root.resolve()) + os.sep
    log.info("Using input parameters: %s", input_parameters_path)
    log.info("Resolved configuration root: %s", config_root)
    log.info(
        "Runtime settings: backend=%s max_workers=%d threads_per_worker=%d processes=%s",
        runtime_settings["backend"],
        runtime_settings["max_workers"],
        runtime_settings["threads_per_worker"],
        runtime_settings["processes"],
    )

    # --- 2. parameters -----------------------------------------------------------
    p_json = str(Path(input_parameters_path).resolve())
    p_h5 = os.path.join(working_path, "parameters.hdf5")
    pfactory = ParametersProcessorFactoryProvider().get_factory()
    #pproc = (pfactory.create_processor(p_h5,  source_type="hdf5", hdf5_file_path=p_h5)
    #         if False #os.path.exists(p_h5)
    #         else
    pproc = pfactory.create_processor(p_json, source_type="file", hdf5_file_path=p_h5)
    pproc.process()
    parameters = _normalize_input_schema(pproc.get_parameters())
    parameters = _normalize_parameter_paths(parameters, config_root, input_parameters_path)

    dim = int(parameters["structInfo"]["dimension"])  # 1 / 2 / 3
    log.info("Running %d-D workflow", dim)

    # --- 3. structure ------------------------------------------------------------
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
        ff_family, ff_calculator = _resolve_form_factor_settings(parameters)
        ff_calc = FormFactorFactoryProducer.get_factory(ff_family).create_calculator(ff_calculator)
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
