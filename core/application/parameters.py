from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.application.common import as_bool, first_present, resolve_path_from, unique_paths
from core.domain.models import (
    FormFactorSelection,
    RunSettings,
    RuntimeSettings,
    WorkflowParameters,
)


class ParameterLoadingService:
    def load(self, run_file: str = "run_parameters.json") -> tuple[RunSettings, WorkflowParameters]:
        run_path, run_settings_payload = self._load_run_settings(run_file)
        input_parameters_path, root_path = self._resolve_input_parameters_path(
            run_path, run_settings_payload
        )
        with input_parameters_path.open("r", encoding="utf-8") as handle:
            raw_parameters = json.load(handle)

        runtime = self._resolve_runtime_settings(raw_parameters)
        config_root = self._resolve_config_root(
            raw_parameters,
            input_parameters_path,
            root_path=root_path,
        )
        normalized = self._normalize_input_schema(raw_parameters)
        normalized = self._normalize_parameter_paths(
            normalized, config_root, input_parameters_path
        )
        run_settings = RunSettings(
            run_parameters_path=run_path,
            input_parameters_path=input_parameters_path.resolve(),
            config_root=config_root.resolve(),
            working_path=config_root.resolve(),
            runtime=runtime,
            root_path=root_path.resolve() if root_path is not None else None,
        )
        return run_settings, WorkflowParameters.from_payload(normalized)

    def apply_runtime_settings(self, runtime_settings: RuntimeSettings) -> None:
        os.environ["DASK_WORKER_DASHBOARD"] = "1" if runtime_settings.worker_dashboard else "0"
        os.environ["DASK_BACKEND"] = runtime_settings.backend
        os.environ["DASK_MAX_WORKERS"] = str(runtime_settings.max_workers)
        os.environ["DASK_THREADS_PER_WORKER"] = str(runtime_settings.threads_per_worker)
        os.environ["DASK_PROCESSES"] = "1" if runtime_settings.processes else "0"

    def resolve_form_factor_settings(
        self, workflow_parameters: WorkflowParameters
    ) -> FormFactorSelection:
        runtime = workflow_parameters.runtime_info or {}
        form_factor = first_present(runtime, ("form_factor", "formFactor")) or {}
        family = first_present(
            form_factor,
            ("family", "factory", "type", "form_factor_family"),
        ) or "neutron"
        calculator = first_present(
            form_factor,
            ("calculator", "method", "name", "form_factor_calculator"),
        ) or "default"
        return FormFactorSelection(
            family=str(family).strip().lower(),
            calculator=str(calculator).strip(),
        )

    def _load_run_settings(self, run_file: str) -> tuple[Path, dict[str, Any]]:
        run_path = Path(run_file)
        if run_path.is_absolute():
            candidates = [run_path]
        else:
            script_dir = Path(__file__).resolve().parents[1]
            candidates = [script_dir / run_path, Path.cwd() / run_path]

        resolved_candidates = unique_paths(candidates)
        for candidate in resolved_candidates:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    return candidate, json.load(handle) or {}
        return resolved_candidates[0], {}

    def _resolve_input_parameters_path(
        self, run_path: Path, run_settings_payload: dict[str, Any]
    ) -> tuple[Path, Path | None]:
        raw_input = first_present(
            run_settings_payload,
            (
                "input_parameters_path",
                "input_parameters",
                "input_parameters_json",
                "input_json",
            ),
        )
        if raw_input is not None:
            candidate = resolve_path_from(run_path.parent, raw_input)
            if candidate.is_dir():
                candidate = candidate / "input_parameters.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured input parameters file '{candidate}' does not exist."
                )
            return candidate, None

        root_path = None
        raw_root = first_present(
            run_settings_payload, ("working_path", "config_dir", "config_path")
        )
        if raw_root is not None:
            candidate = resolve_path_from(run_path.parent, raw_root)
            root_path = candidate if candidate.is_dir() else candidate.parent
            if candidate.is_dir():
                candidate = candidate / "input_parameters.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Configured working path '{candidate}' does not contain input_parameters.json."
                )
            return candidate, root_path

        search_roots = unique_paths(
            [
                run_path.parent / "examples",
                run_path.parent.parent / "examples",
                run_path.parent / "tests",
                run_path.parent.parent / "tests",
                Path.cwd() / "examples",
                Path.cwd().parent / "examples",
                Path.cwd() / "tests",
                Path.cwd().parent / "tests",
            ]
        )
        found: list[Path] = []
        for root in search_roots:
            if root.exists():
                found.extend(root.rglob("input_parameters.json"))
        found = unique_paths(found)
        if len(found) == 1:
            return found[0], found[0].parent
        if not found:
            raise FileNotFoundError(
                "No input_parameters.json found. Set 'input_parameters_path' in run_parameters.json."
            )

        found_msg = "\n".join(f" - {path}" for path in found)
        raise FileNotFoundError(
            "Multiple input_parameters.json files found. Set 'input_parameters_path' in run_parameters.json.\n"
            f"{found_msg}"
        )

    def _resolve_config_root(
        self,
        parameters: dict[str, Any],
        input_parameters_path: Path,
        root_path: Path | None = None,
    ) -> Path:
        json_dir = input_parameters_path.parent.resolve()
        paths = parameters.get("paths", {}) if isinstance(parameters, dict) else {}
        struct = parameters.get("structInfo", {}) if isinstance(parameters, dict) else {}

        explicit_root = first_present(
            paths,
            (
                "config_root",
                "configRoot",
                "base_directory",
                "baseDirectory",
                "data_root",
                "dataRoot",
            ),
        ) or first_present(
            struct,
            (
                "config_root",
                "configRoot",
                "base_directory",
                "baseDirectory",
                "data_root",
                "dataRoot",
            ),
        ) or first_present(
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

        candidates: list[Path] = []
        if explicit_root is not None:
            candidates.append(resolve_path_from(json_dir, explicit_root))
        if root_path is not None:
            candidates.append(root_path.resolve())
        candidates.extend((json_dir, json_dir.parent, Path.cwd().resolve()))
        candidates = unique_paths(candidates)

        cfg_name = first_present(
            paths,
            (
                "structure_file",
                "structureFile",
                "average_structure_file",
                "averageStructureFile",
            ),
        ) or first_present(struct, ("filename", "filename_av"))
        if isinstance(cfg_name, str) and cfg_name.strip():
            cfg_path = Path(cfg_name)
            if cfg_path.is_absolute():
                return cfg_path.parent.resolve()
            for base in candidates:
                if (base / cfg_path).exists():
                    return base

        if explicit_root is not None:
            return candidates[0]
        if root_path is not None:
            return root_path.resolve()
        return json_dir

    def _normalize_special_points(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for point in points or []:
            if not isinstance(point, dict):
                continue
            entry = {
                "coordinate": first_present(point, ("coordinate",)),
                "radius": first_present(point, ("radius",)),
            }
            symmetry = first_present(
                point, ("space_group_symmetry", "spaceGroupSymmetry")
            )
            if symmetry is not None:
                entry["spaceGroupSymmetry"] = symmetry
            shape = first_present(point, ("shape",))
            if shape is not None:
                entry["shape"] = shape
            normalized.append(entry)
        return normalized

    def _expand_processing_points(
        self, points: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        expanded = []
        for point in points or []:
            if not isinstance(point, dict):
                continue
            selector = (
                point.get("selector") if isinstance(point.get("selector"), dict) else point
            )
            window = point.get("window") if isinstance(point.get("window"), dict) else {}

            dist = first_present(window, ("dist_from_atom_center", "distFromAtomCenter"))
            if dist is None:
                dist = first_present(point, ("dist_from_atom_center", "distFromAtomCenter"))

            step = first_present(window, ("step_in_angstrom", "stepInAngstrom"))
            if step is None:
                step = first_present(point, ("step_in_angstrom", "stepInAngstrom"))

            file_name = first_present(selector, ("file", "filename"))
            if file_name is not None:
                expanded.append(
                    {
                        "filename": file_name,
                        "distFromAtomCenter": dist,
                        "stepInAngstrom": step,
                    }
                )
                continue

            element = first_present(
                selector, ("element", "element_symbol", "elementSymbol")
            )
            reference_numbers = first_present(
                selector, ("reference_numbers", "referenceNumbers")
            )
            if reference_numbers is None:
                single_ref = first_present(
                    selector, ("reference_number", "referenceNumber")
                )
                if single_ref is not None:
                    reference_numbers = [single_ref]

            if reference_numbers is None:
                reference_numbers = []
            elif not isinstance(reference_numbers, (list, tuple)):
                reference_numbers = [reference_numbers]

            for reference_number in reference_numbers:
                expanded.append(
                    {
                        "elementSymbol": element,
                        "referenceNumber": reference_number,
                        "distFromAtomCenter": dist,
                        "stepInAngstrom": step,
                    }
                )
        return expanded

    def _normalize_input_schema(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, dict):
            raise ValueError("Input parameters must be a JSON object.")

        is_unified = any(
            key in parameters for key in ("paths", "structure", "reciprocal_space", "processing")
        )
        if not is_unified:
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
        coefficients = (
            structure.get("coefficients")
            if isinstance(structure.get("coefficients"), dict)
            else {}
        )
        processing_coeff = (
            processing.get("coefficients")
            if isinstance(processing.get("coefficients"), dict)
            else {}
        )
        cell_limits = (
            structure.get("cell_limits")
            if isinstance(structure.get("cell_limits"), dict)
            else {}
        )
        mask = reciprocal_space.get("mask") if isinstance(reciprocal_space.get("mask"), dict) else {}

        struct_info = {
            "dimension": int(first_present(structure, ("dimension",))),
            "working_directory": first_present(
                paths, ("output_directory", "working_directory", "outputDirectory")
            ),
            "filename": first_present(paths, ("structure_file", "filename", "structureFile")),
        }
        config_root = first_present(paths, ("config_root", "configRoot"))
        if config_root is not None:
            struct_info["config_root"] = config_root
        average_structure_file = first_present(
            paths,
            ("average_structure_file", "averageStructureFile", "filename_av"),
        )
        if average_structure_file is not None:
            struct_info["filename_av"] = average_structure_file

        coeff_source = first_present(coefficients, ("source", "coeff_source", "coeffSource"))
        if coeff_source is not None:
            struct_info["coeff_source"] = coeff_source
        coeff_file = first_present(
            coefficients, ("file", "path", "filename", "coeff_file", "coeff_path")
        )
        if coeff_file is not None:
            struct_info["coeff_file"] = coeff_file

        cells_limits_min = first_present(cell_limits, ("min", "cells_limits_min"))
        cells_limits_max = first_present(cell_limits, ("max", "cells_limits_max"))
        if cells_limits_min is not None:
            struct_info["cells_limits_min"] = list(cells_limits_min)
        if cells_limits_max is not None:
            struct_info["cells_limits_max"] = list(cells_limits_max)

        peak_info = {
            "reciprocal_space_limits": reciprocal_space.get("intervals")
            or reciprocal_space.get("reciprocal_space_limits")
            or [],
        }
        mask_equation = first_present(mask, ("equation", "mask_equation", "condition"))
        if mask_equation is not None:
            peak_info["mask_equation"] = mask_equation
        special_points = mask.get("special_points") or mask.get("specialPoints")
        if special_points:
            peak_info["specialPoints"] = self._normalize_special_points(special_points)
        shell_radii = mask.get("shell_radii") if isinstance(mask.get("shell_radii"), dict) else {}
        if not shell_radii:
            shell_radii = mask.get("shellRadii") if isinstance(mask.get("shellRadii"), dict) else {}
        r1 = first_present(shell_radii, ("r1", "inner", "inner_radius", "innerRadius"))
        r2 = first_present(shell_radii, ("r2", "outer", "outer_radius", "outerRadius"))
        if r1 is not None:
            peak_info["r1"] = r1
        if r2 is not None:
            peak_info["r2"] = r2

        rspace_info = {
            "num_chunks": int(first_present(processing, ("num_chunks",)) or 1),
            "fresh_start": as_bool(
                first_present(processing, ("fresh_start", "freshStart")),
                default=False,
            ),
            "method": first_present(processing, ("method",)) or "from_average",
            "filter_type": first_present(processing, ("filter_type", "filterType"))
            or "Chebyshev",
            "save_rifft_coordinates": as_bool(
                first_present(
                    processing, ("save_rifft_coordinates", "saveRifftCoordinates")
                ),
                default=False,
            ),
            "smooth_intensities": as_bool(
                first_present(processing, ("smooth_intensities", "smoothIntensities")),
                default=True,
            ),
            "print_intensities": as_bool(
                first_present(processing, ("print_intensities", "printIntensities")),
                default=False,
            ),
            "rspace_parallel_processing": as_bool(
                first_present(
                    processing,
                    (
                        "parallel_processing",
                        "rspace_parallel_processing",
                        "parallelProcessing",
                    ),
                ),
                default=False,
            ),
            "points": self._expand_processing_points(processing.get("points", [])),
            "mode": first_present(
                processing, ("mode", "postprocessing_mode", "postprocess_mode")
            )
            or "displacement",
            "run_postprocessing": as_bool(
                first_present(processing, ("run_postprocessing", "run", "runPostprocessing")),
                default=True,
            ),
        }
        if cells_limits_min is not None:
            rspace_info["cells_limits_min"] = list(cells_limits_min)
        if cells_limits_max is not None:
            rspace_info["cells_limits_max"] = list(cells_limits_max)

        chemical_filtered = first_present(
            processing,
            ("chemical_filtered_ordering", "chemical_filtered", "chemicalFilteredOrdering"),
        )
        if chemical_filtered is not None:
            rspace_info["chemical_filtered_ordering"] = as_bool(chemical_filtered)

        coeff_center_by = first_present(
            processing_coeff, ("center_by", "coeff_center_by", "coeffCenterBy")
        ) or first_present(
            processing,
            ("coeff_center_by", "coeff_center_mode", "chemical_coeff_center_by"),
        ) or first_present(coefficients, ("center_by", "coeff_center_by", "coeffCenterBy"))
        if coeff_center_by is not None:
            rspace_info["coeff_center_by"] = coeff_center_by

        use_coeff = first_present(processing_coeff, ("use", "use_coeff", "useCoeff"))
        if use_coeff is not None:
            rspace_info["use_coeff"] = as_bool(use_coeff)

        return {
            "schema_version": parameters.get("schema_version", 2),
            "structInfo": struct_info,
            "peakInfo": peak_info,
            "rspace_info": rspace_info,
            "runtime_info": runtime,
        }

    def _normalize_parameter_paths(
        self,
        parameters: dict[str, Any],
        config_root: Path,
        input_parameters_path: Path,
    ) -> dict[str, Any]:
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
                struct[key] = str(resolve_path_from(config_root, raw))

        for point in parameters.get("rspace_info", {}).get("points", []):
            raw = point.get("filename")
            if isinstance(raw, str) and raw.strip():
                point["filename"] = str(resolve_path_from(config_root, raw))

        parameters["input_parameters_path"] = str(input_parameters_path.resolve())
        parameters["config_root"] = str(config_root.resolve())
        return parameters

    def _resolve_runtime_settings(
        self, parameters: dict[str, Any]
    ) -> RuntimeSettings:
        runtime = first_present(parameters, ("runtime", "runtime_info", "runtimeInfo")) or {}
        dask = first_present(runtime, ("dask", "dask_info", "daskInfo")) or {}
        return RuntimeSettings(
            worker_dashboard=as_bool(
                first_present(dask, ("worker_dashboard", "dashboard", "dask_worker_dashboard")),
                default=False,
            ),
            backend=str(first_present(dask, ("backend", "dask_backend")) or "local"),
            max_workers=int(first_present(dask, ("max_workers", "dask_max_workers")) or 2),
            threads_per_worker=int(
                first_present(dask, ("threads_per_worker", "dask_threads_per_worker"))
                or 16
            ),
            processes=as_bool(
                first_present(dask, ("processes", "dask_processes")),
                default=False,
            ),
            wait_timeout=str(
                first_present(
                    dask, ("worker_wait_timeout", "wait_timeout", "dask_wait_timeout")
                )
                or "120s"
            ),
        )
