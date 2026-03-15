from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config.values import (
    as_bool,
    first_present,
    resolve_path_from,
)


def normalize_special_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        entry = {
            "coordinate": first_present(point, ("coordinate",)),
            "radius": first_present(point, ("radius",)),
        }
        symmetry = first_present(point, ("space_group_symmetry", "spaceGroupSymmetry"))
        if symmetry is not None:
            entry["spaceGroupSymmetry"] = symmetry
        shape = first_present(point, ("shape",))
        if shape is not None:
            entry["shape"] = shape
        normalized.append(entry)
    return normalized


def expand_processing_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        selector = point.get("selector") if isinstance(point.get("selector"), dict) else point
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

        element = first_present(selector, ("element", "element_symbol", "elementSymbol"))
        reference_numbers = first_present(selector, ("reference_numbers", "referenceNumbers"))
        if reference_numbers is None:
            single_ref = first_present(selector, ("reference_number", "referenceNumber"))
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


def normalize_input_schema(parameters: dict[str, Any]) -> dict[str, Any]:
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
    coefficients = structure.get("coefficients") if isinstance(structure.get("coefficients"), dict) else {}
    processing_coeff = processing.get("coefficients") if isinstance(processing.get("coefficients"), dict) else {}
    decoder = processing.get("decoder") if isinstance(processing.get("decoder"), dict) else {}
    cell_limits = structure.get("cell_limits") if isinstance(structure.get("cell_limits"), dict) else {}
    mask = reciprocal_space.get("mask") if isinstance(reciprocal_space.get("mask"), dict) else {}

    struct_info = {
        "dimension": int(first_present(structure, ("dimension",))),
        "working_directory": first_present(paths, ("output_directory", "working_directory", "outputDirectory")),
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
    coeff_file = first_present(coefficients, ("file", "path", "filename", "coeff_file", "coeff_path"))
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
        peak_info["specialPoints"] = normalize_special_points(special_points)
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
        "fresh_start": as_bool(first_present(processing, ("fresh_start", "freshStart")), default=False),
        "method": first_present(processing, ("method",)) or "from_average",
        "filter_type": first_present(processing, ("filter_type", "filterType")) or "Chebyshev",
        "save_rifft_coordinates": as_bool(
            first_present(processing, ("save_rifft_coordinates", "saveRifftCoordinates")),
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
                ("parallel_processing", "rspace_parallel_processing", "parallelProcessing"),
            ),
            default=False,
        ),
        "points": expand_processing_points(processing.get("points", [])),
        "mode": first_present(processing, ("mode", "postprocessing_mode", "postprocess_mode"))
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

    decoder_source = (
        first_present(decoder, ("source", "mode"))
        or first_present(processing, ("decoder_source", "decoderSource"))
        or "error"
    )
    decoder_cache_path = (
        first_present(decoder, ("cache_path", "path", "cachePath"))
        or first_present(processing, ("decoder_cache_path", "decoderCachePath"))
    )
    decoder_compute_output_directory = (
        first_present(
            decoder,
            ("compute_output_directory", "output_directory", "computeOutputDirectory"),
        )
        or first_present(
            processing,
            ("decoder_compute_output_directory", "decoderComputeOutputDirectory"),
        )
    )
    rspace_info["decoder"] = {
        "source": decoder_source,
    }
    if decoder_cache_path is not None:
        rspace_info["decoder"]["cache_path"] = decoder_cache_path
    if decoder_compute_output_directory is not None:
        rspace_info["decoder"]["compute_output_directory"] = decoder_compute_output_directory

    return {
        "schema_version": parameters.get("schema_version", 2),
        "structInfo": struct_info,
        "peakInfo": peak_info,
        "rspace_info": rspace_info,
        "runtime_info": runtime,
    }


def normalize_parameter_paths(
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

    decoder = parameters.get("rspace_info", {}).get("decoder")
    if isinstance(decoder, dict):
        for key in ("cache_path", "compute_output_directory"):
            raw = decoder.get(key)
            if isinstance(raw, str) and raw.strip():
                decoder[key] = str(resolve_path_from(config_root, raw))

    parameters["input_parameters_path"] = str(input_parameters_path.resolve())
    parameters["config_root"] = str(config_root.resolve())
    return parameters
