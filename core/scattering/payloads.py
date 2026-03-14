from __future__ import annotations

import numpy as np

from core.scattering.coefficients import to_numpy
from core.models import AmplitudeExecutionContext


def build_base_amplitude_parameters(
    context: AmplitudeExecutionContext,
) -> dict[str, object]:
    return {
        "reciprocal_space_intervals": context.intervals,
        "reciprocal_space_intervals_all": context.artifacts.padded_intervals,
        "point_data_list": [
            {
                "central_point_id": row["central_point_id"],
                "coordinates": row["coordinates"],
                "dist_from_atom_center": row["dist_from_atom_center"],
                "step_in_frac": row["step_in_frac"],
                "chunk_id": row["chunk_id"],
                "grid_amplitude_initialized": row["grid_amplitude_initialized"],
                "id": row["central_point_id"],
            }
            for row in context.point_rows
        ],
        "original_coords": to_numpy(context.structure.original_coords),
        "average_coords": to_numpy(context.structure.average_coords),
        "cells_origin": to_numpy(context.structure.cells_origin),
        "elements": to_numpy(context.structure.elements),
        "refnumbers": to_numpy(context.structure.refnumbers),
        "rspace_info": context.workflow_parameters.rspace_info,
        "vectors": context.structure.vectors,
        "supercell": context.structure.supercell,
        "postprocessing_mode": context.postprocessing_mode,
    }


def build_amplitude_adapter_payload(
    context: AmplitudeExecutionContext,
    base_params: dict[str, object],
) -> dict[str, object]:
    amplitude_parameters = dict(base_params)
    if context.use_coeff:
        amplitude_parameters["coeff"] = np.asarray(
            context.centered_coefficients, float
        )
    if context.chemical_filtered:
        amplitude_parameters["original_coords"] = to_numpy(
            context.structure.cells_origin
        )
    return amplitude_parameters
