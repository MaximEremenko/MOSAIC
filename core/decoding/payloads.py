from __future__ import annotations

from core.scattering.coefficients import to_numpy
from core.models import PostprocessingContext


def build_postprocessing_payload(context: PostprocessingContext) -> dict[str, object]:
    return {
        "reciprocal_space_intervals_all": context.artifacts.padded_intervals,
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
