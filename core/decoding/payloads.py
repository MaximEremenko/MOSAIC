from __future__ import annotations

from core.scattering.coefficients import to_numpy
from core.decoding.context import DecodingContext


def build_decoding_payload(context: DecodingContext) -> dict[str, object]:
    rspace_info = context.workflow_parameters.rspace_info.to_mapping()
    return {
        "reciprocal_space_intervals_all": context.artifacts.padded_intervals,
        "original_coords": to_numpy(context.structure.original_coords),
        "average_coords": to_numpy(context.structure.average_coords),
        "cells_origin": to_numpy(context.structure.cells_origin),
        "elements": to_numpy(context.structure.elements),
        "refnumbers": to_numpy(context.structure.refnumbers),
        "rspace_info": rspace_info,
        "decoder": rspace_info.get("decoder"),
        # The run's structure identity (stamped at load). Passed explicitly
        # rather than by handing the decoder the whole runtime_info, which
        # would also expose residual_run_digest and switch the source-identity
        # resolution onto its manifest branch — a change this does not intend.
        "source_structure_digest": context.workflow_parameters.runtime_info.get(
            "source_structure_digest"
        ),
        "vectors": context.structure.vectors,
        "supercell": context.structure.supercell,
        "postprocessing_mode": context.postprocessing_mode,
    }
