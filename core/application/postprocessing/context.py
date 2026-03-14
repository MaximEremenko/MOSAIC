from __future__ import annotations

from core.application.postprocessing.mode import normalize_post_mode
from core.domain.models import PostprocessingContext, StructureData, WorkflowParameters


def build_postprocessing_context(
    *,
    workflow_parameters: WorkflowParameters,
    structure: StructureData,
    artifacts,
) -> PostprocessingContext:
    rspace = workflow_parameters.rspace_info
    post_mode = normalize_post_mode(
        rspace.get("mode")
        or rspace.get("postprocess_mode")
        or rspace.get("postprocessing_mode")
        or "displacement"
    )
    return PostprocessingContext(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        postprocessing_mode=post_mode,
    )
