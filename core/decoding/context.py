from __future__ import annotations

from dataclasses import dataclass

from core.processing_mode import normalize_processing_mode
from core.models import ReciprocalSpaceArtifacts, StructureData, WorkflowParameters


@dataclass(frozen=True)
class DecodingContext:
    workflow_parameters: WorkflowParameters
    structure: StructureData
    artifacts: ReciprocalSpaceArtifacts
    postprocessing_mode: str


def build_decoding_context(
    *,
    workflow_parameters: WorkflowParameters,
    structure: StructureData,
    artifacts,
) -> DecodingContext:
    post_mode = normalize_processing_mode(workflow_parameters.rspace_info.mode or "displacement")
    return DecodingContext(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        postprocessing_mode=post_mode,
    )


__all__ = ["DecodingContext", "build_decoding_context"]
