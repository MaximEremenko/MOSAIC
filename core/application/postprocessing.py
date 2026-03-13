from __future__ import annotations

from core.application.common import normalize_post_mode
from core.application.coefficients import to_numpy
from core.domain.models import PostprocessingContext, StructureData, WorkflowParameters
from core.processors.point_data_postprocessing_processor import (
    PointDataPostprocessingProcessor,
)


class PostprocessingService:
    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
    ) -> None:
        context = self._build_context(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
        )
        if not bool(context.workflow_parameters.rspace_info.get("run_postprocessing", True)):
            return

        postprocessing_parameters = self._build_adapter_payload(context)
        processor = PointDataPostprocessingProcessor(
            context.artifacts.db_manager,
            context.artifacts.point_data_processor,
            postprocessing_parameters,
        )
        for chunk_id in sorted(context.artifacts.db_manager.get_pending_chunk_ids()):
            processor.process_chunk(
                int(chunk_id),
                context.artifacts.saver,
                client,
                output_dir=context.artifacts.output_dir,
            )

    def _build_context(
        self,
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

    def _build_adapter_payload(self, context: PostprocessingContext) -> dict[str, object]:
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
