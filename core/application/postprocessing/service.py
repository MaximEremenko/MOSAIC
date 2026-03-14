from __future__ import annotations

from core.application.postprocessing.context import build_postprocessing_context
from core.application.postprocessing.payloads import build_postprocessing_payload
from core.application.postprocessing.processor import PointDataPostprocessingProcessor
from core.domain.models import StructureData, WorkflowParameters


class PostprocessingService:
    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
    ) -> None:
        context = build_postprocessing_context(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
        )
        if not bool(context.workflow_parameters.rspace_info.get("run_postprocessing", True)):
            return

        postprocessing_parameters = build_postprocessing_payload(context)
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
