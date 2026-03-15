from __future__ import annotations

from core.decoding.context import build_decoding_context
from core.decoding.payloads import build_decoding_payload
from core.decoding.processor import PointDataPostprocessingProcessor
from core.models import StructureData, WorkflowParameters


def build_default_decoding_processor(db_manager, point_data_processor, parameters):
    return PointDataPostprocessingProcessor(
        db_manager,
        point_data_processor,
        parameters,
    )


class DecodingStage:
    def __init__(self, *, processor_factory) -> None:
        self.processor_factory = processor_factory

    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
    ) -> None:
        context = build_decoding_context(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
        )
        if not bool(context.workflow_parameters.rspace_info.run_postprocessing):
            return

        postprocessing_parameters = build_decoding_payload(context)
        processor = self.processor_factory(
            context.artifacts.db_manager,
            context.artifacts.point_data_processor,
            postprocessing_parameters,
        )
        for chunk_id in sorted(context.artifacts.db_manager.get_pending_chunk_ids()):
            processor.process_chunk(
                int(chunk_id),
                context.artifacts.saver,
                output_dir=context.artifacts.output_dir,
            )
