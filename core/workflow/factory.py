from __future__ import annotations

from core.scattering.service import ScatteringExecutionService
from core.patch_centers.service import PointSelectionService
from core.decoding.service import DecodingService
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.service import ResidualFieldExecutionService
from core.structure.service import StructureLoadingService

from .service import WorkflowService


def build_default_workflow_service() -> WorkflowService:
    return WorkflowService(
        structure_loading_service=StructureLoadingService(),
        point_selection_service=PointSelectionService(),
        reciprocal_space_service=ReciprocalSpacePreparationService(),
        amplitude_service=ScatteringExecutionService(),
        residual_field_service=ResidualFieldExecutionService(),
        postprocessing_service=DecodingService(),
    )
