from __future__ import annotations

from core.scattering.stage import ScatteringStage
from core.scattering.coefficients import CoefficientCenteringService
from core.scattering.form_factors.registry import FormFactorRegistry
from core.patch_centers.service import PointSelectionService
from core.decoding.stage import DecodingStage, build_default_decoding_processor
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.stage import ResidualFieldStage
from core.structure.service import StructureLoadingService
from core.config import ParameterLoadingService
from core.qspace.masking.service import MaskStrategyService
from core.qspace.intervals.interval_reconstruction import IntervalReconstructionService

from .service import WorkflowService


def build_default_workflow_service() -> WorkflowService:
    return WorkflowService(
        structure_loading_service=StructureLoadingService(),
        point_selection_service=PointSelectionService(),
        reciprocal_space_service=ReciprocalSpacePreparationService(),
        scattering_stage=ScatteringStage(
            parameter_loading_service=ParameterLoadingService(),
            coefficient_centering_service=CoefficientCenteringService(),
            mask_strategy_service=MaskStrategyService(),
            interval_reconstruction_service=IntervalReconstructionService(),
            form_factor_registry=FormFactorRegistry(),
        ),
        residual_field_stage=ResidualFieldStage(),
        decoding_stage=DecodingStage(processor_factory=build_default_decoding_processor),
    )
