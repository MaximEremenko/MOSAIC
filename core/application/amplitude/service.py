from __future__ import annotations

from typing import Any

from core.application.amplitude.coefficients import CoefficientCenteringService
from core.application.amplitude.context import build_amplitude_execution_context
from core.application.amplitude.payloads import (
    build_amplitude_adapter_payload,
    build_base_amplitude_parameters,
)
from core.application.configuration import ParameterLoadingService
from core.application.form_factors.registry import FormFactorRegistry
from core.application.masking.service import MaskStrategyService
from core.application.reciprocal_space.interval_reconstruction import (
    IntervalReconstructionService,
)
from core.domain.models import StructureData, WorkflowParameters


class AmplitudeExecutionService:
    def __init__(self) -> None:
        self.parameter_loading_service = ParameterLoadingService()
        self.coefficient_centering_service = CoefficientCenteringService()
        self.mask_strategy_service = MaskStrategyService()
        self.interval_reconstruction_service = IntervalReconstructionService()
        self.form_factor_registry = FormFactorRegistry()

    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
    ) -> dict[str, Any]:
        context = build_amplitude_execution_context(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            parameter_loading_service=self.parameter_loading_service,
            coefficient_centering_service=self.coefficient_centering_service,
            mask_strategy_service=self.mask_strategy_service,
            interval_reconstruction_service=self.interval_reconstruction_service,
        )
        if not context.unsaved_interval_chunks:
            return {}
        base_params = build_base_amplitude_parameters(context)
        amplitude_parameters = build_amplitude_adapter_payload(context, base_params)
        ff_calculator_impl = self.form_factor_registry.create_calculator(
            context.form_factor_selection
        )
        from core.application.amplitude.calculator import compute_amplitudes_delta

        compute_amplitudes_delta(
            parameters=amplitude_parameters,
            FormFactorFactoryProducer=ff_calculator_impl,
            MaskStrategy=context.mask_strategy,
            MaskStrategyParameters=context.workflow_parameters.peak_info,
            db_manager=context.artifacts.db_manager,
            output_dir=context.artifacts.output_dir,
            point_data_processor=context.artifacts.point_data_processor,
            client=client,
        )
        return base_params
