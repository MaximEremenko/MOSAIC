from __future__ import annotations

from typing import Any

from core.scattering.coefficients import CoefficientCenteringService
from core.scattering.context import build_amplitude_execution_context
from core.scattering.payloads import (
    build_amplitude_adapter_payload,
    build_base_amplitude_parameters,
)
from core.config import ParameterLoadingService
from core.scattering.form_factors.registry import FormFactorRegistry
from core.qspace.masking.service import MaskStrategyService
from core.qspace.intervals.interval_reconstruction import (
    IntervalReconstructionService,
)
from core.models import StructureData, WorkflowParameters


def _compute_amplitudes(**kwargs):
    from core.scattering.calculator import compute_amplitudes_delta

    return compute_amplitudes_delta(**kwargs)


class ScatteringExecutionService:
    def __init__(
        self,
        *,
        parameter_loading_service: ParameterLoadingService | None = None,
        coefficient_centering_service: CoefficientCenteringService | None = None,
        mask_strategy_service: MaskStrategyService | None = None,
        interval_reconstruction_service: IntervalReconstructionService | None = None,
        form_factor_registry: FormFactorRegistry | None = None,
        compute_amplitudes=_compute_amplitudes,
    ) -> None:
        self.parameter_loading_service = (
            parameter_loading_service or ParameterLoadingService()
        )
        self.coefficient_centering_service = (
            coefficient_centering_service or CoefficientCenteringService()
        )
        self.mask_strategy_service = mask_strategy_service or MaskStrategyService()
        self.interval_reconstruction_service = (
            interval_reconstruction_service or IntervalReconstructionService()
        )
        self.form_factor_registry = form_factor_registry or FormFactorRegistry()
        self.compute_amplitudes = compute_amplitudes

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
        self.compute_amplitudes(
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
