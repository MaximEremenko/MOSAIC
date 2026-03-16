from __future__ import annotations

from typing import Any

from core.scattering.coefficients import CoefficientCenteringService
from core.scattering.context import build_scattering_execution_context
from core.scattering.payloads import (
    build_amplitude_adapter_payload,
    build_base_amplitude_parameters,
)
from core.config import ParameterLoadingService
from core.scattering.form_factors.registry import ScatteringWeightRegistry
from core.qspace.masking.service import MaskStrategyService
from core.qspace.intervals.interval_reconstruction import (
    IntervalReconstructionService,
)
from core.models import StructureData, WorkflowParameters


def _compute_amplitudes(**kwargs):
    from core.scattering.calculator import compute_amplitudes_delta

    return compute_amplitudes_delta(**kwargs)


class ScatteringStage:
    def __init__(
        self,
        *,
        parameter_loading_service: ParameterLoadingService,
        coefficient_centering_service: CoefficientCenteringService,
        mask_strategy_service: MaskStrategyService,
        interval_reconstruction_service: IntervalReconstructionService,
        scattering_weight_registry: ScatteringWeightRegistry | None = None,
        compute_amplitudes=_compute_amplitudes,
    ) -> None:
        registry = scattering_weight_registry or ScatteringWeightRegistry()
        self.parameter_loading_service = parameter_loading_service
        self.coefficient_centering_service = coefficient_centering_service
        self.mask_strategy_service = mask_strategy_service
        self.interval_reconstruction_service = interval_reconstruction_service
        self.scattering_weight_registry = registry
        self.compute_amplitudes = compute_amplitudes

    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
    ) -> dict[str, Any]:
        context = build_scattering_execution_context(
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
        scattering_calculator_impl = self.scattering_weight_registry.create_calculator(
            context.scattering_weight_selection
        )
        self.compute_amplitudes(
            parameters=amplitude_parameters,
            FormFactorFactoryProducer=scattering_calculator_impl,
            MaskStrategy=context.mask_strategy,
            MaskStrategyParameters=context.workflow_parameters.peak_info.to_mapping(),
            db_manager=context.artifacts.db_manager,
            output_dir=context.artifacts.output_dir,
            point_data_processor=context.artifacts.point_data_processor,
            client=client,
        )
        return base_params
