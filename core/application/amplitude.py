from __future__ import annotations

import numpy as np

from core.application.common import first_present, normalize_post_mode, pad_interval
from core.application.coefficients import CoefficientCenteringService, to_numpy
from core.application.form_factor_registry import FormFactorRegistry
from core.application.intervals import IntervalReconstructionService
from core.application.masking import MaskStrategyService
from core.application.parameters import ParameterLoadingService
from core.domain.models import (
    AmplitudeExecutionContext,
    StructureData,
    WorkflowParameters,
)


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
        context = self._build_context(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
        )
        if not context.unsaved_interval_chunks:
            return {}
        base_params = self._build_base_parameters(context)
        amplitude_parameters = self._build_adapter_payload(context, base_params)
        ff_calculator_impl = self.form_factor_registry.create_calculator(
            context.form_factor_selection
        )
        from core.processors.amplitude_delta_calculator import compute_amplitudes_delta

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

    def _build_context(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
    ) -> AmplitudeExecutionContext:
        parameters = workflow_parameters.to_payload()
        dimension = int(parameters["structInfo"]["dimension"])
        rspace = parameters["rspace_info"]
        post_mode = normalize_post_mode(
            first_present(rspace, ("mode", "postprocess_mode", "postprocessing_mode"))
            or "displacement"
        )
        unsaved = artifacts.db_manager.get_unsaved_interval_chunks()
        pending_work = self.interval_reconstruction_service.load_pending_work(
            artifacts,
            dimension,
        )
        chemical_filtered = bool(
            first_present(rspace, ("chemical_filtered_ordering", "chemical_filtered"))
            or first_present(
                parameters["structInfo"],
                ("chemical_filtered_ordering", "chemical_filtered"),
            )
        )
        use_coeff = bool(rspace.get("use_coeff", True)) or chemical_filtered
        coeff_center_mode = first_present(
            rspace, ("coeff_center_by", "coeff_center_mode", "chemical_coeff_center_by")
        ) or ("global" if chemical_filtered else "none")
        centered_coeff = self.coefficient_centering_service.center(
            np.asarray(to_numpy(structure.coeff), float),
            to_numpy(structure.refnumbers),
            coeff_center_mode,
        )
        return AmplitudeExecutionContext(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            dimension=dimension,
            postprocessing_mode=post_mode,
            unsaved_interval_chunks=unsaved,
            point_rows=pending_work.point_rows,
            intervals=pending_work.intervals,
            chemical_filtered=chemical_filtered,
            use_coeff=use_coeff,
            centered_coefficients=centered_coeff,
            mask_strategy=self.mask_strategy_service.build(
                dimension,
                parameters["peakInfo"],
                post_mode=post_mode,
            ),
            form_factor_selection=self.parameter_loading_service.resolve_form_factor_settings(
                workflow_parameters
            ),
        )

    def _build_base_parameters(
        self, context: AmplitudeExecutionContext
    ) -> dict[str, object]:
        return {
            "reciprocal_space_intervals": context.intervals,
            "reciprocal_space_intervals_all": context.artifacts.padded_intervals,
            "point_data_list": [
                {
                    "central_point_id": row["central_point_id"],
                    "coordinates": row["coordinates"],
                    "dist_from_atom_center": row["dist_from_atom_center"],
                    "step_in_frac": row["step_in_frac"],
                    "chunk_id": row["chunk_id"],
                    "grid_amplitude_initialized": row["grid_amplitude_initialized"],
                    "id": row["central_point_id"],
                }
                for row in context.point_rows
            ],
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

    def _build_adapter_payload(
        self,
        context: AmplitudeExecutionContext,
        base_params: dict[str, object],
    ) -> dict[str, object]:
        amplitude_parameters = dict(base_params)
        if context.use_coeff:
            amplitude_parameters["coeff"] = np.asarray(
                context.centered_coefficients, float
            )
        if context.chemical_filtered:
            amplitude_parameters["original_coords"] = to_numpy(
                context.structure.cells_origin
            )
        return amplitude_parameters
