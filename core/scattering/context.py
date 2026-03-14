from __future__ import annotations

import numpy as np

from core.scattering.coefficients import to_numpy
from core.config.values import first_present
from core.processing_mode import normalize_processing_mode
from core.models import AmplitudeExecutionContext, StructureData, WorkflowParameters


def build_amplitude_execution_context(
    *,
    workflow_parameters: WorkflowParameters,
    structure: StructureData,
    artifacts,
    parameter_loading_service,
    coefficient_centering_service,
    mask_strategy_service,
    interval_reconstruction_service,
) -> AmplitudeExecutionContext:
    parameters = workflow_parameters.to_payload()
    dimension = int(parameters["structInfo"]["dimension"])
    rspace = parameters["rspace_info"]
    post_mode = normalize_processing_mode(
        first_present(rspace, ("mode", "postprocess_mode", "postprocessing_mode"))
        or "displacement"
    )
    unsaved = artifacts.db_manager.get_unsaved_interval_chunks()
    pending_work = interval_reconstruction_service.load_pending_work(
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
    centered_coeff = coefficient_centering_service.center(
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
        mask_strategy=mask_strategy_service.build(
            dimension,
            parameters["peakInfo"],
            post_mode=post_mode,
        ),
        form_factor_selection=parameter_loading_service.resolve_form_factor_settings(
            workflow_parameters
        ),
    )
