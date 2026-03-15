from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.scattering.coefficients import to_numpy
from core.scattering.form_factors.contracts import FormFactorSelection
from core.processing_mode import normalize_processing_mode
from core.models import ReciprocalSpaceArtifacts, StructureData, WorkflowParameters


@dataclass(frozen=True)
class ScatteringExecutionContext:
    workflow_parameters: WorkflowParameters
    structure: StructureData
    artifacts: ReciprocalSpaceArtifacts
    dimension: int
    postprocessing_mode: str
    unsaved_interval_chunks: list[tuple[int, int]]
    point_rows: list[dict[str, object]]
    intervals: list[dict[str, object]]
    chemical_filtered: bool
    use_coeff: bool
    centered_coefficients: np.ndarray
    mask_strategy: object
    form_factor_selection: FormFactorSelection


def build_scattering_execution_context(
    *,
    workflow_parameters: WorkflowParameters,
    structure: StructureData,
    artifacts,
    parameter_loading_service,
    coefficient_centering_service,
    mask_strategy_service,
    interval_reconstruction_service,
) -> ScatteringExecutionContext:
    dimension = workflow_parameters.struct_info.dimension
    rspace = workflow_parameters.rspace_info
    post_mode = normalize_processing_mode(rspace.mode or "displacement")
    unsaved = artifacts.db_manager.get_unsaved_interval_chunks()
    pending_work = interval_reconstruction_service.load_pending_work(
        artifacts,
        dimension,
    )
    chemical_filtered = bool(rspace.chemical_filtered_ordering)
    use_coeff = bool(rspace.use_coeff if rspace.use_coeff is not None else True) or chemical_filtered
    coeff_center_mode = rspace.coeff_center_by or ("global" if chemical_filtered else "none")
    centered_coeff = coefficient_centering_service.center(
        np.asarray(to_numpy(structure.coeff), float),
        to_numpy(structure.refnumbers),
        coeff_center_mode,
    )
    return ScatteringExecutionContext(
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
            workflow_parameters.peak_info,
            post_mode=post_mode,
        ),
        form_factor_selection=parameter_loading_service.resolve_form_factor_settings(
            workflow_parameters
        ),
    )


__all__ = ["ScatteringExecutionContext", "build_scattering_execution_context"]
