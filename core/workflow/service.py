from __future__ import annotations

from pathlib import Path

from core.scattering.service import ScatteringExecutionService
from core.patch_centers.service import PointSelectionService
from core.decoding.service import DecodingService
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.service import ResidualFieldExecutionService
from core.structure.service import StructureLoadingService
from core.models import PointSelectionRequest, RunSettings, WorkflowParameters


class WorkflowService:
    def __init__(
        self,
        *,
        structure_loading_service: StructureLoadingService,
        point_selection_service: PointSelectionService,
        reciprocal_space_service: ReciprocalSpacePreparationService,
        amplitude_service: ScatteringExecutionService,
        residual_field_service: ResidualFieldExecutionService,
        postprocessing_service: DecodingService,
    ) -> None:
        self.structure_loading_service = structure_loading_service
        self.point_selection_service = point_selection_service
        self.reciprocal_space_service = reciprocal_space_service
        self.amplitude_service = amplitude_service
        self.residual_field_service = residual_field_service
        self.postprocessing_service = postprocessing_service

    def run(
        self,
        run_settings: RunSettings,
        workflow_parameters: WorkflowParameters,
        client,
    ) -> None:
        artifacts = None
        structure = self.structure_loading_service.load(
            workflow_parameters,
            str(run_settings.working_path),
        )
        output_dir = (
            Path(workflow_parameters.struct_info["working_directory"])
            / "processed_point_data"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        point_data = self.point_selection_service.select(
            PointSelectionRequest(
                method=workflow_parameters.rspace_info["method"],
                parameters=workflow_parameters,
                structure=structure,
                hdf5_file_path=str(output_dir / "point_data.hdf5"),
            )
        )
        artifacts = self.reciprocal_space_service.prepare(
            workflow_parameters=workflow_parameters,
            point_data=point_data,
            supercell=structure.supercell,
            output_dir=str(output_dir),
        )
        try:
            scattering_parameters = self.amplitude_service.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
            self.residual_field_service.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
                scattering_parameters=scattering_parameters,
            )
            self.postprocessing_service.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
        finally:
            if artifacts is not None:
                artifacts.close()
