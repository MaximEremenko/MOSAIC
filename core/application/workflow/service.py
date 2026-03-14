from __future__ import annotations

from pathlib import Path

from core.application.amplitude.service import AmplitudeExecutionService
from core.application.point_selection.service import PointSelectionService
from core.application.postprocessing.service import PostprocessingService
from core.application.reciprocal_space.service import ReciprocalSpacePreparationService
from core.application.structure import StructureLoadingService
from core.domain.models import PointSelectionRequest, RunSettings, WorkflowParameters


class WorkflowService:
    def __init__(self) -> None:
        self.structure_loading_service = StructureLoadingService()
        self.point_selection_service = PointSelectionService()
        self.reciprocal_space_service = ReciprocalSpacePreparationService()
        self.amplitude_service = AmplitudeExecutionService()
        self.postprocessing_service = PostprocessingService()

    def run(
        self,
        run_settings: RunSettings,
        workflow_parameters: WorkflowParameters,
        client,
    ) -> None:
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
            self.amplitude_service.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
            self.postprocessing_service.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
        finally:
            artifacts.db_manager.close()
