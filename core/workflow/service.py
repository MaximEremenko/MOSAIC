from __future__ import annotations

import shutil
from pathlib import Path

from core.scattering.stage import ScatteringStage
from core.patch_centers.service import PointSelectionService
from core.decoding.stage import DecodingStage
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.stage import ResidualFieldStage
from core.structure.service import StructureLoadingService
from core.models import RunSettings, WorkflowParameters
from core.patch_centers.contracts import PointSelectionRequest


class WorkflowService:
    def __init__(
        self,
        *,
        structure_loading_service: StructureLoadingService,
        point_selection_service: PointSelectionService,
        reciprocal_space_service: ReciprocalSpacePreparationService,
        scattering_stage: ScatteringStage,
        residual_field_stage: ResidualFieldStage,
        decoding_stage: DecodingStage,
    ) -> None:
        self.structure_loading_service = structure_loading_service
        self.point_selection_service = point_selection_service
        self.reciprocal_space_service = reciprocal_space_service
        self.scattering_stage = scattering_stage
        self.residual_field_stage = residual_field_stage
        self.decoding_stage = decoding_stage

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
            Path(workflow_parameters.struct_info.working_directory)
            / "processed_point_data"
        )
        self._prepare_output_dir(output_dir, workflow_parameters)
        point_data = self.point_selection_service.select(
            PointSelectionRequest(
                method=workflow_parameters.rspace_info.method,
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
            scattering_parameters = self.scattering_stage.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
            self.residual_field_stage.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
                scattering_parameters=scattering_parameters,
            )
            self.decoding_stage.execute(
                workflow_parameters=workflow_parameters,
                structure=structure,
                artifacts=artifacts,
                client=client,
            )
        finally:
            if artifacts is not None:
                artifacts.close()

    def _prepare_output_dir(
        self, output_dir: Path, workflow_parameters: WorkflowParameters
    ) -> None:
        fresh_start = bool(workflow_parameters.rspace_info.fresh_start)
        if fresh_start and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
