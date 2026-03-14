from __future__ import annotations

from core.residual_field.execution import run_residual_field_stage
from core.models import StructureData, WorkflowParameters


class ResidualFieldExecutionService:
    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts,
        client,
        *,
        scattering_parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not scattering_parameters:
            return {}
        run_residual_field_stage(
            parameters=scattering_parameters,
            db_manager=artifacts.db_manager,
            output_dir=artifacts.output_dir,
            client=client,
        )
        return scattering_parameters


ResidualFieldService = ResidualFieldExecutionService
