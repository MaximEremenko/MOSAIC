from __future__ import annotations

from core.residual_field.execution import run_residual_field_stage
from core.models import StructureData, WorkflowParameters


class ResidualFieldStage:
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
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=client,
        )
        return scattering_parameters
