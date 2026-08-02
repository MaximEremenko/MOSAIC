from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from core.scattering.stage import ScatteringStage
from core.patch_centers.service import PointSelectionService
from core.decoding.stage import DecodingStage
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.stage import ResidualFieldStage
from core.structure.identity import (
    enforce_output_dir_structure,
    structure_content_digest_from_structure,
)
from core.structure.service import StructureLoadingService
from core.models import RunSettings, WorkflowParameters
from core.patch_centers.contracts import PointSelectionRequest
from core.storage.db_cache import resolve_db_cache_config

if TYPE_CHECKING:
    from core.workflow.context import RunArtifacts


logger = logging.getLogger(__name__)


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
        *,
        db_path: str | None = None,
        no_db_cache: bool = False,
    ) -> None:
        artifacts: RunArtifacts | None = None
        structure = self.structure_loading_service.load(
            workflow_parameters,
            str(run_settings.working_path),
        )
        # One structure identity per run, computed before any stage and
        # published to all of them. Every downstream identity that
        # addresses reusable work folds it in; without it a re-run of the
        # same output directory with different coordinates resolves to the
        # same run tree and republishes the previous structure's results.
        workflow_parameters.runtime_info.extra["source_structure_digest"] = (
            structure_content_digest_from_structure(structure)
        )
        output_dir = (
            Path(workflow_parameters.struct_info.working_directory)
            / "processed_point_data"
        )
        self._prepare_output_dir(output_dir, workflow_parameters)
        # After _prepare_output_dir, so fresh_start (which removes the
        # directory) stays the supported way to retarget one at a new
        # structure, and before any stage computes anything.
        enforce_output_dir_structure(
            output_dir,
            workflow_parameters.runtime_info.extra["source_structure_digest"],
        )
        runtime_info = workflow_parameters.runtime_info.to_mapping()
        db_cache_config = resolve_db_cache_config(
            run_settings=run_settings,
            workflow_parameters=workflow_parameters,
            run_digest=(
                runtime_info.get("scattering_run_digest")
                or runtime_info.get("residual_run_digest")
                or runtime_info.get("residual_field_run_digest")
            ),
            output_dir=output_dir,
            db_path=db_path,
            no_db_cache=no_db_cache,
        )
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
            db_cache_config=db_cache_config,
        )
        try:
            self.residual_field_stage.recover_pending(
                workflow_parameters=workflow_parameters,
                artifacts=artifacts,
                client=client,
            )
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
