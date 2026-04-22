from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from core.residual_field.backend import resolve_residual_field_reducer_backend
from core.residual_field.planning import build_residual_field_parameter_digest
from core.scattering.stage import ScatteringStage
from core.patch_centers.service import PointSelectionService
from core.decoding.stage import DecodingStage
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.stage import ResidualFieldStage
from core.structure.service import StructureLoadingService
from core.models import RunSettings, WorkflowParameters
from core.patch_centers.contracts import PointSelectionRequest
from core.runtime import resolve_worker_scratch_root, short_path


logger = logging.getLogger(__name__)


def recover_local_residual_state_before_scattering(
    *,
    workflow_parameters: WorkflowParameters,
    artifacts,
    client,
) -> list[int]:
    db_manager = getattr(artifacts, "db_manager", None)
    get_pending_chunk_ids = getattr(db_manager, "get_pending_chunk_ids", None)
    if not callable(get_pending_chunk_ids):
        return []

    backend = resolve_residual_field_reducer_backend(
        workflow_parameters=workflow_parameters,
        client=client,
    )
    if getattr(getattr(backend, "layout", None), "kind", None) != "local_restartable":
        return []

    load_progress_manifest = getattr(backend, "load_progress_manifest", None)
    finalize_chunk = getattr(backend, "finalize_chunk", None)
    if not callable(load_progress_manifest) or not callable(finalize_chunk):
        return []

    explicit_scratch_root = workflow_parameters.runtime_info.get(
        "residual_shard_scratch_root",
        os.getenv("MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT"),
    )
    scratch_root = resolve_worker_scratch_root(
        preferred=(
            explicit_scratch_root
            if explicit_scratch_root is not None
            else str(Path(artifacts.output_dir) / ".local_restartable")
        ),
        stage="residual_field",
    )
    parameter_digest = build_residual_field_parameter_digest(workflow_parameters)
    recovered_chunks: list[int] = []
    for chunk_id in sorted(int(value) for value in get_pending_chunk_ids()):
        progress = load_progress_manifest(
            output_dir=artifacts.output_dir,
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
        )
        if progress is None:
            continue
        manifest = finalize_chunk(
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=artifacts.output_dir,
            db_path=db_manager.db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
        )
        if manifest is not None:
            recovered_chunks.append(int(chunk_id))

    if recovered_chunks:
        logger.info(
            "Recovered residual-field local restart state before scattering | chunks=%s | digest=%s | scratch=%s",
            recovered_chunks,
            parameter_digest,
            short_path(scratch_root),
        )
    return recovered_chunks


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
            recover_local_residual_state_before_scattering(
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
