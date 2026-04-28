from __future__ import annotations

import logging

from core.residual_field.artifacts import is_residual_field_replacement_complete
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.planning import build_residual_field_parameter_digest
from core.models import StructureData, WorkflowParameters


logger = logging.getLogger(__name__)


def _stage2_replacement_enabled(workflow_parameters: WorkflowParameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    get_value = runtime_info.get if hasattr(runtime_info, "get") else lambda key, default=None: default
    mode = get_value("scattering_stage2_mode")
    if mode is None:
        mode = get_value("stage2_mode")
    if mode is not None:
        return str(mode).strip().lower().replace("-", "_") == "replacement"
    enabled = get_value("scattering_stage2_replacement")
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"1", "true", "yes", "on", "replacement"}
    return bool(enabled)


def _replacement_expected_by_chunk(
    scattering_parameters: dict[str, object],
) -> dict[int, tuple[int, ...]]:
    raw = scattering_parameters.get("stage2_replacement_expected_by_chunk", {}) or {}
    if not isinstance(raw, dict):
        return {}
    expected: dict[int, tuple[int, ...]] = {}
    for chunk_id, interval_ids in raw.items():
        expected[int(chunk_id)] = tuple(
            sorted(int(interval_id) for interval_id in interval_ids)
        )
    return expected


def _replacement_expected_by_chunk_from_db(artifacts) -> dict[int, tuple[int, ...]]:
    db_manager = getattr(artifacts, "db_manager", None)
    get_interval_chunks = getattr(db_manager, "get_interval_chunks", None)
    if not callable(get_interval_chunks):
        return {}
    grouped: dict[int, set[int]] = {}
    for interval_id, chunk_id in get_interval_chunks():
        grouped.setdefault(int(chunk_id), set()).add(int(interval_id))
    return {
        chunk_id: tuple(sorted(interval_ids))
        for chunk_id, interval_ids in sorted(grouped.items())
    }


def _reset_expected_interval_chunks(artifacts, expected_by_chunk: dict[int, tuple[int, ...]]) -> None:
    for chunk_id, interval_ids in expected_by_chunk.items():
        for interval_id in interval_ids:
            artifacts.db_manager.update_interval_chunk_status(
                int(interval_id),
                int(chunk_id),
                saved=False,
            )


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
        if scattering_parameters is None:
            scattering_parameters = {}
        if _stage2_replacement_enabled(workflow_parameters):
            expected_by_chunk = (
                _replacement_expected_by_chunk_from_db(artifacts)
                or _replacement_expected_by_chunk(scattering_parameters)
            )
            if expected_by_chunk:
                parameter_digest = str(
                    scattering_parameters.get("residual_parameter_digest")
                    or build_residual_field_parameter_digest(workflow_parameters)
                )
                complete_by_chunk = {
                    chunk_id: is_residual_field_replacement_complete(
                        chunk_id=int(chunk_id),
                        parameter_digest=parameter_digest,
                        expected_interval_ids=interval_ids,
                        output_dir=artifacts.output_dir,
                        db_path=artifacts.db_manager.db_path,
                    )
                    for chunk_id, interval_ids in expected_by_chunk.items()
                }
                if all(complete_by_chunk.values()):
                    logger.info(
                        "Residual-field skipped: Stage-2 replacement outputs are committed for chunks %s.",
                        sorted(expected_by_chunk),
                    )
                    return scattering_parameters
                logger.warning(
                    "Stage-2 replacement outputs are incomplete for chunks %s; resetting DB status and running residual fallback.",
                    sorted(chunk_id for chunk_id, complete in complete_by_chunk.items() if not complete),
                )
                _reset_expected_interval_chunks(artifacts, expected_by_chunk)
            elif not scattering_parameters:
                return {}
        elif not scattering_parameters:
            return {}
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=client,
        )
        return scattering_parameters
