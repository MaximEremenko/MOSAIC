from __future__ import annotations

import logging

from core.residual_field.artifacts import (
    is_residual_field_replacement_complete,
    load_stage2_replacement_expected_metadata,
    normalize_stage2_replacement_expected_by_chunk,
)
from core.residual_field.commit import write_residual_no_output_manifest
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.planning import build_residual_field_parameter_digest
from core.models import StructureData, WorkflowParameters


logger = logging.getLogger(__name__)
_STAGE2_REPLACEMENT_EXPECTED_KEY = "stage2_replacement_expected_by_chunk"


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


def _replacement_expected_metadata(
    scattering_parameters: dict[str, object],
) -> dict[str, object]:
    raw = scattering_parameters.get(_STAGE2_REPLACEMENT_EXPECTED_KEY, {})
    return {
        "expected_by_chunk": normalize_stage2_replacement_expected_by_chunk(raw),
        "run_digest": scattering_parameters.get("scattering_run_digest")
        or scattering_parameters.get("run_digest"),
        "source_scattering_commit_digest": scattering_parameters.get(
            "source_scattering_commit_digest"
        ),
    }


def _resolve_replacement_expected_metadata(
    *,
    scattering_parameters: dict[str, object],
    artifacts,
    parameter_digest: str,
) -> tuple[dict[str, object], str]:
    if _STAGE2_REPLACEMENT_EXPECTED_KEY in scattering_parameters:
        return _replacement_expected_metadata(scattering_parameters), "scattering"

    expected_from_manifest = load_stage2_replacement_expected_metadata(
        output_dir=artifacts.output_dir,
        parameter_digest=parameter_digest,
    )
    if expected_from_manifest is not None:
        return expected_from_manifest, "manifest"
    return {
        "expected_by_chunk": {},
        "run_digest": None,
        "source_scattering_commit_digest": None,
    }, "absent"


def _write_empty_replacement_no_output_manifest(
    *,
    artifacts,
    metadata: dict[str, object],
    expected_source: str,
) -> None:
    run_digest = metadata.get("run_digest")
    source_scattering_commit_digest = metadata.get("source_scattering_commit_digest")
    if not run_digest or not source_scattering_commit_digest:
        raise RuntimeError(
            "Stage-2 empty replacement evidence must include run_digest and "
            "source_scattering_commit_digest before residual no_output.json can be written."
        )
    write_residual_no_output_manifest(
        output_dir=artifacts.output_dir,
        run_digest=str(run_digest),
        source_scattering_commit_digest=str(source_scattering_commit_digest),
        reason=f"empty replacement coverage from {expected_source}",
    )


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
            parameter_digest = str(
                scattering_parameters.get("residual_parameter_digest")
                or build_residual_field_parameter_digest(workflow_parameters)
            )
            expected_metadata, expected_source = _resolve_replacement_expected_metadata(
                scattering_parameters=scattering_parameters,
                artifacts=artifacts,
                parameter_digest=parameter_digest,
            )
            expected_by_chunk = dict(expected_metadata["expected_by_chunk"])
            if expected_by_chunk:
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
            elif expected_source in {"scattering", "manifest"}:
                _write_empty_replacement_no_output_manifest(
                    artifacts=artifacts,
                    metadata=expected_metadata,
                    expected_source=expected_source,
                )
                logger.info(
                    "Residual-field skipped: Stage-2 replacement expected no residual chunks."
                )
                return scattering_parameters
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
