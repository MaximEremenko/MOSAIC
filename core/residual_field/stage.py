from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from core.contracts import ScatteringHandoff
from core.residual_field.artifacts import (
    is_residual_field_replacement_complete,
    load_stage2_replacement_expected_metadata,
    normalize_stage2_replacement_expected_by_chunk,
)
from core.residual_field.backend import resolve_residual_field_reducer_backend
from core.residual_field.commit import write_residual_no_output_manifest
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.planning import build_residual_field_parameter_digest
from core.models import StructureData, WorkflowParameters
from core.runtime import resolve_worker_scratch_root, short_path

if TYPE_CHECKING:
    from core.workflow.context import RunArtifacts


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
    handoff: ScatteringHandoff,
) -> dict[str, object]:
    return {
        "expected_by_chunk": normalize_stage2_replacement_expected_by_chunk(
            handoff.expected_by_chunk()
        ),
        "run_digest": handoff.scattering_run_digest or handoff.run_digest,
        "source_scattering_commit_digest": handoff.source_scattering_commit_digest,
    }


def _resolve_replacement_expected_metadata(
    *,
    handoff: ScatteringHandoff,
    artifacts: RunArtifacts,
    parameter_digest: str,
) -> tuple[dict[str, object], str]:
    if handoff.has_stage2_replacement_expected:
        return _replacement_expected_metadata(handoff), "scattering"

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
    artifacts: RunArtifacts,
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


def _reset_expected_interval_chunks(artifacts: RunArtifacts, expected_by_chunk: dict[int, tuple[int, ...]]) -> None:
    rows = [
        (int(interval_id), int(chunk_id), 0)
        for chunk_id, interval_ids in expected_by_chunk.items()
        for interval_id in interval_ids
    ]
    batch = getattr(artifacts.db_manager, "update_interval_chunk_status_batch", None)
    if callable(batch):
        batch(rows)
    else:
        for interval_id, chunk_id, _saved in rows:
            artifacts.db_manager.update_interval_chunk_status(
                interval_id, chunk_id, saved=False
            )


class ResidualFieldStage:
    def recover_pending(
        self,
        *,
        workflow_parameters: WorkflowParameters,
        artifacts: RunArtifacts,
        client,
    ) -> list[int]:
        """Finalize any committed local-restart chunk state before scattering.

        On a restart, the durable local-accumulator snapshots and reducer
        progress for a chunk may already be complete even though the final
        chunk artifacts were never committed. Finalizing them here lets the
        downstream stages skip recomputation. Only committed progress is
        finalized; uncommitted task work is left to be recomputed.
        """
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
                # Pre-plan recovery: partitioned families whose completeness
                # cannot be proven are deferred to the residual stage instead
                # of being published or failed.
                opportunistic=True,
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

    def execute(
        self,
        workflow_parameters: WorkflowParameters,
        structure: StructureData,
        artifacts: RunArtifacts,
        client,
        *,
        scattering_parameters: ScatteringHandoff | Mapping[str, object] | None = None,
    ) -> ScatteringHandoff | Mapping[str, object]:
        # Accept either the typed handoff or a plain mapping (or None) and
        # normalise to a ScatteringHandoff for typed reads. The original argument
        # is returned unchanged so the pass-through contract with
        # workflow/service.py (and direct callers) is preserved.
        original = scattering_parameters
        handoff = (
            scattering_parameters
            if isinstance(scattering_parameters, ScatteringHandoff)
            else ScatteringHandoff.from_mapping(scattering_parameters)
        )
        if _stage2_replacement_enabled(workflow_parameters):
            parameter_digest = str(
                handoff.residual_parameter_digest
                or build_residual_field_parameter_digest(workflow_parameters)
            )
            expected_metadata, expected_source = _resolve_replacement_expected_metadata(
                handoff=handoff,
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
                    return original
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
                return original
            elif handoff.is_empty:
                return {}
        elif handoff.is_empty:
            return {}
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=client,
        )
        return original
