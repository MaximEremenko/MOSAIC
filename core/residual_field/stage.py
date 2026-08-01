from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from core.contracts import ScatteringHandoff
from core.residual_field.backend import resolve_residual_field_reducer_backend
from core.residual_field.execution import run_residual_field_stage
from core.residual_field.planning import build_residual_field_parameter_digest
from core.models import StructureData, WorkflowParameters
from core.runtime import resolve_worker_scratch_root, short_path

if TYPE_CHECKING:
    from core.workflow.context import RunArtifacts


logger = logging.getLogger(__name__)


def _fan_out_recover_finalizes(
    *,
    client,
    backend,
    chunk_ids,
    parameter_digest,
    output_dir,
    db_path,
    scratch_root,
):
    """Run recovery finalizes as worker tasks, one per pending chunk.

    Driver-serial recovery cost ~80-90 s/chunk (34 GB snapshot reads +
    13.5 GB writes each). Pre-plan there is no ownership map and no live
    accumulator state, so placement is pure round-robin load balancing;
    allow_other_workers=True so a dying worker cannot strand recovery.
    Returns None when fan-out is unavailable (no client/workers, or a
    test-double backend) — the caller then runs the serial path."""
    if not chunk_ids:
        return []
    layout_kind = getattr(getattr(backend, "layout", None), "kind", None)
    if client is None or layout_kind != "local_restartable":
        return None
    try:
        from core.residual_field.backend import (
            finalize_process_local_residual_chunk,
        )
        from core.residual_field.cluster_helpers import _current_worker_addresses
        from core.runtime.dask_helpers import is_sync_client

        if is_sync_client(client):
            return None
        workers = _current_worker_addresses(client)
    except Exception:
        return None
    if not workers:
        return None
    futures = [
        client.submit(
            finalize_process_local_residual_chunk,
            backend,
            chunk_id=int(chunk_id),
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            cleanup_policy="off",
            scratch_root=scratch_root,
            quiet_logs=True,
            opportunistic=True,
            pure=False,
            workers=[workers[index % len(workers)]],
            allow_other_workers=True,
        )
        for index, chunk_id in enumerate(chunk_ids)
    ]
    return [future.result() for future in futures]


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
        if db_manager is None:
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
        # Cheap serial pre-filter (small shared-FS manifest reads).
        chunks_with_progress: list[int] = []
        for chunk_id in sorted(int(value) for value in db_manager.get_pending_chunk_ids()):
            progress = load_progress_manifest(
                output_dir=artifacts.output_dir,
                chunk_id=int(chunk_id),
                parameter_digest=parameter_digest,
            )
            if progress is not None:
                chunks_with_progress.append(int(chunk_id))

        manifests = _fan_out_recover_finalizes(
            client=client,
            backend=backend,
            chunk_ids=chunks_with_progress,
            parameter_digest=parameter_digest,
            output_dir=artifacts.output_dir,
            db_path=db_manager.db_path,
            scratch_root=scratch_root,
        )
        if manifests is None:
            # No live workers (sync/local runs) — driver-serial as before.
            manifests = [
                finalize_chunk(
                    chunk_id=int(chunk_id),
                    parameter_digest=parameter_digest,
                    output_dir=artifacts.output_dir,
                    db_path=db_manager.db_path,
                    cleanup_policy="off",
                    scratch_root=scratch_root,
                    quiet_logs=True,
                    # Pre-plan recovery: partitioned families whose
                    # completeness cannot be proven are deferred to the
                    # residual stage instead of being published or failed.
                    opportunistic=True,
                )
                for chunk_id in chunks_with_progress
            ]
        recovered_chunks = [
            int(chunk_id)
            for chunk_id, manifest in zip(chunks_with_progress, manifests)
            if manifest is not None
        ]

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
        if handoff.is_empty:
            return {}
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=client,
        )
        return original
