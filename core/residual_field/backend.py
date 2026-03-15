from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from core.residual_field.artifacts import (
    delete_reclaimable_residual_field_shards,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    persist_residual_field_shard_checkpoint,
    reconcile_residual_field_reducer_progress,
    reduce_residual_field_shards_for_chunk,
    write_residual_field_reducer_progress_manifest,
)
from core.residual_field.contracts import (
    ResidualFieldArtifactManifest,
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)
from core.runtime import is_sync_client

if TYPE_CHECKING:
    from core.models import WorkflowParameters


ResidualFieldReducerBackendKind = Literal[
    "local_restartable",
    "durable_shared_restartable",
]
ScatteringIntervalArtifactPolicy = Literal[
    "required_transport",
    "optional_output",
]
ResidualShardCheckpointPolicy = Literal[
    "required_local_restart_state",
    "required_durable_checkpoint",
]
ScratchRolePolicy = Literal[
    "committed_local_restart_state_and_temporary_staging",
    "temporary_staging_only",
]


@dataclass(frozen=True)
class ResidualFieldCheckpointPolicy:
    interval_artifacts: ScatteringIntervalArtifactPolicy
    shard_checkpoints: ResidualShardCheckpointPolicy
    reducer_progress_manifest: Literal["required_durable"]
    final_chunk_artifacts: Literal["required_durable"]
    worker_local_scratch_role: ScratchRolePolicy


@dataclass(frozen=True)
class ResidualFieldReducerBackendLayout:
    """
    Explicit state-placement description for the current restartable reducer path.

    Wave 1 keeps the manifest-driven reducer model unchanged and only makes
    backend/storage semantics explicit. Uncommitted work may still be recomputed;
    only committed shard/progress/final artifacts are restart state.
    """

    kind: ResidualFieldReducerBackendKind
    reducer_ownership: str
    reducer_backing_store: str
    durability_policy: str
    checkpoint_policy: ResidualFieldCheckpointPolicy
    ram_state: tuple[str, ...]
    local_scratch_state: tuple[str, ...]
    durable_state: tuple[str, ...]
    scattering_interval_transport: str
    scattering_interval_outputs_supported: bool
    direct_interval_handoff_supported: bool
    persist_interval_artifacts_by_default: bool
    committed_shard_storage: str
    shard_compression: str
    uncommitted_restart_rule: str


@dataclass(frozen=True)
class ResidualFieldReducerRuntimeState:
    kind: ResidualFieldReducerBackendKind
    reducer_ownership: str
    reducer_backing_store: str
    durability_policy: str
    checkpoint_policy: ResidualFieldCheckpointPolicy
    ram_state: tuple[str, ...]
    local_scratch_root: str | None
    local_scratch_state: tuple[str, ...]
    durable_root: str
    durable_state: tuple[str, ...]
    scattering_interval_transport: str
    scattering_interval_outputs_supported: bool
    direct_interval_handoff_supported: bool
    persist_interval_artifacts_by_default: bool
    committed_shard_root: str
    committed_shard_storage: str
    shard_compression: str
    uncommitted_restart_rule: str


class ResidualFieldReducerBackend(Protocol):
    layout: ResidualFieldReducerBackendLayout

    def describe_runtime_state(
        self,
        *,
        output_dir: str,
        scratch_root: str | None,
    ) -> ResidualFieldReducerRuntimeState:
        ...

    def persist_shard_checkpoint(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        grid_shape_nd,
        total_reciprocal_points: int,
        contribution_reciprocal_points: int,
        amplitudes_delta,
        amplitudes_average,
        point_ids=None,
        output_dir: str,
        scratch_root: str | None = None,
        quiet_logs: bool = False,
    ) -> ResidualFieldShardManifest:
        ...

    def discover_shard_manifests(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
        scratch_root: str | None = None,
    ) -> list[ResidualFieldShardManifest]:
        ...

    def load_progress_manifest(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
    ) -> ResidualFieldReducerProgressManifest | None:
        ...

    def write_progress_manifest(
        self,
        manifest: ResidualFieldReducerProgressManifest,
    ) -> ResidualFieldReducerProgressManifest:
        ...

    def reconcile_progress(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        scratch_root: str | None = None,
    ) -> ResidualFieldReducerProgressManifest | None:
        ...

    def finalize_chunk(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        cleanup_policy: str | bool | None = None,
        scratch_root: str | None = None,
        quiet_logs: bool = False,
    ) -> ResidualFieldArtifactManifest | None:
        ...

    def cleanup_reclaimable_shards(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        scratch_root: str | None = None,
    ) -> tuple[str, ...]:
        ...


LOCAL_RESTARTABLE_LAYOUT = ResidualFieldReducerBackendLayout(
    kind="local_restartable",
    reducer_ownership="single-writer chunk-owned reducer",
    reducer_backing_store="manifest-driven local scratch shard commit with durable final chunk commit",
    durability_policy=(
        "restart from committed local shard/progress/final artifacts for the same-node run; "
        "uncommitted local task work may be recomputed"
    ),
    checkpoint_policy=ResidualFieldCheckpointPolicy(
        interval_artifacts="optional_output",
        shard_checkpoints="required_local_restart_state",
        reducer_progress_manifest="required_durable",
        final_chunk_artifacts="required_durable",
        worker_local_scratch_role="committed_local_restart_state_and_temporary_staging",
    ),
    ram_state=(
        "task-local inverse outputs before shard commit",
        "reducer-local merged chunk state while reconciling committed shards",
    ),
    local_scratch_state=(
        "committed local residual shard checkpoints",
        "temporary shard build files before local shard commit",
    ),
    durable_state=(
        "reducer progress manifest",
        "final residual-field chunk artifact family",
    ),
    scattering_interval_transport="direct in-process interval payload handoff preferred; saved interval outputs optional",
    scattering_interval_outputs_supported=True,
    direct_interval_handoff_supported=True,
    persist_interval_artifacts_by_default=False,
    committed_shard_storage="local scratch",
    shard_compression="np.savez",
    uncommitted_restart_rule=(
        "if a task crashes before local shard commit or before reducer commit, recompute the "
        "forward/inverse work for that interval batch; artifact existence alone never implies completion"
    ),
)

SHARED_DURABLE_LAYOUT = ResidualFieldReducerBackendLayout(
    kind="durable_shared_restartable",
    reducer_ownership="single-writer chunk-owned reducer",
    reducer_backing_store="manifest-driven durable shared-storage commit",
    durability_policy=(
        "restart from committed shard/progress/final artifacts visible to workers/jobs; "
        "uncommitted task-local work may be recomputed"
    ),
    checkpoint_policy=ResidualFieldCheckpointPolicy(
        interval_artifacts="required_transport",
        shard_checkpoints="required_durable_checkpoint",
        reducer_progress_manifest="required_durable",
        final_chunk_artifacts="required_durable",
        worker_local_scratch_role="temporary_staging_only",
    ),
    ram_state=LOCAL_RESTARTABLE_LAYOUT.ram_state,
    local_scratch_state=LOCAL_RESTARTABLE_LAYOUT.local_scratch_state,
    durable_state=LOCAL_RESTARTABLE_LAYOUT.durable_state,
    scattering_interval_transport="durable interval artifacts required execution transport",
    scattering_interval_outputs_supported=True,
    direct_interval_handoff_supported=False,
    persist_interval_artifacts_by_default=True,
    committed_shard_storage="durable shared storage",
    shard_compression="np.savez_compressed",
    uncommitted_restart_rule=LOCAL_RESTARTABLE_LAYOUT.uncommitted_restart_rule,
)


class ManifestDrivenResidualFieldReducerBackend:
    """
    Wave 1 concrete backend.

    Both local and distributed modes keep the current manifest-driven shard and
    reducer-progress semantics. The backend object makes the execution/storage
    policy explicit without changing single-writer reducer ownership.
    """

    def __init__(
        self,
        layout: ResidualFieldReducerBackendLayout,
        *,
        shard_storage_root_override: str | None = None,
    ) -> None:
        self.layout = layout
        self.shard_storage_root_override = (
            str(Path(shard_storage_root_override).expanduser())
            if shard_storage_root_override
            else None
        )

    def describe_runtime_state(
        self,
        *,
        output_dir: str,
        scratch_root: str | None,
    ) -> ResidualFieldReducerRuntimeState:
        committed_shard_root = self.resolve_shard_storage_root(
            output_dir=output_dir,
            scratch_root=scratch_root,
        )
        return ResidualFieldReducerRuntimeState(
            kind=self.layout.kind,
            reducer_ownership=self.layout.reducer_ownership,
            reducer_backing_store=self.layout.reducer_backing_store,
            durability_policy=self.layout.durability_policy,
            checkpoint_policy=self.layout.checkpoint_policy,
            ram_state=self.layout.ram_state,
            local_scratch_root=str(scratch_root) if scratch_root else None,
            local_scratch_state=self.layout.local_scratch_state,
            durable_root=str(output_dir),
            durable_state=self.layout.durable_state,
            scattering_interval_transport=self.layout.scattering_interval_transport,
            scattering_interval_outputs_supported=self.layout.scattering_interval_outputs_supported,
            direct_interval_handoff_supported=self.layout.direct_interval_handoff_supported,
            persist_interval_artifacts_by_default=self.layout.persist_interval_artifacts_by_default,
            committed_shard_root=committed_shard_root,
            committed_shard_storage=self.layout.committed_shard_storage,
            shard_compression=self.layout.shard_compression,
            uncommitted_restart_rule=self.layout.uncommitted_restart_rule,
        )

    def resolve_shard_storage_root(
        self,
        *,
        output_dir: str,
        scratch_root: str | None,
    ) -> str:
        if self.shard_storage_root_override is not None:
            return self.shard_storage_root_override
        if self.layout.kind == "durable_shared_restartable":
            return str(output_dir)
        root = scratch_root or str(Path(output_dir) / ".local_restartable")
        return str(Path(root).expanduser())

    def uses_direct_interval_handoff(self) -> bool:
        return bool(self.layout.direct_interval_handoff_supported)

    def persist_interval_artifacts_by_default(self) -> bool:
        return bool(self.layout.persist_interval_artifacts_by_default)

    def interval_artifacts_required_for_transport(self) -> bool:
        return self.layout.checkpoint_policy.interval_artifacts == "required_transport"

    def shard_checkpoints_require_durable_storage(self) -> bool:
        return (
            self.layout.checkpoint_policy.shard_checkpoints
            == "required_durable_checkpoint"
        )

    def persist_shard_checkpoint(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        grid_shape_nd,
        total_reciprocal_points: int,
        contribution_reciprocal_points: int,
        amplitudes_delta,
        amplitudes_average,
        point_ids=None,
        output_dir: str,
        scratch_root: str | None = None,
        quiet_logs: bool = False,
    ) -> ResidualFieldShardManifest:
        return persist_residual_field_shard_checkpoint(
            work_unit,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_ids=point_ids,
            output_dir=output_dir,
            scratch_root=scratch_root,
            shard_storage_root=self.resolve_shard_storage_root(
                output_dir=output_dir,
                scratch_root=scratch_root,
            ),
            compress=self.layout.kind != "local_restartable",
            quiet_logs=quiet_logs,
        )

    def discover_shard_manifests(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
        scratch_root: str | None = None,
    ) -> list[ResidualFieldShardManifest]:
        return discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=self.resolve_shard_storage_root(
                output_dir=output_dir,
                scratch_root=scratch_root,
            ),
        )

    def load_progress_manifest(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
    ) -> ResidualFieldReducerProgressManifest | None:
        return discover_residual_field_reducer_progress_manifest(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        )

    def write_progress_manifest(
        self,
        manifest: ResidualFieldReducerProgressManifest,
    ) -> ResidualFieldReducerProgressManifest:
        return write_residual_field_reducer_progress_manifest(manifest)

    def reconcile_progress(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        scratch_root: str | None = None,
    ) -> ResidualFieldReducerProgressManifest | None:
        return reconcile_residual_field_reducer_progress(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=manifests,
            shard_storage_root=self.resolve_shard_storage_root(
                output_dir=output_dir,
                scratch_root=scratch_root,
            ),
        )

    def finalize_chunk(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        cleanup_policy: str | bool | None = None,
        scratch_root: str | None = None,
        quiet_logs: bool = False,
    ) -> ResidualFieldArtifactManifest | None:
        return reduce_residual_field_shards_for_chunk(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            manifests=manifests,
            cleanup_policy=cleanup_policy,
            shard_storage_root=self.resolve_shard_storage_root(
                output_dir=output_dir,
                scratch_root=scratch_root,
            ),
            quiet_logs=quiet_logs,
        )

    def cleanup_reclaimable_shards(
        self,
        *,
        output_dir: str,
        chunk_id: int,
        parameter_digest: str,
        db_path: str,
        manifests: list[ResidualFieldShardManifest] | None = None,
        scratch_root: str | None = None,
    ) -> tuple[str, ...]:
        return delete_reclaimable_residual_field_shards(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            db_path=db_path,
            manifests=manifests,
            shard_storage_root=self.resolve_shard_storage_root(
                output_dir=output_dir,
                scratch_root=scratch_root,
            ),
        )


def build_residual_field_reducer_backend(
    kind: ResidualFieldReducerBackendKind,
    *,
    shard_storage_root_override: str | None = None,
) -> ManifestDrivenResidualFieldReducerBackend:
    normalized = _normalize_reducer_backend_kind(kind)
    layout = (
        LOCAL_RESTARTABLE_LAYOUT
        if normalized == "local_restartable"
        else SHARED_DURABLE_LAYOUT
    )
    return ManifestDrivenResidualFieldReducerBackend(
        layout,
        shard_storage_root_override=shard_storage_root_override,
    )


def resolve_residual_field_reducer_backend(
    *,
    workflow_parameters: "WorkflowParameters | object",
    client,
) -> ManifestDrivenResidualFieldReducerBackend:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    shard_storage_root_override = None
    if hasattr(runtime_info, "get"):
        shard_storage_root_override = runtime_info.get("residual_shard_durable_root")
    if shard_storage_root_override is None:
        shard_storage_root_override = os.getenv("MOSAIC_RESIDUAL_SHARD_DURABLE_ROOT")
    return build_residual_field_reducer_backend(
        resolve_residual_field_reducer_backend_kind(
            runtime_info=runtime_info,
            client=client,
        ),
        shard_storage_root_override=shard_storage_root_override,
    )


def resolve_residual_field_reducer_backend_kind(
    *,
    runtime_info,
    client,
) -> ResidualFieldReducerBackendKind:
    override = None
    if hasattr(runtime_info, "get"):
        override = runtime_info.get("residual_field_reducer_backend") or runtime_info.get(
            "reducer_backend"
        )
    if override is None:
        override = os.getenv("MOSAIC_RESIDUAL_FIELD_REDUCER_BACKEND")
    if override is not None:
        return _normalize_reducer_backend_kind(str(override))
    return (
        "local_restartable"
        if is_same_node_local_client(client)
        else "durable_shared_restartable"
    )


def is_same_node_local_client(client) -> bool:
    if client is None or is_sync_client(client):
        return True
    backend = str(os.getenv("DASK_BACKEND", "")).strip().lower()
    if backend in {"local", "cuda-local", "sync", "synchronous", "single-threaded"}:
        return True
    cluster = getattr(client, "cluster", None)
    cluster_name = type(cluster).__name__.lower() if cluster is not None else ""
    return "localcluster" in cluster_name


def _normalize_reducer_backend_kind(value: str) -> ResidualFieldReducerBackendKind:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"local", "local_restartable", "local_restartable_reducer"}:
        return "local_restartable"
    if normalized in {
        "durable",
        "durable_shared",
        "durable_shared_restartable",
        "durable_restartable",
    }:
        return "durable_shared_restartable"
    raise ValueError(
        "Residual-field reducer backend must be 'local_restartable' or "
        "'durable_shared_restartable'."
    )


__all__ = [
    "ManifestDrivenResidualFieldReducerBackend",
    "ResidualFieldReducerBackend",
    "ResidualFieldReducerBackendKind",
    "ResidualFieldCheckpointPolicy",
    "ResidualFieldReducerBackendLayout",
    "ResidualFieldReducerRuntimeState",
    "build_residual_field_reducer_backend",
    "is_same_node_local_client",
    "resolve_residual_field_reducer_backend",
    "resolve_residual_field_reducer_backend_kind",
]
