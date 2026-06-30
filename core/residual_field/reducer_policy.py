"""
core/residual_field/reducer_policy.py

Pure policy types, layout dataclasses, layout constants, and the
ResidualFieldReducerBackend Protocol — all stateless and pickle-free.

These are separated from backend.py purely for size/navigability; the
public API of core.residual_field.backend is unchanged (every name is
re-exported from there).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from core.residual_field.local_accumulator import ResidualFieldLocalAccumulatorPartial
from core.residual_field.contracts import (
    ResidualFieldArtifactManifest,
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Literal type aliases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Policy / layout dataclasses
# ---------------------------------------------------------------------------

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
    durable_truth_unit: str
    live_state_storage_role: str
    durable_checkpoint_storage_role: str
    final_artifact_storage_role: str
    uncommitted_restart_rule: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class ResidualFieldReducerBackend(Protocol):
    layout: ResidualFieldReducerBackendLayout

    def uses_local_chunk_accumulator(self) -> bool:
        ...

    def build_local_partial(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        contribution_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
        point_ids: np.ndarray,
    ) -> ResidualFieldLocalAccumulatorPartial:
        ...

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

    def accept_partial(
        self,
        partial: ResidualFieldLocalAccumulatorPartial,
        *,
        output_dir: str,
        scratch_root: str,
        db_path: str,
        total_expected_partials: int,
        cleanup_policy: str = "off",
    ) -> None:
        ...

    def accept_local_contribution(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        contribution_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
        point_ids: np.ndarray,
        output_dir: str,
        scratch_root: str,
        db_path: str,
        total_expected_partials: int,
        cleanup_policy: str = "off",
    ) -> None:
        ...

    def inspect_local_reducer_target(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        partition_id: int | None = None,
    ) -> dict[str, object] | None:
        ...

    def flush_local_reducer_target(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        partition_id: int | None,
        output_dir: str,
        db_path: str,
        cleanup_policy: str = "off",
    ) -> bool:
        ...


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

LOCAL_RESTARTABLE_LAYOUT = ResidualFieldReducerBackendLayout(
    kind="local_restartable",
    reducer_ownership="single-writer chunk-owned reducer",
    reducer_backing_store="reducer-owned local accumulator with periodic durable snapshots",
    durability_policy=(
        "restart from committed local accumulator snapshots, reducer progress, and final artifacts; "
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
        "live local accumulator files when file-backed",
        "temporary local transport and staging state before the next snapshot",
    ),
    durable_state=(
        "durable local accumulator snapshots",
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
    reducer_backing_store="owner-local accumulator with immutable shared-storage generations",
    durability_policy=(
        "restart from committed accumulator generations, reducer progress, and final artifacts "
        "visible to workers/jobs; uncommitted task-local work may be recomputed"
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


__all__ = [
    "LOCAL_RESTARTABLE_LAYOUT",
    "ResidualFieldCheckpointPolicy",
    "ResidualFieldReducerBackend",
    "ResidualFieldReducerBackendKind",
    "ResidualFieldReducerBackendLayout",
    "ResidualFieldReducerRuntimeState",
    "ResidualShardCheckpointPolicy",
    "ScatteringIntervalArtifactPolicy",
    "SHARED_DURABLE_LAYOUT",
    "ScratchRolePolicy",
]
