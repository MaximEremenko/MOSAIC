from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from core.residual_field.accumulation import (
    build_existing_materialized_residual_field_state,
)
from core.residual_field.artifacts import (
    _ResidualFieldChunkStatusUpdater,
    _build_residual_field_reducer_progress_manifest,
    _write_residual_field_chunk_state,
    assess_residual_field_manifest,
    build_residual_field_chunk_manifest,
    build_residual_field_output_artifact_refs,
    delete_reclaimable_residual_field_shards,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    ResidualFieldArtifactStore,
    persist_residual_field_shard_checkpoint,
    reconcile_residual_field_reducer_progress,
    reduce_residual_field_shards_for_chunk,
    write_residual_field_reducer_progress_manifest,
)
from core.residual_field.local_accumulator import (
    LiveLocalAccumulator,
    ResidualFieldLocalAccumulatorPartial,
    build_local_accumulator_snapshot_path,
    load_local_accumulator_snapshot,
    make_local_accumulator_snapshot_key,
    parse_local_accumulator_snapshot_key,
    write_local_accumulator_snapshot,
)
from core.residual_field.contracts import (
    ResidualFieldArtifactManifest,
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
    ResidualFieldWorkUnit,
)
from core.contracts import CompletionStatus
from core.runtime import chunk_mutex
from core.runtime import is_sync_client

if TYPE_CHECKING:
    from core.models import WorkflowParameters


DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES = 256 * 1024 * 1024
_PROCESS_LOCAL_REDUCER_BACKENDS: dict[
    tuple[str, str | None, int],
    "ManifestDrivenResidualFieldReducerBackend",
] = {}


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
    durable_truth_unit: str
    live_state_storage_role: str
    durable_checkpoint_storage_role: str
    final_artifact_storage_role: str
    uncommitted_restart_rule: str


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
        local_accumulator_max_ram_bytes: int = DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES,
    ) -> None:
        self.layout = layout
        self.shard_storage_root_override = (
            str(Path(shard_storage_root_override).expanduser())
            if shard_storage_root_override
            else None
        )
        self.local_accumulator_max_ram_bytes = int(local_accumulator_max_ram_bytes)
        self._local_accumulators: dict[tuple[int, str], LiveLocalAccumulator] = {}

    def uses_local_chunk_accumulator(self) -> bool:
        return self.layout.kind == "local_restartable"

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
        return ResidualFieldLocalAccumulatorPartial(
            work_unit=work_unit,
            point_ids=point_ids,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
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
            durable_truth_unit=(
                "committed_local_snapshot_generation"
                if self.layout.kind == "local_restartable"
                else "committed_shard_checkpoint"
            ),
            live_state_storage_role=(
                "owner-local-live-accumulator"
                if self.layout.kind == "local_restartable"
                else "owner-local-scratch-staging"
            ),
            durable_checkpoint_storage_role=(
                "durable-local-snapshot-generation"
                if self.layout.kind == "local_restartable"
                else "durable-shared-shard-checkpoint"
            ),
            final_artifact_storage_role="durable-final-chunk-artifact",
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

    def _local_accumulator_key(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        partition_id: int | None,
    ) -> tuple[int, str, int | None]:
        return int(chunk_id), str(parameter_digest), (
            int(partition_id) if partition_id is not None else None
        )

    def _load_latest_local_snapshot(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        partition_id: int | None,
    ) -> tuple[int, dict[str, object]] | None:
        progress = self.load_progress_manifest(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        )
        snapshot_seq = 0
        if progress is not None:
            for key in progress.incorporated_shard_keys:
                parsed = parse_local_accumulator_snapshot_key(key)
                if parsed is None:
                    continue
                (
                    parsed_chunk_id,
                    parsed_digest,
                    parsed_partition_id,
                    parsed_seq,
                ) = parsed
                if (
                    parsed_chunk_id == int(chunk_id)
                    and parsed_digest == str(parameter_digest)
                    and parsed_partition_id
                    == (int(partition_id) if partition_id is not None else None)
                ):
                    snapshot_seq = max(snapshot_seq, int(parsed_seq))
        if snapshot_seq <= 0:
            return None
        snapshot = load_local_accumulator_snapshot(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            snapshot_seq=snapshot_seq,
        )
        if snapshot is None:
            return None
        return int(snapshot_seq), snapshot

    def _latest_local_snapshot_refs(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
    ) -> list[tuple[int | None, int]]:
        progress = self.load_progress_manifest(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        )
        if progress is None:
            return []
        latest: dict[int | None, int] = {}
        for key in progress.incorporated_shard_keys:
            parsed = parse_local_accumulator_snapshot_key(key)
            if parsed is None:
                continue
            parsed_chunk_id, parsed_digest, parsed_partition_id, parsed_seq = parsed
            if (
                parsed_chunk_id == int(chunk_id)
                and parsed_digest == str(parameter_digest)
            ):
                latest[parsed_partition_id] = max(
                    int(parsed_seq),
                    int(latest.get(parsed_partition_id, 0)),
                )
        return sorted(
            latest.items(),
            key=lambda item: (-1 if item[0] is None else int(item[0])),
        )

    def _restore_local_accumulator(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        scratch_root: str,
        partition_id: int | None,
    ) -> LiveLocalAccumulator | None:
        snapshot_state = self._load_latest_local_snapshot(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            partition_id=partition_id,
        )
        if snapshot_state is None:
            return None
        snapshot_seq, snapshot = snapshot_state
        return LiveLocalAccumulator.from_snapshot(
            snapshot,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            snapshot_seq=snapshot_seq,
            scratch_root=scratch_root,
            max_ram_bytes=self.local_accumulator_max_ram_bytes,
        )

    def _get_or_create_local_accumulator(
        self,
        partial: ResidualFieldLocalAccumulatorPartial,
        *,
        output_dir: str,
        scratch_root: str,
    ) -> LiveLocalAccumulator:
        key = self._local_accumulator_key(
            chunk_id=partial.chunk_id,
            parameter_digest=partial.parameter_digest,
            partition_id=partial.work_unit.partition_id,
        )
        accumulator = self._local_accumulators.get(key)
        if accumulator is not None:
            return accumulator
        accumulator = self._restore_local_accumulator(
            chunk_id=partial.chunk_id,
            parameter_digest=partial.parameter_digest,
            output_dir=output_dir,
            scratch_root=scratch_root,
            partition_id=partial.work_unit.partition_id,
        )
        if accumulator is None:
            accumulator = LiveLocalAccumulator.from_partial(
                partial,
                scratch_root=scratch_root,
                max_ram_bytes=self.local_accumulator_max_ram_bytes,
            )
        self._local_accumulators[key] = accumulator
        return accumulator

    def local_intervals_already_durable(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        output_dir: str,
    ) -> bool:
        snapshot_state = self._load_latest_local_snapshot(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            output_dir=output_dir,
            partition_id=work_unit.partition_id,
        )
        if snapshot_state is None:
            return False
        _, snapshot = snapshot_state
        durable_intervals = set(
            int(interval_id) for interval_id in snapshot["incorporated_interval_ids"]
        )
        return set(int(interval_id) for interval_id in work_unit.interval_ids).issubset(
            durable_intervals
        )

    def _snapshot_local_accumulator(
        self,
        accumulator: LiveLocalAccumulator,
        *,
        output_dir: str,
        db_path: str,
        cleanup_policy: str,
    ) -> None:
        snapshot_seq = accumulator.next_snapshot_seq()
        snapshot_payload = accumulator.snapshot_payload()
        write_local_accumulator_snapshot(
            output_dir,
            chunk_id=accumulator.chunk_id,
            parameter_digest=accumulator.parameter_digest,
            partition_id=accumulator.partition_id,
            snapshot_seq=snapshot_seq,
            point_ids=snapshot_payload["point_ids"],
            grid_shape_nd=snapshot_payload["grid_shape_nd"],
            amplitudes_delta=snapshot_payload["amplitudes_delta"],
            amplitudes_average=snapshot_payload["amplitudes_average"],
            reciprocal_point_count=int(snapshot_payload["reciprocal_point_count"]),
            total_reciprocal_points=int(snapshot_payload["total_reciprocal_points"]),
            incorporated_interval_ids=snapshot_payload["incorporated_interval_ids"],
            storage_mode=str(snapshot_payload["storage_mode"]),
        )
        with chunk_mutex(accumulator.chunk_id):
            existing_progress = self.load_progress_manifest(
                output_dir=output_dir,
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
            )
            new_snapshot_key = make_local_accumulator_snapshot_key(
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
                partition_id=accumulator.partition_id,
                snapshot_seq=snapshot_seq,
            )
            retained_snapshot_keys: list[str] = []
            prior_interval_ids: tuple[int, ...] = ()
            if existing_progress is not None:
                prior_interval_ids = existing_progress.incorporated_interval_ids
                for key in existing_progress.incorporated_shard_keys:
                    parsed = parse_local_accumulator_snapshot_key(key)
                    if parsed is None:
                        retained_snapshot_keys.append(key)
                        continue
                    (
                        parsed_chunk_id,
                        parsed_digest,
                        parsed_partition_id,
                        _,
                    ) = parsed
                    if (
                        parsed_chunk_id == int(accumulator.chunk_id)
                        and parsed_digest == str(accumulator.parameter_digest)
                        and parsed_partition_id
                        == (
                            int(accumulator.partition_id)
                            if accumulator.partition_id is not None
                            else None
                        )
                    ):
                        continue
                    retained_snapshot_keys.append(key)
            progress_manifest = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
                completion_status=CompletionStatus.MATERIALIZED,
                durable_truth_unit="committed_local_snapshot_generation",
                incorporated_shard_keys=tuple(
                    sorted(tuple(retained_snapshot_keys) + (new_snapshot_key,))
                ),
                incorporated_interval_ids=tuple(
                    sorted(
                        set(int(v) for v in prior_interval_ids).union(
                            int(v) for v in snapshot_payload["incorporated_interval_ids"]
                        )
                    )
                ),
                reclaimable_shard_keys=(),
                final_artifacts=build_residual_field_output_artifact_refs(
                    output_dir,
                    accumulator.chunk_id,
                ),
                pending_shard_keys=(),
                pending_interval_ids=(),
                cleanup_policy=cleanup_policy,
            )
            self.write_progress_manifest(progress_manifest)
        newly_durable = accumulator.mark_snapshot_committed(snapshot_seq)
        if accumulator.partition_id is None:
            status_updater = _ResidualFieldChunkStatusUpdater(db_path)
            for interval_id in newly_durable:
                status_updater.mark_saved(int(interval_id), int(accumulator.chunk_id))
        previous_snapshot_seq = snapshot_seq - 1
        if previous_snapshot_seq > 0:
            previous_snapshot_path = build_local_accumulator_snapshot_path(
                output_dir,
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
                partition_id=accumulator.partition_id,
                snapshot_seq=previous_snapshot_seq,
            )
            previous_snapshot_path.unlink(missing_ok=True)

    def _build_local_materialized_state(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        snapshot_payload: dict[str, object],
    ):
        return build_existing_materialized_residual_field_state(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_artifacts=build_residual_field_output_artifact_refs(output_dir, chunk_id),
            amplitudes_payload=np.column_stack(
                (
                    np.asarray(snapshot_payload["point_ids"], dtype=np.int64).astype(np.complex128),
                    np.asarray(snapshot_payload["amplitudes_delta"], dtype=np.complex128),
                )
            ),
            amplitudes_average_payload=np.column_stack(
                (
                    np.asarray(snapshot_payload["point_ids"], dtype=np.int64).astype(np.complex128),
                    np.asarray(snapshot_payload["amplitudes_average"], dtype=np.complex128),
                )
            ),
            grid_shape_nd=np.asarray(snapshot_payload["grid_shape_nd"], dtype=np.int64),
            reciprocal_point_count=int(snapshot_payload["reciprocal_point_count"]),
            applied_interval_ids=tuple(
                int(interval_id) for interval_id in snapshot_payload["incorporated_interval_ids"]
            ),
        )

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
        if not self.uses_local_chunk_accumulator():
            raise ValueError("accept_partial is only valid for local_restartable mode.")
        accumulator = self._get_or_create_local_accumulator(
            partial,
            output_dir=output_dir,
            scratch_root=scratch_root,
        )
        before = tuple(sorted(accumulator.current_interval_ids))
        accumulator.accept_partial(partial)
        after = tuple(sorted(accumulator.current_interval_ids))
        if before == after:
            return
        snapshot_every = max(int(total_expected_partials) // 4, 1)
        if accumulator.accepted_since_snapshot >= snapshot_every:
            self._snapshot_local_accumulator(
                accumulator,
                output_dir=output_dir,
                db_path=db_path,
                cleanup_policy=cleanup_policy,
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
        if self.uses_local_chunk_accumulator():
            if scratch_root is None:
                raise ValueError("Local accumulator finalization requires scratch_root.")
            matching_keys = [
                key
                for key in list(self._local_accumulators)
                if key[0] == int(chunk_id) and key[1] == str(parameter_digest)
            ]
            for key in matching_keys:
                accumulator = self._local_accumulators.get(key)
                if accumulator is None:
                    continue
                if accumulator.current_interval_ids != accumulator.durable_interval_ids:
                    self._snapshot_local_accumulator(
                        accumulator,
                        output_dir=output_dir,
                        db_path=db_path,
                        cleanup_policy=str(cleanup_policy or "off"),
                    )
            snapshot_refs = self._latest_local_snapshot_refs(
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                output_dir=output_dir,
            )
            if not snapshot_refs:
                accumulator = self._restore_local_accumulator(
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    output_dir=output_dir,
                    scratch_root=scratch_root,
                    partition_id=None,
                )
                if accumulator is None:
                    return None
                snapshot_refs = [(None, int(accumulator.durable_snapshot_seq))]
            snapshots: list[dict[str, object]] = []
            for partition_id, snapshot_seq in snapshot_refs:
                snapshot = load_local_accumulator_snapshot(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                )
                if snapshot is not None:
                    snapshots.append(snapshot)
            if not snapshots:
                return None
            partitioned = any(
                snapshot.get("partition_id") not in (None, -1)
                for snapshot in snapshots
            )
            if partitioned:
                point_offset = 0
                point_ids_blocks: list[np.ndarray] = []
                delta_blocks: list[np.ndarray] = []
                average_blocks: list[np.ndarray] = []
                grid_shape_blocks: list[np.ndarray] = []
                applied_set: set[int] = set()
                total_reciprocal_points = int(snapshots[0]["total_reciprocal_points"])
                reciprocal_point_count = 0
                for snapshot in snapshots:
                    delta_block = np.asarray(
                        snapshot["amplitudes_delta"], dtype=np.complex128
                    ).reshape(-1)
                    average_block = np.asarray(
                        snapshot["amplitudes_average"], dtype=np.complex128
                    ).reshape(-1)
                    block_size = int(delta_block.shape[0])
                    point_ids_blocks.append(
                        np.arange(point_offset, point_offset + block_size, dtype=np.int64)
                    )
                    point_offset += block_size
                    delta_blocks.append(delta_block)
                    average_blocks.append(average_block)
                    grid_shape_blocks.append(
                        np.asarray(snapshot["grid_shape_nd"], dtype=np.int64)
                    )
                    reciprocal_point_count += int(snapshot["reciprocal_point_count"])
                    applied_set.update(
                        int(interval_id)
                        for interval_id in snapshot["incorporated_interval_ids"]
                    )
                snapshot_payload = {
                    "point_ids": np.concatenate(point_ids_blocks) if point_ids_blocks else np.array([], dtype=np.int64),
                    "grid_shape_nd": np.vstack(grid_shape_blocks) if grid_shape_blocks else np.array([], dtype=np.int64),
                    "amplitudes_delta": np.concatenate(delta_blocks) if delta_blocks else np.array([], dtype=np.complex128),
                    "amplitudes_average": np.concatenate(average_blocks) if average_blocks else np.array([], dtype=np.complex128),
                    "reciprocal_point_count": int(reciprocal_point_count),
                    "total_reciprocal_points": int(total_reciprocal_points),
                    "incorporated_interval_ids": tuple(sorted(applied_set)),
                }
            else:
                snapshot_payload = snapshots[0]
                applied_set = set(
                    int(interval_id)
                    for interval_id in snapshot_payload["incorporated_interval_ids"]
                )
            with chunk_mutex(chunk_id):
                store = ResidualFieldArtifactStore(output_dir)
                _write_residual_field_chunk_state(
                    store=store,
                    chunk_id=chunk_id,
                    merged_state=self._build_local_materialized_state(
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        output_dir=output_dir,
                        snapshot_payload=snapshot_payload,
                    ),
                    total_reciprocal_points=int(snapshot_payload["total_reciprocal_points"]),
                    applied_set=(
                        applied_set
                        if partitioned
                        else set(
                            int(interval_id)
                            for interval_id in snapshot_payload["incorporated_interval_ids"]
                        )
                    ),
                )
            representative_interval_id = max(
                int(v) for v in snapshot_payload["incorporated_interval_ids"]
            )
            work_unit = ResidualFieldWorkUnit.interval_chunk(
                interval_id=representative_interval_id,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                output_dir=output_dir,
            )
            manifest = build_residual_field_chunk_manifest(
                work_unit,
                output_dir=output_dir,
                completion_status=CompletionStatus.COMMITTED,
            )
            manifest_assessment = assess_residual_field_manifest(
                manifest,
                db_path=db_path,
            )
            progress_manifest = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                completion_status=(
                    CompletionStatus.COMMITTED
                    if manifest_assessment.is_complete
                    else CompletionStatus.MATERIALIZED
                ),
                durable_truth_unit="committed_local_snapshot_generation",
                incorporated_shard_keys=(),
                incorporated_interval_ids=tuple(
                    sorted(
                        int(v)
                        for v in snapshot_payload["incorporated_interval_ids"]
                    )
                ),
                reclaimable_shard_keys=(),
                final_artifacts=manifest.artifacts,
                pending_shard_keys=(),
                pending_interval_ids=(),
                cleanup_policy=str(cleanup_policy or "off"),
            )
            self.write_progress_manifest(progress_manifest)
            status_updater = _ResidualFieldChunkStatusUpdater(db_path)
            for interval_id in sorted(
                int(v) for v in snapshot_payload["incorporated_interval_ids"]
            ):
                status_updater.mark_saved(int(interval_id), int(chunk_id))
            for partition_id, snapshot_seq in snapshot_refs:
                build_local_accumulator_snapshot_path(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                ).unlink(missing_ok=True)
            for key in matching_keys:
                accumulator = self._local_accumulators.pop(key, None)
                if accumulator is not None:
                    accumulator.cleanup_live_files()
            return manifest
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
        if self.uses_local_chunk_accumulator():
            return ()
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
    local_accumulator_max_ram_bytes: int = DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES,
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
        local_accumulator_max_ram_bytes=local_accumulator_max_ram_bytes,
    )


def get_process_local_residual_field_backend(
    template_backend: ManifestDrivenResidualFieldReducerBackend,
) -> ManifestDrivenResidualFieldReducerBackend:
    key = (
        str(template_backend.layout.kind),
        template_backend.shard_storage_root_override,
        int(template_backend.local_accumulator_max_ram_bytes),
    )
    backend = _PROCESS_LOCAL_REDUCER_BACKENDS.get(key)
    if backend is None:
        backend = build_residual_field_reducer_backend(
            template_backend.layout.kind,
            shard_storage_root_override=template_backend.shard_storage_root_override,
            local_accumulator_max_ram_bytes=template_backend.local_accumulator_max_ram_bytes,
        )
        _PROCESS_LOCAL_REDUCER_BACKENDS[key] = backend
    return backend


def finalize_process_local_residual_chunk(
    template_backend: ManifestDrivenResidualFieldReducerBackend,
    *,
    chunk_id: int,
    parameter_digest: str,
    output_dir: str,
    db_path: str,
    cleanup_policy: str | bool | None = None,
    scratch_root: str | None = None,
    quiet_logs: bool = False,
) -> ResidualFieldArtifactManifest | None:
    backend = get_process_local_residual_field_backend(template_backend)
    return backend.finalize_chunk(
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        output_dir=output_dir,
        db_path=db_path,
        cleanup_policy=cleanup_policy,
        scratch_root=scratch_root,
        quiet_logs=quiet_logs,
    )


def resolve_residual_field_reducer_backend(
    *,
    workflow_parameters: "WorkflowParameters | object",
    client,
) -> ManifestDrivenResidualFieldReducerBackend:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    shard_storage_root_override = None
    local_accumulator_max_ram_bytes = DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES
    if hasattr(runtime_info, "get"):
        shard_storage_root_override = runtime_info.get("residual_shard_durable_root")
        local_accumulator_max_ram_bytes = int(
            runtime_info.get(
                "residual_local_accumulator_max_ram_bytes",
                local_accumulator_max_ram_bytes,
            )
        )
    if shard_storage_root_override is None:
        shard_storage_root_override = os.getenv("MOSAIC_RESIDUAL_SHARD_DURABLE_ROOT")
    local_accumulator_max_ram_bytes = int(
        os.getenv(
            "MOSAIC_LOCAL_REDUCER_MAX_RAM_BYTES",
            str(local_accumulator_max_ram_bytes),
        )
    )
    return build_residual_field_reducer_backend(
        resolve_residual_field_reducer_backend_kind(
            runtime_info=runtime_info,
            client=client,
        ),
        shard_storage_root_override=shard_storage_root_override,
        local_accumulator_max_ram_bytes=local_accumulator_max_ram_bytes,
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
    "finalize_process_local_residual_chunk",
    "get_process_local_residual_field_backend",
    "ManifestDrivenResidualFieldReducerBackend",
    "ResidualFieldReducerBackend",
    "ResidualFieldReducerBackendKind",
    "ResidualFieldCheckpointPolicy",
    "ResidualFieldLocalAccumulatorPartial",
    "ResidualFieldReducerBackendLayout",
    "ResidualFieldReducerRuntimeState",
    "build_residual_field_reducer_backend",
    "is_same_node_local_client",
    "resolve_residual_field_reducer_backend",
    "resolve_residual_field_reducer_backend_kind",
]
