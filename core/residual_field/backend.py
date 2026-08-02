from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.residual_field.artifacts import (
    _ResidualFieldChunkStatusUpdater,
    _build_residual_field_reducer_progress_manifest,
    _write_residual_field_chunk_payload_components,
    build_residual_field_chunk_manifest,
    build_residual_field_output_artifact_refs,
    delete_reclaimable_residual_field_shards,
    discover_residual_field_reducer_progress_manifest,
    discover_residual_field_shard_manifests,
    ResidualFieldArtifactStore,
    persist_residual_field_shard_checkpoint,
    reconcile_residual_field_reducer_progress,
    write_residual_field_reducer_progress_manifest,
    _normalize_residual_shard_cleanup_policy,
)
from core.residual_field.local_accumulator import (
    LiveLocalAccumulator,
    ResidualFieldLocalAccumulatorPartial,
    SnapshotCloneUnsupported,
    build_local_accumulator_snapshot_path,
    load_local_accumulator_snapshot,
    load_local_accumulator_snapshot_metadata,
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

# Re-exports from extracted sibling modules — keep all existing imports working.
from core.residual_field.reducer_helpers import (  # noqa: F401
    _allocate_finalize_output,
    _finalize_scratch_dir,
    _manifest_final_artifacts_present,
    _mark_residual_intervals_saved,
    _normalize_reducer_backend_kind,
    checkpoint_cadence as _checkpoint_cadence_policy,
    live_trim_cadence as _live_trim_cadence_policy,
    DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES,
    is_same_node_local_client,
)
from core.residual_field.snapshot_writer import (  # noqa: F401
    _LocalSnapshotWriter,
    _PendingLocalSnapshot,
)
# Re-exported for callers that import the assembly helpers from here.
from core.residual_field.assembly import (  # noqa: F401
    assemble_interval_partitioned_chunk_payload,
    assemble_local_snapshot_chunk_payload,
    validate_interval_partitioned_snapshot_family,
    validate_local_partition_snapshot_family,
    _require_expected_interval_coverage,
    _require_expected_partition_family,
)
from core.residual_field.reducer_policy import (  # noqa: F401
    LOCAL_RESTARTABLE_LAYOUT,
    ResidualFieldCheckpointPolicy,
    ResidualFieldReducerBackend,
    ResidualFieldReducerBackendKind,
    ResidualFieldReducerBackendLayout,
    ResidualFieldReducerRuntimeState,
    ResidualShardCheckpointPolicy,
    ScatteringIntervalArtifactPolicy,
    ScratchRolePolicy,
)

if TYPE_CHECKING:
    from core.models import WorkflowParameters


logger = logging.getLogger(__name__)


_PROCESS_LOCAL_REDUCER_BACKENDS: dict[
    tuple[str, str | None, int],
    "ManifestDrivenResidualFieldReducerBackend",
] = {}




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
        self._local_accumulators: dict[
            tuple[int, str, int | None], LiveLocalAccumulator
        ] = {}
        self._local_accumulator_locks: dict[
            tuple[int, str, int | None], threading.RLock
        ] = {}
        self._local_accumulator_locks_guard = threading.Lock()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Live accumulators, locks and the snapshot writer are process-local
        # runtime state. Dask serializes this backend into task graphs, so
        # only configuration can cross the scheduler boundary.
        state["_local_accumulators"] = {}
        state["_local_accumulator_locks"] = {}
        state.pop("_local_accumulator_locks_guard", None)
        state.pop("_snapshot_writer_obj", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._local_accumulators = {}
        self._local_accumulator_locks = {}
        self._local_accumulator_locks_guard = threading.Lock()
        self._snapshot_writer_obj = None

    def _snapshot_writer(self) -> _LocalSnapshotWriter:
        writer = getattr(self, "_snapshot_writer_obj", None)
        if writer is None:
            with self._local_accumulator_locks_guard:
                writer = getattr(self, "_snapshot_writer_obj", None)
                if writer is None:
                    writer = _LocalSnapshotWriter()
                    self._snapshot_writer_obj = writer
        return writer

    def _async_snapshot_writes_enabled(self) -> bool:
        """Async durable snapshot writes (default on for local_restartable).

        MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES=0 restores the fully
        synchronous write-under-fold-lock behavior for A/B verification."""
        if self.layout.kind != "local_restartable":
            return False
        return os.getenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "1").strip() != "0"

    def _repair_progress_final_artifacts(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        db_path: str,
        cleanup_policy: str | bool | None,
        scratch_root: str | None = None,
        mark_intervals_saved: bool = True,
    ) -> ResidualFieldArtifactManifest | None:
        progress = self.load_progress_manifest(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        )
        if progress is None:
            return None
        if progress.pending_shard_keys or progress.pending_interval_ids:
            return None
        incorporated_interval_ids = tuple(
            sorted(int(value) for value in progress.incorporated_interval_ids)
        )
        if not incorporated_interval_ids:
            return None

        representative_interval_id = max(incorporated_interval_ids)
        manifest = build_residual_field_chunk_manifest(
            ResidualFieldWorkUnit.interval_chunk(
                interval_id=representative_interval_id,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                output_dir=output_dir,
            ),
            output_dir=output_dir,
            completion_status=CompletionStatus.COMMITTED,
        )
        if not _manifest_final_artifacts_present(manifest):
            return None
        applied_interval_ids = ResidualFieldArtifactStore(output_dir).load_applied_interval_ids(
            chunk_id
        )
        if not set(incorporated_interval_ids).issubset(applied_interval_ids):
            return None

        resolved_cleanup_policy = _normalize_residual_shard_cleanup_policy(
            cleanup_policy
            if cleanup_policy is not None
            else progress.cleanup_policy
        )
        reclaimable_shard_keys = (
            progress.reclaimable_shard_keys
            if progress.reclaimable_shard_keys
            else progress.incorporated_shard_keys
        )
        needs_progress_rewrite = (
            progress.completion_status is not CompletionStatus.COMMITTED
            or progress.cleanup_policy != resolved_cleanup_policy
            or (
                resolved_cleanup_policy == "delete_reclaimable"
                and bool(progress.incorporated_shard_keys)
                and not progress.reclaimable_shard_keys
            )
        )
        if needs_progress_rewrite:
            committed_progress = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                completion_status=CompletionStatus.COMMITTED,
                durable_truth_unit=progress.durable_truth_unit,
                incorporated_shard_keys=progress.incorporated_shard_keys,
                incorporated_interval_ids=incorporated_interval_ids,
                reclaimable_shard_keys=reclaimable_shard_keys,
                final_artifacts=manifest.artifacts,
                pending_shard_keys=(),
                pending_interval_ids=(),
                cleanup_policy=resolved_cleanup_policy,
            )
            self.write_progress_manifest(committed_progress)

        if mark_intervals_saved:
            _mark_residual_intervals_saved(
                db_path=db_path,
                chunk_id=chunk_id,
                interval_ids=incorporated_interval_ids,
            )
        if resolved_cleanup_policy == "delete_reclaimable":
            self.cleanup_reclaimable_shards(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                db_path=db_path,
                scratch_root=scratch_root,
            )
        return manifest

    def uses_local_chunk_accumulator(self) -> bool:
        return self.layout.kind == "local_restartable"

    def uses_owner_local_accumulator(self) -> bool:
        # local_restartable is the only layout since durable_shared_restartable
        # was retired; every backend keeps an owner-local accumulator.
        return self.uses_local_chunk_accumulator()

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
            durable_truth_unit="committed_local_snapshot_generation",
            live_state_storage_role="owner-local-live-accumulator",
            durable_checkpoint_storage_role="durable-local-snapshot-generation",
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
        root = scratch_root or str(Path(output_dir) / ".local_restartable")
        return str(Path(root).expanduser())

    def uses_direct_interval_handoff(self) -> bool:
        return bool(self.layout.direct_interval_handoff_supported)

    def persist_interval_artifacts_by_default(self) -> bool:
        return bool(self.layout.persist_interval_artifacts_by_default)

    def interval_artifacts_required_for_transport(self) -> bool:
        return self.layout.checkpoint_policy.interval_artifacts == "required_transport"

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

    def _release_local_accumulator(
        self,
        key: tuple[int, str, int | None],
    ) -> bool:
        accumulator = self._local_accumulators.pop(key, None)
        if accumulator is None:
            return False
        try:
            accumulator.cleanup_live_files()
        except Exception:
            pass
        return True

    def _local_accumulator_lock(
        self,
        key: tuple[int, str, int | None],
    ) -> threading.RLock:
        with self._local_accumulator_locks_guard:
            lock = self._local_accumulator_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._local_accumulator_locks[key] = lock
            return lock

    def _load_latest_local_snapshot(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        partition_id: int | None,
        include_payload: bool = True,
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
        snapshot_loader = (
            load_local_accumulator_snapshot
            if include_payload
            else load_local_accumulator_snapshot_metadata
        )
        snapshot = snapshot_loader(
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
            include_payload=True,
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
        return self._get_or_create_local_accumulator_for_target(
            work_unit=partial.work_unit,
            point_ids=partial.point_ids,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            amplitudes_delta=partial.amplitudes_delta,
            amplitudes_average=partial.amplitudes_average,
            output_dir=output_dir,
            scratch_root=scratch_root,
        )

    def _get_or_create_local_accumulator_for_target(
        self,
        *,
        work_unit: ResidualFieldWorkUnit,
        point_ids: np.ndarray,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
        output_dir: str,
        scratch_root: str,
    ) -> LiveLocalAccumulator:
        key = self._local_accumulator_key(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            partition_id=work_unit.partition_id,
        )
        accumulator = self._local_accumulators.get(key)
        if accumulator is not None:
            return accumulator
        accumulator = self._restore_local_accumulator(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            output_dir=output_dir,
            scratch_root=scratch_root,
            partition_id=work_unit.partition_id,
        )
        if accumulator is not None:
            current_start = getattr(work_unit, "point_start", None)
            current_stop = getattr(work_unit, "point_stop", None)
            layout_changed = (
                (accumulator.point_start is not None or accumulator.point_stop is not None)
                and (accumulator.point_start != current_start or accumulator.point_stop != current_stop)
            ) or (
                accumulator.point_start is None
                and accumulator.point_stop is None
                and not np.array_equal(
                    accumulator.point_ids,
                    np.asarray(point_ids, dtype=np.int64).reshape(-1),
                )
            )
            if layout_changed:
                logger.warning(
                    "Partition layout changed for chunk=%d partition=%s: "
                    "checkpoint atoms=[%s:%s], current atoms=[%s:%s] "
                    "(checkpoint had %d intervals). Discarding checkpoint.",
                    int(work_unit.chunk_id),
                    work_unit.partition_id,
                    accumulator.point_start, accumulator.point_stop,
                    current_start, current_stop,
                    len(accumulator.current_interval_ids),
                )
                accumulator.cleanup_live_files()
                accumulator = None
        if accumulator is None:
            accumulator = LiveLocalAccumulator.from_arrays(
                work_unit,
                point_ids=point_ids,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
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
            include_payload=False,
        )
        if snapshot_state is None:
            return False
        _, snapshot = snapshot_state
        snap_start = snapshot.get("point_start")
        snap_stop = snapshot.get("point_stop")
        current_start = getattr(work_unit, "point_start", None)
        current_stop = getattr(work_unit, "point_stop", None)
        if snap_start is not None or snap_stop is not None:
            if snap_start != current_start or snap_stop != current_stop:
                return False
        durable_intervals = set(
            int(interval_id) for interval_id in snapshot["incorporated_interval_ids"]
        )
        expected_interval_ids = tuple(
            int(interval_id)
            for interval_id in (
                work_unit.interval_ids
                or (
                    (work_unit.interval_id,)
                    if work_unit.interval_id is not None
                    else ()
                )
            )
        )
        if not expected_interval_ids:
            return False
        return set(expected_interval_ids).issubset(durable_intervals)

    def invalidate_incompatible_local_checkpoints(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        expected_targets: dict[int | None, dict[str, object]],
    ) -> int:
        """Drop local checkpoint snapshots that do not fit the current plan.

        After a crash + resume across a code or configuration change, the
        partition layout or the interval grouping of a chunk can differ from
        what wrote the on-disk snapshots. Snapshots for partition ids absent
        from the current plan, whose atom ranges moved, or whose durable
        interval set is not a union of the current plan's interval batches
        (the durable-skip filter and the accumulator both reason in whole
        batches) would otherwise corrupt the resume: stale layouts get
        concatenated at finalize, and regrouped batches either double-count
        or strand durable intervals no work unit re-covers. Called once per
        (chunk, digest) at plan time with ``expected_targets`` mapping each
        planned partition id to ``{"point_start", "point_stop",
        "interval_batches"}``. Returns the number of snapshots invalidated.
        """
        if not self.uses_local_chunk_accumulator():
            return 0
        with chunk_mutex(chunk_id, lock_root=output_dir):
            progress = self.load_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
            )
            if progress is None or progress.completion_status is CompletionStatus.COMMITTED:
                return 0
            retained_keys: list[str] = []
            retained_interval_ids: set[int] = set()
            dropped: list[tuple[int | None, int]] = []
            for key in progress.incorporated_shard_keys:
                parsed = parse_local_accumulator_snapshot_key(key)
                if parsed is None:
                    retained_keys.append(key)
                    continue
                parsed_chunk_id, parsed_digest, partition_id, snapshot_seq = parsed
                if parsed_chunk_id != int(chunk_id) or parsed_digest != str(parameter_digest):
                    retained_keys.append(key)
                    continue
                stale = partition_id not in expected_targets
                metadata = None
                if not stale:
                    metadata = load_local_accumulator_snapshot_metadata(
                        output_dir,
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        partition_id=partition_id,
                        snapshot_seq=snapshot_seq,
                    )
                    if metadata is None:
                        stale = True
                    else:
                        expected = expected_targets[partition_id]
                        snap_start = metadata.get("point_start")
                        snap_stop = metadata.get("point_stop")
                        expected_start = expected.get("point_start")
                        expected_stop = expected.get("point_stop")
                        expected_axis = expected.get("partition_axis")
                        if (
                            expected_axis is not None
                            and str(metadata.get("accumulator_axis", "points"))
                            != str(expected_axis)
                        ):
                            # A points-axis snapshot in an intervals-axis plan
                            # (or vice versa) describes a different chunk
                            # decomposition; reconciling it would either
                            # double-count or truncate at merge.
                            stale = True
                        elif (snap_start is not None or snap_stop is not None) and (
                            snap_start != expected_start or snap_stop != expected_stop
                        ):
                            stale = True
                        elif expected.get("interval_batches") is not None:
                            durable = set(
                                int(v) for v in metadata["incorporated_interval_ids"]
                            )
                            covered: set[int] = set()
                            for batch in expected["interval_batches"]:
                                if set(batch) <= durable:
                                    covered.update(int(v) for v in batch)
                            if covered != durable:
                                stale = True
                if stale:
                    dropped.append((partition_id, int(snapshot_seq)))
                    continue
                retained_keys.append(key)
                retained_interval_ids.update(
                    int(v) for v in metadata["incorporated_interval_ids"]
                )
            if not dropped:
                return 0
            for partition_id, snapshot_seq in dropped:
                build_local_accumulator_snapshot_path(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                ).unlink(missing_ok=True)
                self._release_local_accumulator(
                    self._local_accumulator_key(
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        partition_id=partition_id,
                    )
                )
            progress_manifest = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                completion_status=CompletionStatus.MATERIALIZED,
                durable_truth_unit=progress.durable_truth_unit,
                incorporated_shard_keys=tuple(sorted(retained_keys)),
                incorporated_interval_ids=tuple(sorted(retained_interval_ids)),
                reclaimable_shard_keys=(),
                final_artifacts=build_residual_field_output_artifact_refs(
                    output_dir,
                    chunk_id,
                ),
                pending_shard_keys=(),
                pending_interval_ids=(),
                cleanup_policy=progress.cleanup_policy,
            )
            self.write_progress_manifest(progress_manifest)
        logger.warning(
            "Residual-field checkpoint invalidation | chunk=%d | dropped %d snapshot(s) "
            "from a previous partition layout (partitions %s); the affected "
            "partitions will be recomputed by the current plan.",
            int(chunk_id),
            len(dropped),
            sorted(
                "owner" if partition_id is None else str(partition_id)
                for partition_id, _seq in dropped
            ),
        )
        return len(dropped)

    def _checkpoint_cadence(
        self,
        total_expected_partials: int,
        *,
        partition_axis: str | None = None,
    ) -> int:
        # Cadence is pure POLICY (see reducer_helpers); the method stays as a thin
        # delegator so call sites are unchanged.
        return _checkpoint_cadence_policy(
            total_expected_partials,
            partition_axis=partition_axis,
        )

    def _live_trim_cadence(self, checkpoint_cadence: int) -> int:
        return _live_trim_cadence_policy(checkpoint_cadence)

    def _snapshot_local_accumulator(
        self,
        accumulator: LiveLocalAccumulator,
        *,
        output_dir: str,
        db_path: str,
        cleanup_policy: str,
    ) -> None:
        """Synchronous capture + commit (caller holds the per-target lock)."""
        pending = self._capture_local_snapshot(
            accumulator,
            output_dir=output_dir,
            db_path=db_path,
            cleanup_policy=cleanup_policy,
            capture_mode="view",
        )
        self._commit_local_snapshot(pending)

    def _capture_local_snapshot(
        self,
        accumulator: LiveLocalAccumulator,
        *,
        output_dir: str,
        db_path: str,
        cleanup_policy: str,
        capture_mode: str,
    ) -> _PendingLocalSnapshot:
        """Runs UNDER the per-target lock. capture_mode='copy' materializes
        private array copies (RAM targets) and 'clone' reflink-clones the
        memmap backing files (file targets) so the commit may run on the
        writer thread while folds continue; 'view' keeps live views for the
        synchronous path (no extra memory). 'clone' raises
        SnapshotCloneUnsupported on filesystems without reflink."""
        snapshot_seq = accumulator.next_snapshot_seq()
        if capture_mode == "copy":
            payload = accumulator.capture_snapshot_payload()
        elif capture_mode == "clone":
            payload = accumulator.capture_snapshot_payload_file_clone()
        else:
            payload = accumulator.snapshot_payload()
        accumulator.mark_snapshot_captured()
        return _PendingLocalSnapshot(
            key=self._local_accumulator_key(
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
                partition_id=accumulator.partition_id,
            ),
            accumulator=accumulator,
            snapshot_seq=int(snapshot_seq),
            payload=payload,
            captured_interval_ids=tuple(
                int(v) for v in payload["incorporated_interval_ids"]
            ),
            output_dir=str(output_dir),
            db_path=str(db_path),
            cleanup_policy=str(cleanup_policy),
        )

    def _commit_and_release_clones(self, pending: _PendingLocalSnapshot) -> None:
        """Writer-thread commit for clone/copy captures: after the durable
        commit (or its failure), unlink the reflink clone files a file-mode
        capture left in the live dir."""
        try:
            self._commit_local_snapshot(pending)
        finally:
            for clone_path in pending.payload.get("_clone_paths", ()):
                try:
                    Path(clone_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def _commit_local_snapshot(self, pending: _PendingLocalSnapshot) -> None:
        """The durable commit sequence, byte-identical to the historical
        inline path: write -> atomic rename -> chunk-mutex manifest union ->
        mark committed -> SQLite -> unlink previous seq. Runs either inline
        (sync path, caller holds the per-target lock; re-acquisition is
        reentrant) or on the writer thread (acquires the per-target lock
        only for the brief accumulator-mutating sections)."""
        accumulator = pending.accumulator
        snapshot_seq = pending.snapshot_seq
        snapshot_payload = pending.payload
        output_dir = pending.output_dir
        db_path = pending.db_path
        cleanup_policy = pending.cleanup_policy
        local_snapshot_path = write_local_accumulator_snapshot(
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
            checkpoint_write_count=int(accumulator.checkpoint_write_count) + 1,
            checkpoint_bytes_written_total=int(
                accumulator.checkpoint_bytes_written_total
            ),
            checkpoint_wall_seconds_total=float(
                accumulator.checkpoint_wall_seconds_total
            ),
            checkpoint_cadence_batches=int(accumulator.checkpoint_cadence_batches),
            point_start=accumulator.point_start,
            point_stop=accumulator.point_stop,
            accumulator_axis=getattr(accumulator, "accumulator_axis", "points"),
        )
        with self._local_accumulator_lock(pending.key):
            accumulator.record_checkpoint_metrics(
                bytes_written=int(local_snapshot_path.stat().st_size),
                wall_seconds=0.0,
                checkpoint_cadence_batches=max(
                    1,
                    int(accumulator.checkpoint_cadence_batches or 1),
                ),
            )
        new_snapshot_key = make_local_accumulator_snapshot_key(
            chunk_id=accumulator.chunk_id,
            parameter_digest=accumulator.parameter_digest,
            partition_id=accumulator.partition_id,
            snapshot_seq=snapshot_seq,
        )
        with chunk_mutex(accumulator.chunk_id, lock_root=output_dir):
            existing_progress = self.load_progress_manifest(
                output_dir=output_dir,
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
            )
            retained_snapshot_keys: list[str] = []
            replaced_snapshot_seqs: list[int] = []
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
                        parsed_seq,
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
                        # Fencing: a partition's manifest seq only ever
                        # ADVANCES. A replacement owner sequences from
                        # epoch * STRIDE, so a scheduler-declared-dead-
                        # but-alive predecessor that tries to commit
                        # after the remap lands here — loudly — instead
                        # of racing the live owner's rename and dropping
                        # its snapshot key from the family (which
                        # finalize would only discover at the very end
                        # of the run).
                        if int(parsed_seq) >= int(snapshot_seq):
                            raise RuntimeError(
                                "Residual-field snapshot commit superseded: "
                                f"chunk={int(accumulator.chunk_id)} "
                                f"partition={accumulator.partition_id} already "
                                f"has durable seq {int(parsed_seq)} >= "
                                f"{int(snapshot_seq)} — a replacement owner "
                                "has advanced this target; this process's "
                                "ownership is stale."
                            )
                        replaced_snapshot_seqs.append(int(parsed_seq))
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
        with self._local_accumulator_lock(pending.key):
            newly_durable = accumulator.mark_snapshot_committed(
                snapshot_seq, pending.captured_interval_ids
            )
        if accumulator.partition_id is None:
            status_updater = _ResidualFieldChunkStatusUpdater(db_path)
            status_updater.mark_saved_many(
                sorted(int(v) for v in newly_durable), int(accumulator.chunk_id)
            )
        # The manifest no longer references the replaced seqs; unlink
        # their files (covers both the standard previous-seq case and a
        # predecessor tenure's last snapshot after an epoch jump, which
        # snapshot_seq - 1 alone would orphan on disk).
        for stale_seq in set(replaced_snapshot_seqs) | {snapshot_seq - 1}:
            if stale_seq <= 0 or stale_seq == snapshot_seq:
                continue
            build_local_accumulator_snapshot_path(
                output_dir,
                chunk_id=accumulator.chunk_id,
                parameter_digest=accumulator.parameter_digest,
                partition_id=accumulator.partition_id,
                snapshot_seq=stale_seq,
            ).unlink(missing_ok=True)

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
        self.accept_local_contribution(
            partial.work_unit,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            contribution_reciprocal_points=partial.contribution_reciprocal_points,
            amplitudes_delta=partial.amplitudes_delta,
            amplitudes_average=partial.amplitudes_average,
            point_ids=partial.point_ids,
            output_dir=output_dir,
            scratch_root=scratch_root,
            db_path=db_path,
            total_expected_partials=total_expected_partials,
            cleanup_policy=cleanup_policy,
        )

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
        owner_epoch: int = 0,
    ) -> None:
        if not self.uses_owner_local_accumulator():
            raise ValueError(
                "accept_local_contribution requires an owner-local accumulator backend."
            )
        key = self._local_accumulator_key(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            partition_id=work_unit.partition_id,
        )
        with self._local_accumulator_lock(key):
            accumulator = self._get_or_create_local_accumulator_for_target(
                work_unit=work_unit,
                point_ids=point_ids,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
                output_dir=output_dir,
                scratch_root=scratch_root,
            )
            if owner_epoch:
                accumulator.raise_owner_epoch_floor(owner_epoch)
            before = tuple(sorted(accumulator.current_interval_ids))
            accumulator.accept_contribution(
                work_unit,
                point_ids=point_ids,
                grid_shape_nd=grid_shape_nd,
                total_reciprocal_points=total_reciprocal_points,
                contribution_reciprocal_points=contribution_reciprocal_points,
                amplitudes_delta=amplitudes_delta,
                amplitudes_average=amplitudes_average,
            )
            after = tuple(sorted(accumulator.current_interval_ids))
            if before == after:
                return
            snapshot_every = self._checkpoint_cadence(
                total_expected_partials,
                partition_axis=getattr(work_unit, "partition_axis", None),
            )
            accumulator.checkpoint_cadence_batches = int(snapshot_every)
            async_writes = self._async_snapshot_writes_enabled()
            if async_writes:
                # A failed async commit surfaces on the fold path, exactly
                # where the synchronous write would have raised.
                writer_error = self._snapshot_writer().pop_error(key)
                if writer_error is not None:
                    raise writer_error
            flushed = False
            captured = False
            if accumulator.accepted_since_snapshot >= snapshot_every:
                # RAM targets capture by private copy (memcpy << savez).
                # File targets capture by reflink-cloning their memmap
                # backing files — O(1) copy-on-write on XFS/btrfs/NFS4.2 —
                # so the multi-second savez of a GB-scale accumulator no
                # longer blocks folds (hkl40-scale targets are ALWAYS
                # file-mode; the old RAM-only gate meant the writer could
                # never fire on the workload that motivated it). Where the
                # scratch FS cannot reflink (ext4), file targets fall back
                # to the synchronous flush.
                if async_writes:
                    if not self._snapshot_writer().in_flight(key):
                        pending = None
                        if getattr(accumulator, "storage_mode", "ram") != "file":
                            pending = self._capture_local_snapshot(
                                accumulator,
                                output_dir=output_dir,
                                db_path=db_path,
                                cleanup_policy=cleanup_policy,
                                capture_mode="copy",
                            )
                        elif getattr(self, "_file_clone_captures_usable", True):
                            try:
                                pending = self._capture_local_snapshot(
                                    accumulator,
                                    output_dir=output_dir,
                                    db_path=db_path,
                                    cleanup_policy=cleanup_policy,
                                    capture_mode="clone",
                                )
                            except SnapshotCloneUnsupported as exc:
                                self._file_clone_captures_usable = False
                                logger.info(
                                    "Residual-field async snapshots: %s; "
                                    "file-mode targets use the synchronous "
                                    "flush.",
                                    exc,
                                )
                        if pending is not None:
                            self._snapshot_writer().submit(
                                key,
                                lambda pending=pending: self._commit_and_release_clones(
                                    pending
                                ),
                            )
                            accumulator.trim_live_memory()
                            captured = True
                        else:
                            flushed = self.flush_local_reducer_target(
                                chunk_id=work_unit.chunk_id,
                                parameter_digest=work_unit.parameter_digest,
                                partition_id=work_unit.partition_id,
                                output_dir=output_dir,
                                db_path=db_path,
                                cleanup_policy=cleanup_policy,
                            )
                    # else: a write is already in flight — keep folding;
                    # the cadence counter re-arms after that commit.
                else:
                    flushed = self.flush_local_reducer_target(
                        chunk_id=work_unit.chunk_id,
                        parameter_digest=work_unit.parameter_digest,
                        partition_id=work_unit.partition_id,
                        output_dir=output_dir,
                        db_path=db_path,
                        cleanup_policy=cleanup_policy,
                    )
            trim_every = self._live_trim_cadence(snapshot_every)
            if (
                not flushed
                and not captured
                and accumulator.accepted_since_snapshot > 0
                and accumulator.accepted_since_snapshot % trim_every == 0
            ):
                accumulator.trim_live_memory()

    def inspect_local_reducer_target(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        output_dir: str,
        partition_id: int | None = None,
        include_payload: bool = False,
    ) -> dict[str, object] | None:
        if not self.uses_owner_local_accumulator():
            raise ValueError(
                "inspect_local_reducer_target requires an owner-local accumulator backend."
            )
        key = self._local_accumulator_key(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
        )
        accumulator = self._local_accumulators.get(key)
        snapshot_state = self._load_latest_local_snapshot(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            partition_id=partition_id,
            include_payload=include_payload,
        )
        if accumulator is None and snapshot_state is None:
            return None
        snapshot_seq = 0
        snapshot_payload = None
        snapshot_path = None
        snapshot_manifest_path = None
        if snapshot_state is not None:
            snapshot_seq, snapshot_payload = snapshot_state
            snapshot_path = str(
                build_local_accumulator_snapshot_path(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                )
            )
        snapshot_bytes = 0
        if snapshot_path is not None:
            try:
                snapshot_bytes = int(Path(snapshot_path).stat().st_size)
            except OSError:
                pass
        checkpoint_metrics = {
            "total_checkpoint_bytes_written": snapshot_bytes,
            "total_checkpoint_writes": int(snapshot_seq),
            "total_checkpoint_wall_seconds": 0.0,
            "latest_generation_seq": int(snapshot_seq),
            "latest_checkpoint_bytes_written": snapshot_bytes,
            "latest_checkpoint_wall_seconds": 0.0,
        }
        return {
            "chunk_id": int(chunk_id),
            "parameter_digest": str(parameter_digest),
            "partition_id": int(partition_id) if partition_id is not None else None,
            "has_live_accumulator": accumulator is not None,
            "live_interval_ids": (
                tuple(sorted(int(v) for v in accumulator.current_interval_ids))
                if accumulator is not None
                else ()
            ),
            "durable_interval_ids": (
                tuple(sorted(int(v) for v in accumulator.durable_interval_ids))
                if accumulator is not None
                else (
                    tuple(
                        int(interval_id)
                        for interval_id in snapshot_payload["incorporated_interval_ids"]
                    )
                    if snapshot_payload is not None
                    else ()
                )
            ),
            "live_dirty": (
                accumulator is not None
                and accumulator.current_interval_ids != accumulator.durable_interval_ids
            ),
            "durable_snapshot_seq": int(snapshot_seq),
            "durable_snapshot_path": snapshot_path,
            "durable_snapshot_manifest_path": snapshot_manifest_path,
            "durable_snapshot_payload": snapshot_payload if include_payload else None,
            "checkpoint_metrics": checkpoint_metrics,
            **checkpoint_metrics,
        }

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
        if not self.uses_owner_local_accumulator():
            raise ValueError(
                "flush_local_reducer_target requires an owner-local accumulator backend."
            )
        key = self._local_accumulator_key(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
        )
        # Drain BEFORE taking the per-target lock (the async commit acquires
        # that lock and chunk_mutex — draining under it would deadlock), then
        # RE-CHECK in-flight under the lock: a fold can submit a fresh async
        # capture in the drain->lock gap, and a concurrent sync capture would
        # then claim the SAME snapshot_seq (durable_snapshot_seq only bumps
        # at commit) — two seq-N renames racing while the manifest union
        # records the superset. Submissions only happen under the per-target
        # lock, so once the check passes while we hold it, no new write can
        # start beneath us.
        drained = False
        writer = (
            self._snapshot_writer()
            if self._async_snapshot_writes_enabled()
            else None
        )
        lock = self._local_accumulator_lock(key)
        while True:
            if writer is not None:
                drained = writer.drain(key) or drained
            with lock:
                if writer is not None and writer.in_flight(key):
                    continue
                accumulator = self._local_accumulators.get(key)
                if accumulator is None:
                    return drained
                if accumulator.current_interval_ids == accumulator.durable_interval_ids:
                    accumulator.trim_live_memory()
                    return drained
                self._snapshot_local_accumulator(
                    accumulator,
                    output_dir=output_dir,
                    db_path=db_path,
                    cleanup_policy=cleanup_policy,
                )
                accumulator.trim_live_memory()
                return True

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
            # Current checkpoint payloads are uniformly uncompressed; old-codebase
            # checkpoint formats are not a supported compatibility input.
            compress=False,
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
        opportunistic: bool = False,
        expected_partitions: tuple[tuple[int | None, int | None, int | None], ...] | None = None,
        expected_interval_ids: tuple[int, ...] | None = None,
        mark_intervals_saved: bool = True,
    ) -> ResidualFieldArtifactManifest | None:
        """Fold committed local snapshots into the final chunk artifact.

        ``mark_intervals_saved=False`` defers SQLite interval marking to the
        driver (single writer) when finalizes run concurrently on many
        workers; SQLite is a rebuildable cache, so ordering only shifts the
        crash-repair window (manifests remain the authority).

        ``opportunistic=True`` marks a caller (startup recovery) that runs
        BEFORE planning and therefore cannot know the expected partition
        family: partitioned families are deferred (return ``None``) instead of
        published or failed, because partition snapshots flush on independent
        cadences and routinely disagree mid-run. ``expected_partitions`` is
        the plan's ``(partition_id, point_start, point_stop)`` family for this
        chunk; when provided, the snapshot family must match it exactly.
        ``expected_interval_ids`` is the plan's full interval set for this
        chunk; interval-partitioned (subchunk) families must union to it
        exactly before their snapshots are summed."""
        # In-flight async snapshot commits for this chunk must land before
        # finalize reads durable state (drain outside all locks). Loop:
        # a straggler fold can submit a new capture during the drain, and a
        # commit landing AFTER finalize published would rewrite the progress
        # manifest and re-reference snapshots finalize unlinked.
        if self._async_snapshot_writes_enabled():
            writer = self._snapshot_writer()

            def _matches(key) -> bool:
                return key[0] == int(chunk_id) and key[1] == str(parameter_digest)

            for _ in range(1000):
                writer.drain_matching(_matches)
                if not writer.has_matching(_matches):
                    break
        repaired_manifest = self._repair_progress_final_artifacts(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            db_path=db_path,
            cleanup_policy=cleanup_policy,
            scratch_root=scratch_root,
            mark_intervals_saved=mark_intervals_saved,
        )
        if repaired_manifest is not None:
            return repaired_manifest

        if self.uses_local_chunk_accumulator():
            if scratch_root is None:
                raise ValueError("Local accumulator finalization requires scratch_root.")
            matching_keys = [
                key
                for key in list(self._local_accumulators)
                if key[0] == int(chunk_id) and key[1] == str(parameter_digest)
            ]
            for key in matching_keys:
                # Same locking contract as the flush path: capture over live
                # array views may not interleave with a concurrent fold. The
                # dead-owner rescue can leave a zombie fold running on a
                # scheduler-evicted-but-alive worker, and an unlocked savez
                # interleaving with its accept_contribution yields internally
                # inconsistent amplitudes under a VALID interval set — the one
                # corruption shape family validation cannot detect.
                with self._local_accumulator_lock(key):
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
                # No committed snapshot keys in the progress manifest. The
                # old owner-restore fallback read the SAME manifest and so
                # could never produce a snapshot either — removed as
                # unreachable.
                return None
            snapshot_metadata: list[tuple[int | None, int, dict[str, object]]] = []
            for partition_id, snapshot_seq in snapshot_refs:
                metadata = load_local_accumulator_snapshot_metadata(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                )
                if metadata is not None:
                    snapshot_metadata.append((partition_id, int(snapshot_seq), metadata))
            if not snapshot_metadata:
                return None
            partitioned = any(
                metadata.get("partition_id") not in (None, -1)
                for _, _, metadata in snapshot_metadata
            )
            family_axes = {
                str(metadata.get("accumulator_axis", "points"))
                for _, _, metadata in snapshot_metadata
            }
            if len(family_axes) > 1:
                raise RuntimeError(
                    "Residual-field local finalization found checkpoints from "
                    f"BOTH partition axes for chunk={int(chunk_id)} "
                    f"({sorted(family_axes)}). A points-axis and an "
                    "intervals-axis (subchunk) layout are irreconcilable; "
                    "delete 'residual_checkpoints/' under the output "
                    "directory and re-run to recompute this chunk."
                )
            family_axis = family_axes.pop() if family_axes else "points"
            if opportunistic and (partitioned or expected_interval_ids is None):
                # Startup recovery runs BEFORE planning. Partition snapshots
                # flush on independent cadences, so a mid-run crash routinely
                # leaves them with different interval subsets and possibly a
                # missing tail partition -- and a NON-partitioned owner-level
                # snapshot just as routinely covers a strict subset of the
                # chunk's intervals (cadence flushes). Publishing either as
                # COMMITTED would credit the partial set and silently drop
                # the pre-crash amplitudes when the completion run overwrites
                # the chunk. Coverage is only provable against the plan;
                # defer to the stage.
                log_fn = logger.debug if quiet_logs else logger.info
                log_fn(
                    "Residual-field startup recovery deferring chunk %d to "
                    "the residual stage (family/coverage completeness is "
                    "only provable against the plan).",
                    int(chunk_id),
                )
                return None
            if partitioned and expected_partitions is not None:
                _require_expected_partition_family(
                    snapshot_metadata,
                    expected_partitions=expected_partitions,
                    chunk_id=chunk_id,
                )
            if partitioned:
                scratch_dir = _finalize_scratch_dir(
                    scratch_root,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                )
                if family_axis == "intervals":
                    snapshot_payload = assemble_interval_partitioned_chunk_payload(
                        snapshot_metadata=snapshot_metadata,
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        output_dir=output_dir,
                        scratch_dir=scratch_dir,
                        expected_interval_ids=expected_interval_ids,
                    )
                else:
                    snapshot_payload = assemble_local_snapshot_chunk_payload(
                        snapshot_metadata=snapshot_metadata,
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        output_dir=output_dir,
                        scratch_dir=scratch_dir,
                    )
                if snapshot_payload is None:
                    return None
                applied_set = set(
                    int(x) for x in snapshot_payload["incorporated_interval_ids"]
                )
            else:
                partition_id, snapshot_seq, _metadata = snapshot_metadata[0]
                snapshot_payload = load_local_accumulator_snapshot(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                )
                if snapshot_payload is None:
                    return None
                applied_set = set(
                    int(interval_id)
                    for interval_id in snapshot_payload["incorporated_interval_ids"]
                )
                if expected_interval_ids is not None:
                    expected_set = {int(v) for v in expected_interval_ids}
                    if applied_set != expected_set:
                        raise RuntimeError(
                            "Residual-field local finalization coverage "
                            f"mismatch for chunk={int(chunk_id)}: "
                            f"missing={sorted(expected_set - applied_set)} "
                            f"unexpected={sorted(applied_set - expected_set)}."
                        )
            with chunk_mutex(chunk_id, lock_root=output_dir):
                store = ResidualFieldArtifactStore(output_dir)
                _write_residual_field_chunk_payload_components(
                    store=store,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    point_ids=snapshot_payload["point_ids"],
                    grid_shape_nd=snapshot_payload["grid_shape_nd"],
                    amplitudes_delta=snapshot_payload["amplitudes_delta"],
                    amplitudes_average=snapshot_payload["amplitudes_average"],
                    reciprocal_point_count=int(snapshot_payload["reciprocal_point_count"]),
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
            final_artifacts_present = _manifest_final_artifacts_present(manifest)
            if not final_artifacts_present:
                raise RuntimeError(
                    "Residual-field local finalization did not publish all final artifacts "
                    f"for chunk={int(chunk_id)}."
                )
            snapshot_keys = tuple(
                sorted(
                    make_local_accumulator_snapshot_key(
                        chunk_id=chunk_id,
                        parameter_digest=parameter_digest,
                        partition_id=partition_id,
                        snapshot_seq=snapshot_seq,
                    )
                    for partition_id, snapshot_seq, _metadata in snapshot_metadata
                )
            )
            progress_manifest = _build_residual_field_reducer_progress_manifest(
                output_dir=output_dir,
                chunk_id=chunk_id,
                parameter_digest=parameter_digest,
                completion_status=CompletionStatus.COMMITTED,
                durable_truth_unit="committed_local_snapshot_generation",
                incorporated_shard_keys=snapshot_keys,
                incorporated_interval_ids=tuple(
                    sorted(
                        int(v)
                        for v in snapshot_payload["incorporated_interval_ids"]
                    )
                ),
                reclaimable_shard_keys=snapshot_keys,
                final_artifacts=manifest.artifacts,
                pending_shard_keys=(),
                pending_interval_ids=(),
                cleanup_policy=str(cleanup_policy or "off"),
            )
            self.write_progress_manifest(progress_manifest)
            if mark_intervals_saved:
                _mark_residual_intervals_saved(
                    db_path=db_path,
                    chunk_id=chunk_id,
                    interval_ids=tuple(
                        int(v)
                        for v in snapshot_payload["incorporated_interval_ids"]
                    ),
                )
            for partition_id, snapshot_seq, _metadata in snapshot_metadata:
                build_local_accumulator_snapshot_path(
                    output_dir,
                    chunk_id=chunk_id,
                    parameter_digest=parameter_digest,
                    partition_id=partition_id,
                    snapshot_seq=snapshot_seq,
                ).unlink(missing_ok=True)
            for key in matching_keys:
                self._release_local_accumulator(key)
            return manifest
        return None

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
    # Normalization raises for the retired durable_shared_restartable kind,
    # so local_restartable is the only layout that can be built.
    _normalize_reducer_backend_kind(kind)
    return ManifestDrivenResidualFieldReducerBackend(
        LOCAL_RESTARTABLE_LAYOUT,
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


def clear_process_local_residual_field_backends() -> None:
    for backend in list(_PROCESS_LOCAL_REDUCER_BACKENDS.values()):
        writer = getattr(backend, "_snapshot_writer_obj", None)
        if writer is not None:
            try:
                writer.drain_all()
            except Exception:
                logger.exception("async snapshot writer drain failed at clear")
        for key in list(getattr(backend, "_local_accumulators", {})):
            release = getattr(backend, "_release_local_accumulator", None)
            if callable(release):
                release(key)
                continue
            accumulator = getattr(backend, "_local_accumulators", {}).pop(key, None)
            if accumulator is not None:
                try:
                    accumulator.cleanup_live_files()
                except Exception:
                    pass
    _PROCESS_LOCAL_REDUCER_BACKENDS.clear()


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
    expected_partitions: tuple[tuple[int | None, int | None, int | None], ...] | None = None,
    expected_interval_ids: tuple[int, ...] | None = None,
    mark_intervals_saved: bool = True,
    opportunistic: bool = False,
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
        expected_partitions=expected_partitions,
        expected_interval_ids=expected_interval_ids,
        mark_intervals_saved=mark_intervals_saved,
        opportunistic=opportunistic,
    )


def flush_process_local_residual_reducer_target(
    template_backend: ManifestDrivenResidualFieldReducerBackend,
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None,
    output_dir: str,
    db_path: str,
    cleanup_policy: str = "off",
) -> bool:
    backend = get_process_local_residual_field_backend(template_backend)
    return backend.flush_local_reducer_target(
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        partition_id=partition_id,
        output_dir=output_dir,
        db_path=db_path,
        cleanup_policy=cleanup_policy,
    )


def inspect_process_local_residual_reducer_target(
    template_backend: ManifestDrivenResidualFieldReducerBackend,
    *,
    chunk_id: int,
    parameter_digest: str,
    output_dir: str,
    partition_id: int | None = None,
    include_payload: bool = False,
) -> dict[str, object] | None:
    backend = get_process_local_residual_field_backend(template_backend)
    return backend.inspect_local_reducer_target(
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        output_dir=output_dir,
        partition_id=partition_id,
        include_payload=include_payload,
    )


def resolve_residual_field_reducer_backend(
    *,
    workflow_parameters: "WorkflowParameters | object",
    client,
) -> ManifestDrivenResidualFieldReducerBackend:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    # Precedence rule, uniform for every MOSAIC_* knob: OPERATOR ENV WINS
    # over JSON config, config over default. This function used to apply
    # both orderings within twenty lines (config-wins for the shard root,
    # env-wins for the RAM cap) — pick one and say so.
    shard_storage_root_override = os.getenv("MOSAIC_RESIDUAL_SHARD_DURABLE_ROOT")
    local_accumulator_max_ram_bytes = os.getenv("MOSAIC_LOCAL_REDUCER_MAX_RAM_BYTES")
    if hasattr(runtime_info, "get"):
        if shard_storage_root_override is None:
            shard_storage_root_override = runtime_info.get(
                "residual_shard_durable_root"
            )
        if local_accumulator_max_ram_bytes is None:
            local_accumulator_max_ram_bytes = runtime_info.get(
                "residual_local_accumulator_max_ram_bytes"
            )
    local_accumulator_max_ram_bytes = int(
        local_accumulator_max_ram_bytes
        if local_accumulator_max_ram_bytes is not None
        else DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES
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
    # local_restartable is the only reducer layout. The tile-owner
    # architecture made it multi-node-safe (working accumulators on
    # node-local scratch, durable snapshots + progress manifests on the
    # shared output dir), and the multi-node NON-streaming case was
    # validated in the cluster sim on 2026-08-01 at max|diff| 1.7e-12 Å
    # vs reference — retiring the locality heuristic that routed
    # non-streaming distributed clients to durable_shared_restartable.
    return "local_restartable"


__all__ = [
    "clear_process_local_residual_field_backends",
    "finalize_process_local_residual_chunk",
    "flush_process_local_residual_reducer_target",
    "get_process_local_residual_field_backend",
    "inspect_process_local_residual_reducer_target",
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
