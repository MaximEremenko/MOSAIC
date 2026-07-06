"""Crash/resume integrity of the local-restartable reducer.

Regression tests for the truncated-chunk-on-resume defect: partition
checkpoint snapshots surviving a crash across a partition-layout change used
to be concatenated silently at finalize, publishing a truncated or corrupt
chunk. The fixes under test:

  * ``validate_local_partition_snapshot_family`` rejects checkpoint families
    with mismatched interval coverage or non-tiling atom ranges;
  * ``assemble_local_snapshot_chunk_payload`` refuses to assemble such a
    family;
  * ``invalidate_incompatible_local_checkpoints`` prunes snapshots that no
    longer match the planned partition layout OR interval grouping at plan
    time;
  * ``finalize_chunk(opportunistic=True)`` (startup recovery, which runs
    before planning) defers partitioned families instead of publishing or
    failing them — partition snapshots flush on independent cadences, so a
    mid-run crash routinely leaves them with different interval subsets;
  * ``finalize_chunk(expected_partitions=...)`` rejects snapshot families
    that do not match the plan (catches a missing TAIL partition, which the
    intra-family tiling check cannot see);
  * ``LiveLocalAccumulator.accept_contribution`` raises instead of
    double-counting when a contribution partially overlaps the checkpointed
    interval set (interval grouping changed across runs).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.contracts import CompletionStatus
from core.residual_field.artifacts import (
    _build_residual_field_reducer_progress_manifest,
    build_residual_field_output_artifact_refs,
)
from core.residual_field.backend import (
    LOCAL_RESTARTABLE_LAYOUT,
    ManifestDrivenResidualFieldReducerBackend,
    assemble_local_snapshot_chunk_payload,
    validate_local_partition_snapshot_family,
)
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.local_accumulator import (
    LiveLocalAccumulator,
    build_local_accumulator_snapshot_path,
    make_local_accumulator_snapshot_key,
    write_local_accumulator_snapshot,
)

DIGEST = "f" * 64
CHUNK_ID = 0


def _metadata(
    *,
    partition_id,
    point_start,
    point_stop,
    interval_ids=(1, 2, 3),
):
    return {
        "reciprocal_point_count": 10,
        "total_reciprocal_points": 100,
        "incorporated_interval_ids": tuple(interval_ids),
        "partition_id": partition_id,
        "storage_mode": "ram",
        "point_start": point_start,
        "point_stop": point_stop,
    }


def _family(entries):
    return [
        (partition_id, 1, _metadata(partition_id=partition_id, **kwargs))
        for partition_id, kwargs in entries
    ]


class TestValidatePartitionSnapshotFamily:
    def test_accepts_contiguous_tiling_with_matching_intervals(self):
        family = _family(
            [
                (0, {"point_start": 0, "point_stop": 10}),
                (1, {"point_start": 10, "point_stop": 25}),
                (2, {"point_start": 25, "point_stop": 30}),
            ]
        )
        validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_rejects_mismatched_interval_sets(self):
        family = _family(
            [
                (0, {"point_start": 0, "point_stop": 10}),
                (1, {"point_start": 10, "point_stop": 20, "interval_ids": (1, 2)}),
            ]
        )
        with pytest.raises(RuntimeError, match="mismatched interval coverage"):
            validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_rejects_gap_from_missing_partition(self):
        family = _family(
            [
                (0, {"point_start": 0, "point_stop": 10}),
                (2, {"point_start": 20, "point_stop": 30}),
            ]
        )
        with pytest.raises(RuntimeError, match="do not tile the chunk"):
            validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_rejects_overlap_from_stale_layout(self):
        # A zombie snapshot from an older, coarser layout overlaps the new one.
        family = _family(
            [
                (0, {"point_start": 0, "point_stop": 20}),
                (1, {"point_start": 10, "point_stop": 30}),
            ]
        )
        with pytest.raises(RuntimeError, match="do not tile the chunk"):
            validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_rejects_mixed_owner_and_partitioned(self):
        family = _family(
            [
                (None, {"point_start": None, "point_stop": None}),
                (0, {"point_start": 0, "point_stop": 10}),
            ]
        )
        with pytest.raises(RuntimeError, match="mix of owner-level and partitioned"):
            validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_legacy_snapshots_without_ranges_only_warn(self, caplog):
        family = _family(
            [
                (0, {"point_start": None, "point_stop": None}),
                (1, {"point_start": None, "point_stop": None}),
            ]
        )
        validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)

    def test_single_snapshot_is_trivially_valid(self):
        family = _family([(0, {"point_start": 5, "point_stop": 10})])
        validate_local_partition_snapshot_family(family, chunk_id=CHUNK_ID)


def _write_snapshot(
    output_dir,
    *,
    partition_id,
    snapshot_seq,
    n_points,
    point_start,
    point_stop,
    interval_ids=(1, 2, 3),
    fill=1.0,
):
    write_local_accumulator_snapshot(
        str(output_dir),
        chunk_id=CHUNK_ID,
        parameter_digest=DIGEST,
        partition_id=partition_id,
        snapshot_seq=snapshot_seq,
        point_ids=np.arange(n_points, dtype=np.int64),
        grid_shape_nd=np.array([[4, 4, 4]], dtype=np.int64),
        amplitudes_delta=np.full(n_points, fill, dtype=np.complex128),
        amplitudes_average=np.full(n_points, fill * 1j, dtype=np.complex128),
        reciprocal_point_count=10,
        total_reciprocal_points=100,
        incorporated_interval_ids=tuple(interval_ids),
        storage_mode="ram",
        point_start=point_start,
        point_stop=point_stop,
    )


class TestAssembleLocalSnapshotChunkPayload:
    def test_assembles_valid_family(self, tmp_path):
        _write_snapshot(
            tmp_path, partition_id=0, snapshot_seq=1,
            n_points=8, point_start=0, point_stop=2, fill=1.0,
        )
        _write_snapshot(
            tmp_path, partition_id=1, snapshot_seq=1,
            n_points=12, point_start=2, point_stop=5, fill=2.0,
        )
        family = [
            (0, 1, _metadata(partition_id=0, point_start=0, point_stop=2)),
            (1, 1, _metadata(partition_id=1, point_start=2, point_stop=5)),
        ]
        payload = assemble_local_snapshot_chunk_payload(
            snapshot_metadata=family,
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            scratch_dir=None,
        )
        assert payload is not None
        assert np.array_equal(
            np.asarray(payload["point_ids"]), np.arange(20, dtype=np.int64)
        )
        assert np.array_equal(
            np.asarray(payload["amplitudes_delta"]),
            np.concatenate([np.full(8, 1.0), np.full(12, 2.0)]).astype(np.complex128),
        )

    def test_refuses_zombie_partition_from_old_layout(self, tmp_path):
        _write_snapshot(
            tmp_path, partition_id=0, snapshot_seq=1,
            n_points=8, point_start=0, point_stop=2,
        )
        _write_snapshot(
            tmp_path, partition_id=1, snapshot_seq=1,
            n_points=12, point_start=2, point_stop=5,
        )
        # Zombie from an older layout: overlapping atom range.
        _write_snapshot(
            tmp_path, partition_id=2, snapshot_seq=1,
            n_points=6, point_start=3, point_stop=5,
        )
        family = [
            (0, 1, _metadata(partition_id=0, point_start=0, point_stop=2)),
            (1, 1, _metadata(partition_id=1, point_start=2, point_stop=5)),
            (2, 1, _metadata(partition_id=2, point_start=3, point_stop=5)),
        ]
        with pytest.raises(RuntimeError, match="do not tile the chunk"):
            assemble_local_snapshot_chunk_payload(
                snapshot_metadata=family,
                chunk_id=CHUNK_ID,
                parameter_digest=DIGEST,
                output_dir=str(tmp_path),
                scratch_dir=None,
            )


def _make_backend():
    return ManifestDrivenResidualFieldReducerBackend(LOCAL_RESTARTABLE_LAYOUT)


def _write_progress_manifest(backend, output_dir, snapshot_keys, interval_ids):
    manifest = _build_residual_field_reducer_progress_manifest(
        output_dir=str(output_dir),
        chunk_id=CHUNK_ID,
        parameter_digest=DIGEST,
        completion_status=CompletionStatus.MATERIALIZED,
        durable_truth_unit="committed_local_snapshot_generation",
        incorporated_shard_keys=tuple(sorted(snapshot_keys)),
        incorporated_interval_ids=tuple(sorted(interval_ids)),
        reclaimable_shard_keys=(),
        final_artifacts=build_residual_field_output_artifact_refs(
            str(output_dir), CHUNK_ID
        ),
        pending_shard_keys=(),
        pending_interval_ids=(),
        cleanup_policy="off",
    )
    backend.write_progress_manifest(manifest)


def _snapshot_key(partition_id, snapshot_seq):
    return make_local_accumulator_snapshot_key(
        chunk_id=CHUNK_ID,
        parameter_digest=DIGEST,
        partition_id=partition_id,
        snapshot_seq=snapshot_seq,
    )


def _target(start, stop, batches=((1, 2, 3),)):
    return {
        "point_start": start,
        "point_stop": stop,
        "interval_batches": tuple(frozenset(batch) for batch in batches),
    }


class TestInvalidateIncompatibleLocalCheckpoints:
    def _seed_two_partitions(self, tmp_path, backend):
        for partition_id, (start, stop) in {0: (0, 10), 1: (10, 20)}.items():
            _write_snapshot(
                tmp_path, partition_id=partition_id, snapshot_seq=2,
                n_points=stop - start, point_start=start, point_stop=stop,
            )
        _write_progress_manifest(
            backend,
            tmp_path,
            [_snapshot_key(0, 2), _snapshot_key(1, 2)],
            (1, 2, 3),
        )

    def test_matching_layout_keeps_everything(self, tmp_path):
        backend = _make_backend()
        self._seed_two_partitions(tmp_path, backend)
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={0: _target(0, 10), 1: _target(10, 20)},
        )
        assert dropped == 0
        progress = backend.load_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
        )
        assert set(progress.incorporated_shard_keys) == {
            _snapshot_key(0, 2),
            _snapshot_key(1, 2),
        }

    def test_durable_set_matching_a_union_of_batches_is_kept(self, tmp_path):
        backend = _make_backend()
        self._seed_two_partitions(tmp_path, backend)
        # durable {1,2,3} == union of planned batches {1,2} and {3}
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={
                0: _target(0, 10, batches=((1, 2), (3,), (4, 5))),
                1: _target(10, 20, batches=((1, 2), (3,), (4, 5))),
            },
        )
        assert dropped == 0

    def test_interval_regrouping_drops_unreconcilable_snapshots(self, tmp_path):
        backend = _make_backend()
        self._seed_two_partitions(tmp_path, backend)
        # durable {1,2,3} cannot be expressed as a union of {1,2},{3,4}:
        # skipping batch {1,2} and computing {3,4} would double-count 3,
        # and no batch re-covers a discarded 3 -- so the checkpoint must go.
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={
                0: _target(0, 10, batches=((1, 2), (3, 4))),
                1: _target(10, 20, batches=((1, 2), (3, 4))),
            },
        )
        assert dropped == 2
        progress = backend.load_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
        )
        assert progress.incorporated_shard_keys == ()

    def test_layout_change_drops_stale_snapshots_and_prunes_manifest(self, tmp_path):
        backend = _make_backend()
        self._seed_two_partitions(tmp_path, backend)
        # New plan: same partition 0, but partition 1 moved and partition 2 is new.
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={
                0: _target(0, 10),
                1: _target(10, 15),
                2: _target(15, 20),
            },
        )
        assert dropped == 1
        assert build_local_accumulator_snapshot_path(
            str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            partition_id=1,
            snapshot_seq=2,
        ).exists() is False
        assert build_local_accumulator_snapshot_path(
            str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            partition_id=0,
            snapshot_seq=2,
        ).exists()
        progress = backend.load_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
        )
        assert set(progress.incorporated_shard_keys) == {_snapshot_key(0, 2)}

    def test_zombie_partition_not_in_plan_is_dropped(self, tmp_path):
        backend = _make_backend()
        self._seed_two_partitions(tmp_path, backend)
        # Old layout had 2 partitions; new plan has a single owner partition 0.
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={0: _target(0, 20)},
        )
        assert dropped == 2
        progress = backend.load_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
        )
        assert progress.incorporated_shard_keys == ()
        assert progress.incorporated_interval_ids == ()

    def test_committed_chunk_is_left_alone(self, tmp_path):
        backend = _make_backend()
        for partition_id, (start, stop) in {0: (0, 10), 1: (10, 20)}.items():
            _write_snapshot(
                tmp_path, partition_id=partition_id, snapshot_seq=2,
                n_points=stop - start, point_start=start, point_stop=stop,
            )
        manifest = _build_residual_field_reducer_progress_manifest(
            output_dir=str(tmp_path),
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            completion_status=CompletionStatus.COMMITTED,
            durable_truth_unit="committed_local_snapshot_generation",
            incorporated_shard_keys=(_snapshot_key(0, 2), _snapshot_key(1, 2)),
            incorporated_interval_ids=(1, 2, 3),
            reclaimable_shard_keys=(),
            final_artifacts=build_residual_field_output_artifact_refs(
                str(tmp_path), CHUNK_ID
            ),
            pending_shard_keys=(),
            pending_interval_ids=(),
            cleanup_policy="off",
        )
        backend.write_progress_manifest(manifest)
        dropped = backend.invalidate_incompatible_local_checkpoints(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            expected_targets={0: _target(0, 20)},
        )
        assert dropped == 0


def _work_unit(interval_ids, *, partition_id=0, point_start=0, point_stop=4):
    return ResidualFieldWorkUnit.interval_chunk_batch(
        interval_ids=tuple(int(v) for v in interval_ids),
        chunk_id=CHUNK_ID,
        parameter_digest=DIGEST,
        output_dir="unused",
    ).with_partition(
        partition_id=partition_id,
        point_start=point_start,
        point_stop=point_stop,
    )


class TestAccumulatorPartialOverlapRestart:
    def _fresh_accumulator(self, tmp_path, work_unit, n_points=6):
        return LiveLocalAccumulator.from_arrays(
            work_unit,
            point_ids=np.arange(n_points, dtype=np.int64),
            grid_shape_nd=np.array([[4, 4, 4]], dtype=np.int64),
            total_reciprocal_points=100,
            amplitudes_delta=np.zeros(n_points, dtype=np.complex128),
            amplitudes_average=np.zeros(n_points, dtype=np.complex128),
            scratch_root=str(tmp_path / "scratch"),
            max_ram_bytes=1 << 30,
        )

    def _accept(self, accumulator, work_unit, value, n_points=6):
        accumulator.accept_contribution(
            work_unit,
            point_ids=np.arange(n_points, dtype=np.int64),
            grid_shape_nd=np.array([[4, 4, 4]], dtype=np.int64),
            total_reciprocal_points=100,
            contribution_reciprocal_points=10,
            amplitudes_delta=np.full(n_points, value, dtype=np.complex128),
            amplitudes_average=np.full(n_points, value, dtype=np.complex128),
        )

    def test_partial_overlap_raises_instead_of_double_counting(self, tmp_path):
        unit_a = _work_unit((1, 2))
        accumulator = self._fresh_accumulator(tmp_path, unit_a)
        self._accept(accumulator, unit_a, 1.0)
        assert accumulator.current_interval_ids == {1, 2}

        # Interval grouping changed across a crash/resume: the new work unit
        # covers {1, 2, 3, 4} while {1, 2} is already accumulated. Adding
        # would double-count 1 and 2; resetting could destroy intervals that
        # filtered-out work units never re-cover. Plan-time checkpoint
        # invalidation makes this state unreachable in the orchestrated flow,
        # so reaching it is a hard error -- and the accumulated state must be
        # left untouched.
        unit_b = _work_unit((1, 2, 3, 4))
        with pytest.raises(ValueError, match="interval grouping changed"):
            self._accept(accumulator, unit_b, 2.0)
        assert accumulator.current_interval_ids == {1, 2}
        assert np.array_equal(
            np.asarray(accumulator.amplitudes_delta),
            np.full(6, 1.0, dtype=np.complex128),
        )
        assert accumulator.reciprocal_point_count == 10

    def test_subset_contribution_still_skipped(self, tmp_path):
        unit_a = _work_unit((1, 2, 3))
        accumulator = self._fresh_accumulator(tmp_path, unit_a)
        self._accept(accumulator, unit_a, 1.0)
        unit_b = _work_unit((2, 3))
        self._accept(accumulator, unit_b, 5.0)
        assert np.array_equal(
            np.asarray(accumulator.amplitudes_delta),
            np.full(6, 1.0, dtype=np.complex128),
        )
        assert accumulator.current_interval_ids == {1, 2, 3}

    def test_disjoint_contribution_accumulates(self, tmp_path):
        unit_a = _work_unit((1, 2))
        accumulator = self._fresh_accumulator(tmp_path, unit_a)
        self._accept(accumulator, unit_a, 1.0)
        unit_b = _work_unit((3, 4))
        self._accept(accumulator, unit_b, 2.0)
        assert np.array_equal(
            np.asarray(accumulator.amplitudes_delta),
            np.full(6, 3.0, dtype=np.complex128),
        )
        assert accumulator.current_interval_ids == {1, 2, 3, 4}
        assert accumulator.reciprocal_point_count == 20


class TestFinalizeChunkRecoverySemantics:
    """finalize_chunk behavior for partitioned families across resume modes."""

    def _seed_partitions(self, tmp_path, backend, partitions, interval_ids=(1, 2, 3)):
        keys = []
        for partition_id, (start, stop) in partitions.items():
            _write_snapshot(
                tmp_path, partition_id=partition_id, snapshot_seq=1,
                n_points=(stop - start) * 2, point_start=start, point_stop=stop,
                interval_ids=interval_ids,
            )
            keys.append(_snapshot_key(partition_id, 1))
        _write_progress_manifest(backend, tmp_path, keys, interval_ids)

    def test_opportunistic_finalize_defers_partitioned_family(self, tmp_path):
        """Startup recovery runs before planning: a mid-run partitioned crash
        state (here: partitions with DIFFERENT interval subsets, the routine
        result of independent flush cadences) must be deferred to the stage,
        not raised on and not published."""
        backend = _make_backend()
        for partition_id, (start, stop), intervals in (
            (0, (0, 10), (1, 2, 3)),
            (1, (10, 20), (1,)),
        ):
            _write_snapshot(
                tmp_path, partition_id=partition_id, snapshot_seq=1,
                n_points=(stop - start) * 2, point_start=start, point_stop=stop,
                interval_ids=intervals,
            )
        _write_progress_manifest(
            backend, tmp_path,
            [_snapshot_key(0, 1), _snapshot_key(1, 1)],
            (1, 2, 3),
        )
        manifest = backend.finalize_chunk(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            db_path=str(tmp_path / "unused.db"),
            cleanup_policy="off",
            scratch_root=str(tmp_path / "scratch"),
            quiet_logs=True,
            opportunistic=True,
        )
        assert manifest is None
        # nothing was published
        assert not list(tmp_path.glob("residual_chunk_*"))

    def test_opportunistic_finalize_defers_even_a_consistent_partitioned_family(self, tmp_path):
        """Even a family that LOOKS complete (equal interval sets, contiguous
        tiling) cannot be proven complete without the plan -- a missing tail
        partition would look identical. recover_pending must always defer."""
        backend = _make_backend()
        self._seed_partitions(tmp_path, backend, {0: (0, 10), 1: (10, 20)})
        manifest = backend.finalize_chunk(
            chunk_id=CHUNK_ID,
            parameter_digest=DIGEST,
            output_dir=str(tmp_path),
            db_path=str(tmp_path / "unused.db"),
            cleanup_policy="off",
            scratch_root=str(tmp_path / "scratch"),
            quiet_logs=True,
            opportunistic=True,
        )
        assert manifest is None

    def test_strict_finalize_rejects_missing_tail_partition(self, tmp_path):
        """A K-of-N family that tiles contiguously from 0 passes the
        intra-family checks; only the plan's expected family catches the
        missing tail."""
        backend = _make_backend()
        self._seed_partitions(tmp_path, backend, {0: (0, 10), 1: (10, 20)})
        with pytest.raises(RuntimeError, match="does not match the plan"):
            backend.finalize_chunk(
                chunk_id=CHUNK_ID,
                parameter_digest=DIGEST,
                output_dir=str(tmp_path),
                db_path=str(tmp_path / "unused.db"),
                cleanup_policy="off",
                scratch_root=str(tmp_path / "scratch"),
                quiet_logs=True,
                expected_partitions=((0, 0, 10), (1, 10, 20), (2, 20, 30)),
            )

    def test_strict_finalize_rejects_moved_partition_range(self, tmp_path):
        backend = _make_backend()
        self._seed_partitions(tmp_path, backend, {0: (0, 10), 1: (10, 20)})
        with pytest.raises(RuntimeError, match="where the plan expects"):
            backend.finalize_chunk(
                chunk_id=CHUNK_ID,
                parameter_digest=DIGEST,
                output_dir=str(tmp_path),
                db_path=str(tmp_path / "unused.db"),
                cleanup_policy="off",
                scratch_root=str(tmp_path / "scratch"),
                quiet_logs=True,
                expected_partitions=((0, 0, 12), (1, 12, 20)),
            )
