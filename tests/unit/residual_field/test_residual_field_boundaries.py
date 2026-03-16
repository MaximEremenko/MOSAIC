import logging
from types import SimpleNamespace

import numpy as np
import pytest

from core.runtime import configure_progress
from core.residual_field.backend import (
    ResidualFieldLocalAccumulatorPartial,
    build_residual_field_reducer_backend,
    resolve_residual_field_reducer_backend,
)
from core.residual_field.artifacts import (
    ResidualFieldArtifactStore,
    persist_residual_field_interval_chunk_result,
)
from core.residual_field.loader import load_chunk_residual_field_and_grid
from core.residual_field.planning import (
    _weighted_partition_split,
    build_adaptive_partition_plan,
    build_residual_field_parameter_digest,
    build_residual_field_work_units,
    partition_residual_field_work_units,
)
from core.residual_field.execution import (
    _build_task_reducer_backend,
    _cap_async_max_inflight,
    _distributed_owner_affinity_enabled,
    _distributed_owner_local_reducer_supported,
    _residual_partition_runtime_policy,
    run_residual_field_stage,
)
from core.residual_field.contracts import (
    ResidualFieldAccumulatorStatus,
    ResidualFieldWorkUnit,
)
from core.residual_field.stage import ResidualFieldStage
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.kernels import IntervalTask
from core.storage.database_manager import DatabaseManager


class _CapturingReducerBackend:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def uses_local_chunk_accumulator(self):
        return False

    def persist_shard_checkpoint(self, work_unit, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def build_local_partial(self, work_unit, **kwargs):
        return ResidualFieldLocalAccumulatorPartial(work_unit=work_unit, **kwargs)


class _StopAfterIntervalPayloadScatter(RuntimeError):
    pass


class _FakeLocalReducerBackend:
    layout = SimpleNamespace(kind="local_restartable")

    def __init__(self):
        self.accepted: list[ResidualFieldLocalAccumulatorPartial] = []
        self.reconciled: list[int] = []
        self.flushed: list[tuple[int, int | None]] = []
        self.pending_by_target: dict[tuple[int, int | None], set[int]] = {}
        self.durable_by_target: dict[tuple[int, int | None], set[int]] = {}

    def uses_local_chunk_accumulator(self):
        return True

    def describe_runtime_state(self, *, output_dir, scratch_root):
        return SimpleNamespace(
            kind="local_restartable",
            local_scratch_root=scratch_root,
            durable_root=output_dir,
            ram_state=(),
            local_scratch_state=(),
            durable_state=(),
            scattering_interval_transport="direct-handoff",
            uncommitted_restart_rule="recompute",
            committed_shard_root=output_dir,
            committed_shard_storage="n/a",
            shard_compression="n/a",
            durable_truth_unit="committed_local_snapshot_generation",
            live_state_storage_role="owner-local-live-accumulator",
            durable_checkpoint_storage_role="durable-local-snapshot-generation",
            final_artifact_storage_role="durable-final-chunk-artifact",
            direct_interval_handoff_supported=True,
            checkpoint_policy=SimpleNamespace(
                interval_artifacts="optional_output",
                shard_checkpoints="required_local_restart_state",
                reducer_progress_manifest="required_durable",
                final_chunk_artifacts="required_durable",
                worker_local_scratch_role="committed_local_restart_state_and_temporary_staging",
            ),
        )

    def accept_partial(
        self,
        partial,
        *,
        output_dir,
        scratch_root,
        db_path,
        total_expected_partials,
        cleanup_policy="off",
    ):
        self.accepted.append(partial)
        target_key = (
            int(partial.work_unit.chunk_id),
            (
                int(partial.work_unit.partition_id)
                if partial.work_unit.partition_id is not None
                else None
            ),
        )
        interval_ids = set(partial.interval_ids or ((partial.work_unit.interval_id,) if partial.work_unit.interval_id is not None else ()))
        self.pending_by_target.setdefault(target_key, set()).update(int(interval_id) for interval_id in interval_ids)

    def accept_local_contribution(
        self,
        work_unit,
        *,
        grid_shape_nd,
        total_reciprocal_points,
        contribution_reciprocal_points,
        amplitudes_delta,
        amplitudes_average,
        point_ids,
        output_dir,
        scratch_root,
        db_path,
        total_expected_partials,
        cleanup_policy="off",
    ):
        partial = ResidualFieldLocalAccumulatorPartial(
            work_unit=work_unit,
            point_ids=point_ids,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            contribution_reciprocal_points=contribution_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
        )
        self.accept_partial(
            partial,
            output_dir=output_dir,
            scratch_root=scratch_root,
            db_path=db_path,
            total_expected_partials=total_expected_partials,
            cleanup_policy=cleanup_policy,
        )

    def reconcile_progress(
        self,
        *,
        chunk_id,
        parameter_digest,
        output_dir,
        db_path,
        manifests=None,
        scratch_root=None,
    ):
        self.reconciled.append(int(chunk_id))
        return None

    def local_intervals_already_durable(self, work_unit, *, output_dir):
        return False

    def flush_local_reducer_target(
        self,
        *,
        chunk_id,
        parameter_digest,
        output_dir,
        db_path,
        scratch_root=None,
        partition_id=None,
        cleanup_policy="off",
    ):
        target_key = (int(chunk_id), None if partition_id is None else int(partition_id))
        self.flushed.append(target_key)
        self.durable_by_target.setdefault(target_key, set()).update(
            self.pending_by_target.get(target_key, set())
        )
        self.pending_by_target[target_key] = set()
        return True

    def inspect_local_reducer_target(
        self,
        *,
        chunk_id,
        parameter_digest,
        output_dir,
        partition_id=None,
    ):
        target_key = (int(chunk_id), None if partition_id is None else int(partition_id))
        return {
            "durable_interval_ids": tuple(sorted(self.durable_by_target.get(target_key, set()))),
        }

    def finalize_chunk(
        self,
        *,
        chunk_id,
        parameter_digest,
        output_dir,
        db_path,
        manifests=None,
        cleanup_policy=None,
        scratch_root=None,
        quiet_logs=False,
    ):
        return None

    def cleanup_reclaimable_shards(
        self,
        *,
        output_dir,
        chunk_id,
        parameter_digest,
        db_path,
        manifests=None,
        scratch_root=None,
    ):
        return ()


class _FakeDistributedOwnerLocalBackend(_FakeLocalReducerBackend):
    distributed_owner_local_reducer_supported = True

    def __init__(self, *, supported=True, metrics_by_target=None):
        super().__init__()
        self.distributed_owner_local_reducer_supported = supported
        self.metrics_by_target = metrics_by_target or {}
        self.layout = SimpleNamespace(kind="durable_shared_restartable")

    def uses_local_chunk_accumulator(self):
        return False

    def describe_runtime_state(self, *, output_dir, scratch_root):
        return SimpleNamespace(
            kind="durable_shared_restartable",
            local_scratch_root=scratch_root,
            durable_root=output_dir,
            ram_state=(),
            local_scratch_state=(),
            durable_state=(),
            scattering_interval_transport="durable interval artifacts required execution transport",
            uncommitted_restart_rule="recompute",
            committed_shard_root=output_dir,
            committed_shard_storage="durable shared storage",
            shard_compression="np.savez_compressed",
            durable_truth_unit="committed_local_snapshot_generation",
            live_state_storage_role="owner-local-live-accumulator",
            durable_checkpoint_storage_role="durable-local-snapshot-generation",
            final_artifact_storage_role="durable-final-chunk-artifact",
            direct_interval_handoff_supported=False,
            checkpoint_policy=SimpleNamespace(
                interval_artifacts="required_transport",
                shard_checkpoints="required_durable_checkpoint",
                reducer_progress_manifest="required_durable",
                final_chunk_artifacts="required_durable",
                worker_local_scratch_role="temporary_staging_only",
            ),
        )

    def inspect_local_reducer_target(
        self,
        *,
        chunk_id,
        parameter_digest,
        output_dir,
        partition_id=None,
    ):
        target_key = (int(chunk_id), None if partition_id is None else int(partition_id))
        target_state = super().inspect_local_reducer_target(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            output_dir=output_dir,
            partition_id=partition_id,
        ) or {}
        target_state.update(self.metrics_by_target.get(target_key, {}))
        return target_state


def _point_rows_2d(count: int) -> list[dict[str, object]]:
    return [
        {
            "coordinates": np.array([float(index), float(index)]),
        }
        for index in range(count)
    ]


def _point_rows_3d(count: int) -> list[dict[str, object]]:
    return [
        {
            "coordinates": np.array([float(index), float(index), float(index)]),
        }
        for index in range(count)
    ]


def test_residual_field_planning_builds_interval_chunk_work_units(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(2, 3), (1, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
    )

    assert [(unit.interval_id, unit.chunk_id) for unit in work_units] == [(1, 3), (2, 3)]
    assert work_units[0].source_artifacts[0].kind == "interval-precompute"
    assert work_units[0].retry.idempotency_key.endswith("interval-1")
    assert build_residual_field_parameter_digest(parameters) == work_units[0].parameter_digest


def test_residual_field_planning_batches_intervals_per_chunk(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(1, 3), (2, 3), (3, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
        max_intervals_per_shard=2,
    )

    assert [unit.interval_ids for unit in work_units] == [(1, 2), (3,)]
    assert work_units[0].retry.idempotency_key.endswith("batch-1-2-n2")
    assert work_units[1].retry.idempotency_key.endswith("interval-3")


def test_residual_field_planning_can_partition_work_units_for_local_owner(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(1, 3), (2, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
    )

    partitioned = partition_residual_field_work_units(
        work_units,
        point_counts_by_chunk={3: 4},
        target_partitions_by_chunk={3: 2},
    )

    assert [(unit.partition_id, unit.point_start, unit.point_stop) for unit in partitioned] == [
        (0, 0, 2),
        (1, 2, 4),
        (0, 0, 2),
        (1, 2, 4),
    ]
    assert all("partition-" in unit.artifact_key for unit in partitioned)


def test_weighted_partition_split_balances_rifft_weights_with_contiguous_ranges():
    selections = _weighted_partition_split(
        np.arange(4, dtype=np.int64),
        np.array([1, 1, 8, 8], dtype=np.int64),
        2,
    )

    assert [selection.tolist() for selection in selections] == [[0, 1, 2], [3]]


def test_residual_field_planning_partitions_by_rifft_weight_not_equal_atom_count(tmp_path):
    parameters = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "rspace_info": {"mode": "displacement"},
    }
    work_units = build_residual_field_work_units(
        [(1, 3)],
        parameters=parameters,
        output_dir=str(tmp_path),
    )

    partitioned = partition_residual_field_work_units(
        work_units,
        point_counts_by_chunk={3: 4},
        target_partitions_by_chunk={3: 2},
        rifft_points_by_chunk={3: np.array([1, 1, 8, 8], dtype=np.int64)},
    )

    assert [(unit.partition_id, unit.point_start, unit.point_stop) for unit in partitioned] == [
        (0, 0, 3),
        (1, 3, 4),
    ]


def test_adaptive_partition_plan_splits_when_estimated_bytes_exceed_budget():
    point_rows_by_chunk = {
        3: [
            {
                "coordinates": np.array([0.0, 0.0]),
                "dist_from_atom_center": np.array([2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5]),
            },
            {
                "coordinates": np.array([1.0, 1.0]),
                "dist_from_atom_center": np.array([2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5]),
            },
        ]
    }

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=4,
        target_partition_bytes=100,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.target_partitions > 1
    assert plan.reason in {"byte-budget", "worker-capacity-floor"}
    assert plan.rifft_points_per_atom == (81, 81)
    assert plan.partition_rifft_points == (81, 81)
    assert plan.partition_imbalance_ratio == pytest.approx(1.0)


def test_adaptive_partition_plan_stays_whole_chunk_below_hysteresis_low_threshold():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(2)},
        effective_nufft_workers=1,
        target_partition_bytes=160,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 112
    assert plan.target_partitions == 1
    assert plan.reason == "whole-chunk"


def test_adaptive_partition_plan_uses_hysteresis_in_dead_zone():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(2)},
        effective_nufft_workers=1,
        target_partition_bytes=112,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 112
    assert plan.target_partitions == 2
    assert plan.reason == "byte-budget-hysteresis"


def test_adaptive_partition_plan_uses_normal_byte_budget_above_hysteresis_high_threshold():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(2)},
        effective_nufft_workers=1,
        target_partition_bytes=74,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 112
    assert plan.target_partitions == 2
    assert plan.reason == "byte-budget"


def test_adaptive_partition_plan_scales_normally_well_above_target_bytes():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(3)},
        effective_nufft_workers=1,
        target_partition_bytes=56,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 168
    assert plan.target_partitions == 3
    assert plan.reason == "byte-budget"


def test_adaptive_partition_plan_preserves_worker_floor_dominance_over_hysteresis():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(4)},
        effective_nufft_workers=4,
        target_partition_bytes=500,
        max_partitions_per_chunk=8,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 224
    assert plan.target_partitions == 4
    assert plan.reason == "worker-capacity-floor"


def test_adaptive_partition_plan_can_disable_hysteresis():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_2d(2)},
        effective_nufft_workers=1,
        target_partition_bytes=111,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
        hysteresis_low_factor=1.0,
        hysteresis_high_factor=1.0,
    )[3]

    assert plan.estimated_bytes == 112
    assert plan.target_partitions == 2
    assert plan.reason == "byte-budget"


def test_adaptive_partition_plan_uses_3d_hysteresis_reason_in_dead_zone():
    plan = build_adaptive_partition_plan(
        {3: _point_rows_3d(2)},
        effective_nufft_workers=1,
        target_partition_bytes=512,
        target_partition_bytes_3d=128,
        max_partitions_per_chunk=4,
        min_points_per_partition=1,
    )[3]

    assert plan.estimated_bytes == 128
    assert plan.target_partitions == 2
    assert plan.reason == "3d-byte-budget-hysteresis"


def test_adaptive_partition_plan_tracks_weighted_partition_balance():
    point_rows_by_chunk = {
        3: [
            {
                "coordinates": np.array([0.0, 0.0, 0.0]),
                "dist_from_atom_center": np.array([0.1, 0.1, 0.1]),
                "step_in_frac": np.array([0.1, 0.1, 0.1]),
            },
            {
                "coordinates": np.array([1.0, 1.0, 1.0]),
                "dist_from_atom_center": np.array([0.1, 0.1, 0.1]),
                "step_in_frac": np.array([0.1, 0.1, 0.1]),
            },
            {
                "coordinates": np.array([2.0, 2.0, 2.0]),
                "dist_from_atom_center": np.array([0.4, 0.4, 0.4]),
                "step_in_frac": np.array([0.1, 0.1, 0.1]),
            },
            {
                "coordinates": np.array([3.0, 3.0, 3.0]),
                "dist_from_atom_center": np.array([0.4, 0.4, 0.4]),
                "step_in_frac": np.array([0.1, 0.1, 0.1]),
            },
        ]
    }

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=2,
        target_partition_bytes=10_000_000,
        target_partition_bytes_3d=10_000_000,
        max_partitions_per_chunk=2,
        min_points_per_partition=1,
    )[3]

    assert plan.rifft_points_per_atom == (1, 1, 729, 729)
    assert plan.target_partitions == 2
    assert plan.partition_rifft_points == (731, 729)
    assert plan.partition_imbalance_ratio == pytest.approx(731 / 729)


def test_adaptive_partition_plan_biases_3d_cutover():
    point_rows_2d = {
        3: [
            {
                "coordinates": np.array([0.0, 0.0]),
                "dist_from_atom_center": np.array([2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5]),
            },
            {
                "coordinates": np.array([1.0, 1.0]),
                "dist_from_atom_center": np.array([2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5]),
            },
        ]
    }
    point_rows_3d = {
        3: [
            {
                "coordinates": np.array([0.0, 0.0, 0.0]),
                "dist_from_atom_center": np.array([2.0, 2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5, 0.5]),
            },
            {
                "coordinates": np.array([1.0, 1.0, 1.0]),
                "dist_from_atom_center": np.array([2.0, 2.0, 2.0]),
                "step_in_frac": np.array([0.5, 0.5, 0.5]),
            },
        ]
    }

    plan_2d = build_adaptive_partition_plan(
        point_rows_2d,
        effective_nufft_workers=2,
        target_partition_bytes=10_000,
        target_partition_bytes_3d=2_000,
    )[3]
    plan_3d = build_adaptive_partition_plan(
        point_rows_3d,
        effective_nufft_workers=2,
        target_partition_bytes=10_000,
        target_partition_bytes_3d=2_000,
    )[3]

    assert plan_3d.estimated_bytes > plan_2d.estimated_bytes
    assert plan_3d.target_partition_bytes < plan_2d.target_partition_bytes
    assert plan_3d.target_partitions >= plan_2d.target_partitions
    assert plan_3d.reason in {"3d-byte-budget", "worker-capacity-floor", "partition-cap"}


def _make_uniform_2d_point_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "coordinates": np.array([float(index), float(index)], dtype=np.float64),
            "dist_from_atom_center": np.array([2.0, 2.0], dtype=np.float64),
            "step_in_frac": np.array([0.5, 0.5], dtype=np.float64),
        }
        for index in range(count)
    ]


def _make_uniform_3d_point_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "coordinates": np.array([float(index), float(index), float(index)], dtype=np.float64),
            "dist_from_atom_center": np.array([2.0, 2.0, 2.0], dtype=np.float64),
            "step_in_frac": np.array([0.5, 0.5, 0.5], dtype=np.float64),
        }
        for index in range(count)
    ]


def test_adaptive_partition_plan_stays_whole_chunk_below_hysteresis_low_threshold():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(2)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
    )[3]
    target_bytes = int(np.ceil(float(baseline.estimated_bytes) / 0.7))

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=target_bytes,
    )[3]

    assert plan.target_partitions == 1
    assert plan.reason == "whole-chunk"


def test_adaptive_partition_plan_enters_hysteresis_dead_zone_for_2d_cutover():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(2)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
    )[3]

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=baseline.estimated_bytes,
    )[3]

    assert plan.target_partitions == 2
    assert plan.reason == "byte-budget-hysteresis"


def test_adaptive_partition_plan_uses_normal_byte_budget_above_hysteresis_high_threshold():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(2)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
    )[3]
    target_bytes = int(np.ceil(float(baseline.estimated_bytes) / 1.5))

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=target_bytes,
    )[3]

    assert plan.target_partitions == 2
    assert plan.reason == "byte-budget"


def test_adaptive_partition_plan_keeps_ceil_behavior_well_above_hysteresis_high_threshold():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(4)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
    )[3]
    target_bytes = int(np.ceil(float(baseline.estimated_bytes) / 3.0))

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=target_bytes,
        max_partitions_per_chunk=4,
    )[3]

    assert plan.target_partitions == 3
    assert plan.reason == "byte-budget"


def test_adaptive_partition_plan_worker_floor_still_dominates_hysteresis():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(4)}

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=4,
        target_partition_bytes=1_000_000,
        max_partitions_per_chunk=4,
    )[3]

    assert plan.target_partitions == 4
    assert plan.reason == "worker-capacity-floor"


def test_adaptive_partition_plan_can_disable_hysteresis():
    point_rows_by_chunk = {3: _make_uniform_2d_point_rows(2)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
    )[3]

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=baseline.estimated_bytes,
        hysteresis_low_factor=1.0,
        hysteresis_high_factor=1.0,
    )[3]

    assert plan.target_partitions == 1
    assert plan.reason == "whole-chunk"


def test_adaptive_partition_plan_enters_hysteresis_dead_zone_for_3d_cutover():
    point_rows_by_chunk = {3: _make_uniform_3d_point_rows(2)}
    baseline = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=1_000_000,
        target_partition_bytes_3d=1_000_000,
    )[3]

    plan = build_adaptive_partition_plan(
        point_rows_by_chunk,
        effective_nufft_workers=1,
        target_partition_bytes=10_000_000,
        target_partition_bytes_3d=baseline.estimated_bytes,
    )[3]

    assert plan.target_partitions == 2
    assert plan.reason == "3d-byte-budget-hysteresis"


def test_residual_partition_runtime_policy_includes_hysteresis_overrides(monkeypatch):
    policy = _residual_partition_runtime_policy(
        SimpleNamespace(
            runtime_info={
                "residual_partition_hysteresis_low": 0.9,
                "residual_partition_hysteresis_high": 1.1,
            }
        ),
        default_target_bytes=256,
        effective_nufft_workers=4,
    )

    assert policy["hysteresis_low_factor"] == pytest.approx(0.9)
    assert policy["hysteresis_high_factor"] == pytest.approx(1.1)

    monkeypatch.setenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_LOW", "0.85")
    monkeypatch.setenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_HIGH", "1.25")
    try:
        env_policy = _residual_partition_runtime_policy(
            SimpleNamespace(runtime_info={}),
            default_target_bytes=256,
            effective_nufft_workers=4,
        )
    finally:
        monkeypatch.delenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_LOW", raising=False)
        monkeypatch.delenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_HIGH", raising=False)

    assert env_policy["hysteresis_low_factor"] == pytest.approx(0.85)
    assert env_policy["hysteresis_high_factor"] == pytest.approx(1.25)


def test_residual_field_artifacts_preserve_current_saved_and_applied_semantics(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ResidualFieldArtifactStore(str(tmp_path))
    try:
        db.insert_point_data_batch(
            [
                {
                    "central_point_id": 10,
                    "coordinates": [0.1],
                    "dist_from_atom_center": [0.2],
                    "step_in_frac": [0.05],
                    "chunk_id": 3,
                    "grid_amplitude_initialized": 1,
                }
            ]
        )
        interval_id = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])[0]
        db.insert_interval_chunk_status_batch([(interval_id, 3, 0)])

        baseline = np.array([[10 + 0j, 0 + 0j], [10 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )

        from core.residual_field.contracts import ResidualFieldWorkUnit

        work_unit = ResidualFieldWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        )
        manifest = persist_residual_field_interval_chunk_result(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 10]),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        current, current_av, nrec, _ = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)
        assert manifest.chunk_id == 3
        assert interval_id in applied
        assert (interval_id, 3) not in set(db.get_unsaved_interval_chunks())
        np.testing.assert_allclose(current[:, 1], np.array([1 + 0j, 2 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j, 0.75 + 0j]))
        assert nrec == 5
    finally:
        db.close()


def test_residual_field_loader_reconstructs_grid_and_normalizes_values(tmp_path):
    store = ResidualFieldArtifactStore(str(tmp_path))
    payload = np.array([[10 + 0j, 2 + 0j], [10 + 0j, 4 + 0j]], dtype=np.complex128)
    store.save_chunk_payloads(
        3,
        amplitudes_payload=payload,
        amplitudes_average_payload=payload.copy(),
        reciprocal_point_count=2,
    )
    store.saver.save_data(
        {
            "ntotal_reciprocal_space_points": np.array([2], dtype=np.int64),
            "ntotal_reciprocal_points": np.array([2], dtype=np.int64),
        },
        store.saver.generate_filename(3, suffix="_amplitudes_ntotal_reciprocal_space_points"),
    )

    class FakePointDataProcessor:
        def generate_grid(
            self,
            *,
            chunk_id,
            dimensionality,
            step_in_frac,
            central_point,
            dist,
            central_point_id,
        ):
            return np.array([[0.1], [0.2]]), np.array([2])

    processor = SimpleNamespace(point_data_processor=FakePointDataProcessor())
    point_data_list = [
        {
            "central_point_id": 10,
            "coordinates": np.array([0.0]),
            "dist_from_atom_center": np.array([0.2]),
            "step_in_frac": np.array([0.1]),
        }
    ]
    data, amplitudes, rifft_grid = load_chunk_residual_field_and_grid(
        processor,
        chunk_id=3,
        point_data_list=point_data_list,
        rifft_saver=store.saver,
        logger=None,
    )

    assert "amplitudes" in data
    np.testing.assert_allclose(amplitudes[:, 1], np.array([1 + 0j, 2 + 0j]))
    assert rifft_grid.shape == (2, 2)
    np.testing.assert_allclose(rifft_grid[:, 1], np.array([10, 10]))


def test_residual_field_stage_delegates_to_execution(monkeypatch):
    captured = {}

    def fake_execute(*, workflow_parameters, structure, artifacts, client):
        captured["workflow_parameters"] = workflow_parameters
        captured["structure"] = structure
        captured["artifacts"] = artifacts
        captured["client"] = client

    monkeypatch.setattr(
        "core.residual_field.stage.run_residual_field_stage",
        fake_execute,
    )

    stage = ResidualFieldStage()
    artifacts = SimpleNamespace(db_manager="db", output_dir="/tmp/out")
    params = {"point_data_list": [], "supercell": np.array([4]), "reciprocal_space_intervals_all": []}

    result = stage.execute(
        workflow_parameters=SimpleNamespace(name="workflow"),
        structure=SimpleNamespace(name="structure"),
        artifacts=artifacts,
        client=None,
        scattering_parameters=params,
    )

    assert result is params
    assert captured["workflow_parameters"].name == "workflow"
    assert captured["structure"].name == "structure"
    assert captured["artifacts"] is artifacts


def test_residual_field_reducer_backend_resolution_is_mode_aware(monkeypatch):
    parameters = SimpleNamespace(runtime_info={})

    local_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=None,
    )
    local_state = local_backend.describe_runtime_state(
        output_dir="/tmp/out",
        scratch_root="/tmp/scratch",
    )
    assert local_backend.layout.kind == "local_restartable"
    assert local_state.durable_root == "/tmp/out"
    assert local_state.local_scratch_root == "/tmp/scratch"
    assert local_state.scattering_interval_transport.startswith(
        "direct in-process interval payload handoff"
    )
    assert local_state.checkpoint_policy.interval_artifacts == "optional_output"
    assert (
        local_state.checkpoint_policy.shard_checkpoints
        == "required_local_restart_state"
    )

    async_client = SimpleNamespace(loop=SimpleNamespace(asyncio_loop=object()))
    durable_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=async_client,
    )
    assert durable_backend.layout.kind == "durable_shared_restartable"
    durable_state = durable_backend.describe_runtime_state(
        output_dir="/tmp/out",
        scratch_root="/tmp/scratch",
    )
    assert durable_state.checkpoint_policy.interval_artifacts == "required_transport"
    assert (
        durable_state.checkpoint_policy.shard_checkpoints
        == "required_durable_checkpoint"
    )
    assert (
        durable_state.checkpoint_policy.worker_local_scratch_role
        == "temporary_staging_only"
    )
    assert local_state.durable_truth_unit == "committed_local_snapshot_generation"
    assert durable_state.durable_truth_unit == "committed_local_snapshot_generation"
    assert durable_state.live_state_storage_role == "owner-local-live-accumulator-with-shared-durable-generations"
    assert durable_state.durable_checkpoint_storage_role == "durable-shared-generation"
    assert durable_state.final_artifact_storage_role == "durable-final-chunk-artifact"

    monkeypatch.setenv("DASK_BACKEND", "local")
    local_async_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=async_client,
    )
    assert local_async_backend.layout.kind == "local_restartable"
    monkeypatch.delenv("DASK_BACKEND", raising=False)

    override_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(
            runtime_info={"residual_field_reducer_backend": "local_restartable"}
        ),
        client=async_client,
    )
    assert override_backend.layout.kind == "local_restartable"

    durable_root_override = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(
            runtime_info={"residual_shard_durable_root": "/tmp/shared-shards"}
        ),
        client=async_client,
    )
    assert (
        durable_root_override.describe_runtime_state(
            output_dir="/tmp/out",
            scratch_root="/tmp/scratch",
        ).committed_shard_root
        == "/tmp/shared-shards"
    )


def test_residual_field_async_local_handoff_reuses_scattered_interval_payloads(monkeypatch, tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    payload = IntervalTask(
        1,
        "All",
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([0.5 + 0.0j], dtype=np.complex128),
    )
    scatter_calls = []

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, data, broadcast=False, hash=True, **kwargs):
            scatter_calls.append((data, broadcast, hash))
            return f"future-{len(scatter_calls)}"

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterIntervalPayloadScatter()),
    )

    workflow_parameters = SimpleNamespace(runtime_info={})
    structure = SimpleNamespace(supercell=np.array([1]))
    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={1: payload},
    )

    try:
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=_FakeClient(),
        )
    except _StopAfterIntervalPayloadScatter:
        pass
    finally:
        monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert scatter_calls == []


def test_residual_field_async_stage_uses_owner_affinity_for_distributed_backend_when_enabled(
    monkeypatch,
    tmp_path,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    submits = []

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {
                "workers": {
                    "worker-a": {"resources": {"nufft": 1}},
                    "worker-b": {"resources": {"nufft": 1}},
                }
            }

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            submits.append(kwargs)
            if kwargs.get("key", "").startswith("residual-"):
                work_unit = args[0]
                return _FakeFuture(
                    ResidualFieldAccumulatorStatus(
                        artifact_key=work_unit.artifact_key,
                        chunk_id=work_unit.chunk_id,
                        parameter_digest=work_unit.parameter_digest,
                        interval_ids=work_unit.interval_ids,
                        partition_id=work_unit.partition_id,
                        contribution_reciprocal_point_count=1,
                        total_reciprocal_points=1,
                    )
                )
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            return _FakeFuture(func(*args, **call_kwargs))

    durable_backend = _FakeDistributedOwnerLocalBackend()

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: durable_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: np.array(
            [([0.0], [0.1], [0.05], 3)],
            dtype=[
                ("coordinates", object),
                ("dist_from_atom_center", object),
                ("step_in_frac", object),
                ("chunk_id", np.int64),
            ],
        ).view(np.recarray),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution._finalize_residual_field_chunks",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: {"chunk_id": kwargs["chunk_id"]},
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: {"chunk_id": kwargs["chunk_id"]},
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )
    workflow_parameters = SimpleNamespace(
        runtime_info={}
    )
    structure = SimpleNamespace(supercell=np.array([1]))

    run_residual_field_stage(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        client=_FakeClient(),
    )

    task_submit = next(kwargs for kwargs in submits if kwargs.get("key", "").startswith("residual-"))
    assert task_submit["workers"] == ["worker-a"]
    assert task_submit["allow_other_workers"] is False


def test_residual_field_async_stage_remaps_missing_owner_before_retry_submit(
    monkeypatch,
    tmp_path,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    task_submit_workers = []
    scheduler_states = [
        {"worker-a": {"resources": {"nufft": 1}}, "worker-b": {"resources": {"nufft": 1}}},
        {"worker-a": {"resources": {"nufft": 1}}, "worker-b": {"resources": {"nufft": 1}}},
        {"worker-b": {"resources": {"nufft": 1}}},
    ]

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def __init__(self):
            self.task_results = [
                None,
                ResidualFieldAccumulatorStatus(
                    artifact_key=work_unit.artifact_key,
                    chunk_id=work_unit.chunk_id,
                    parameter_digest=work_unit.parameter_digest,
                    interval_ids=work_unit.interval_ids,
                    partition_id=work_unit.partition_id,
                    contribution_reciprocal_point_count=1,
                    total_reciprocal_points=1,
                ),
            ]
            self.scheduler_call_count = 0

        def scheduler_info(self):
            index = min(self.scheduler_call_count, len(scheduler_states) - 1)
            self.scheduler_call_count += 1
            return {"workers": scheduler_states[index]}

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            if kwargs.get("key", "").startswith("residual-"):
                task_submit_workers.append(kwargs.get("workers"))
                return _FakeFuture(self.task_results.pop(0))
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            return _FakeFuture(func(*args, **call_kwargs))

    durable_backend = _FakeDistributedOwnerLocalBackend()

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: durable_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: np.array(
            [([0.0], [0.1], [0.05], 3)],
            dtype=[
                ("coordinates", object),
                ("dist_from_atom_center", object),
                ("step_in_frac", object),
                ("chunk_id", np.int64),
            ],
        ).view(np.recarray),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution._flush_local_reducer_targets_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution._inspect_owner_local_reducer_targets_or_raise",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: {"chunk_id": kwargs["chunk_id"]},
    )

    run_residual_field_stage(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        structure=SimpleNamespace(supercell=np.array([1])),
        artifacts=SimpleNamespace(
            db_manager=SimpleNamespace(
                get_unsaved_interval_chunks=lambda: [(1, 3)],
                get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
                db_path=str(tmp_path / "state.db"),
            ),
            padded_intervals=[{"h_range": (0.0, 0.0)}],
            output_dir=str(tmp_path),
            transient_interval_payloads={},
        ),
        client=_FakeClient(),
    )

    assert task_submit_workers == [["worker-a"], ["worker-b"]]


def test_distributed_owner_affinity_defaults_to_true():
    assert _distributed_owner_affinity_enabled(SimpleNamespace(runtime_info={})) is True


def test_real_distributed_backend_support_check_accepts_shared_generation_role():
    parameters = SimpleNamespace(runtime_info={})
    async_client = SimpleNamespace(loop=SimpleNamespace(asyncio_loop=object()))
    durable_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=parameters,
        client=async_client,
    )
    durable_state = durable_backend.describe_runtime_state(
        output_dir="/tmp/out",
        scratch_root="/tmp/scratch",
    )

    assert durable_state.durable_checkpoint_storage_role == "durable-shared-generation"
    assert (
        _distributed_owner_local_reducer_supported(
            durable_backend,
            reducer_runtime_state=durable_state,
        )
        is True
    )


def test_residual_field_distributed_stage_rejects_disabled_owner_affinity(
    monkeypatch,
    tmp_path,
):
    fake_backend = _FakeDistributedOwnerLocalBackend()

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [],
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [],
            get_point_data_for_chunk=lambda chunk_id: [],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )

    with pytest.raises(ValueError, match="requires owner affinity"):
        run_residual_field_stage(
            workflow_parameters=SimpleNamespace(
                runtime_info={"residual_distributed_owner_affinity": False}
            ),
            structure=SimpleNamespace(supercell=np.array([1])),
            artifacts=artifacts,
            client=SimpleNamespace(loop=SimpleNamespace(asyncio_loop=object())),
        )


def test_residual_field_distributed_stage_rejects_missing_owner_local_backend_support(
    monkeypatch,
    tmp_path,
):
    fake_backend = _FakeDistributedOwnerLocalBackend(supported=False)

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [],
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [],
            get_point_data_for_chunk=lambda chunk_id: [],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )

    with pytest.raises(RuntimeError, match="requires backend support"):
        run_residual_field_stage(
            workflow_parameters=SimpleNamespace(runtime_info={}),
            structure=SimpleNamespace(supercell=np.array([1])),
            artifacts=artifacts,
            client=SimpleNamespace(loop=SimpleNamespace(asyncio_loop=object())),
        )


def test_residual_field_sync_stage_filters_already_durable_local_work_units_before_dispatch(
    monkeypatch,
    tmp_path,
):
    durable_work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    pending_work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=2,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    fake_backend = _FakeLocalReducerBackend()
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)
    dispatched = []
    finalized = []

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution._build_task_reducer_backend",
        lambda backend: backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [durable_work_unit, pending_work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.run_residual_field_interval_chunk_task",
        lambda work_unit, *args, **kwargs: dispatched.append(work_unit.interval_ids or (work_unit.interval_id,)) or ResidualFieldAccumulatorStatus(
            artifact_key=work_unit.artifact_key,
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            interval_ids=work_unit.interval_ids,
            partition_id=work_unit.partition_id,
            contribution_reciprocal_point_count=1,
            total_reciprocal_points=1,
        ),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: finalized.append(kwargs["chunk_id"]) or {"chunk_id": kwargs["chunk_id"]},
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        fake_backend,
        "local_intervals_already_durable",
        lambda work_unit, *, output_dir: int(work_unit.interval_id or work_unit.interval_ids[0]) == 1,
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3), (2, 3)],
            get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )
    workflow_parameters = SimpleNamespace(runtime_info={})
    structure = SimpleNamespace(supercell=np.array([1]))

    run_residual_field_stage(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        client=None,
    )

    assert dispatched == [(2,)]
    assert finalized == [3]


def test_residual_field_sync_stage_filters_already_durable_distributed_work_units_before_dispatch(
    monkeypatch,
    tmp_path,
):
    durable_work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    pending_work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=2,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    fake_backend = _FakeDistributedOwnerLocalBackend()
    fake_backend.durable_by_target[(3, None)] = {1}
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)
    dispatched = []
    finalized = []

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution._build_task_reducer_backend",
        lambda backend: backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [durable_work_unit, pending_work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.run_residual_field_interval_chunk_task",
        lambda work_unit, *args, **kwargs: dispatched.append(work_unit.interval_ids or (work_unit.interval_id,)) or ResidualFieldAccumulatorStatus(
            artifact_key=work_unit.artifact_key,
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            interval_ids=work_unit.interval_ids,
            partition_id=work_unit.partition_id,
            contribution_reciprocal_point_count=1,
            total_reciprocal_points=1,
        ),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: finalized.append(kwargs["chunk_id"]) or {"chunk_id": kwargs["chunk_id"]},
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        fake_backend,
        "local_intervals_already_durable",
        lambda work_unit, *, output_dir: int(work_unit.interval_id or work_unit.interval_ids[0]) == 1,
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3), (2, 3)],
            get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )

    run_residual_field_stage(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        structure=SimpleNamespace(supercell=np.array([1])),
        artifacts=artifacts,
        client=None,
    )

    assert dispatched == [(2,)]
    assert finalized == [3]


def test_residual_field_async_local_stage_uses_owner_affinity_per_unique_reducer_target(
    monkeypatch,
    tmp_path,
):
    work_units = [
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=0, point_start=0, point_stop=1),
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=2,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=0, point_start=0, point_stop=1),
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=3,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ).with_partition(partition_id=1, point_start=1, point_stop=2),
    ]

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    submits = []

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {
                "workers": {
                    "worker-a": {"resources": {"nufft": 1}},
                    "worker-b": {"resources": {"nufft": 1}},
                }
            }

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            submits.append((func, kwargs))
            if kwargs.get("key", "").startswith("residual-"):
                work_unit = args[0]
                return _FakeFuture(
                    ResidualFieldAccumulatorStatus(
                        artifact_key=work_unit.artifact_key,
                        chunk_id=work_unit.chunk_id,
                        parameter_digest=work_unit.parameter_digest,
                        interval_ids=work_unit.interval_ids,
                        partition_id=work_unit.partition_id,
                        contribution_reciprocal_point_count=1,
                        total_reciprocal_points=1,
                    )
                )
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            return _FakeFuture(func(*args, **call_kwargs))

    fake_backend = _FakeLocalReducerBackend()
    rec = np.array(
        [
            ([0.0], [0.1], [0.05], 3),
            ([1.0], [0.2], [0.05], 3),
        ],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: work_units,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_adaptive_partition_plan",
        lambda *args, **kwargs: {
            3: SimpleNamespace(
                target_partitions=1,
                dimensionality=1,
                point_count=2,
                estimated_rifft_points=2,
                estimated_bytes=1,
                target_partition_bytes=1,
                reason="test",
            )
        },
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: {"chunk_id": kwargs["chunk_id"]},
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3), (2, 3), (3, 3)],
            get_point_data_for_chunk=lambda chunk_id: [
                {
                    "chunk_id": 3,
                    "coordinates": np.array([0.0]),
                    "dist_from_atom_center": np.array([0.1]),
                    "step_in_frac": np.array([0.05]),
                },
                {
                    "chunk_id": 3,
                    "coordinates": np.array([1.0]),
                    "dist_from_atom_center": np.array([0.2]),
                    "step_in_frac": np.array([0.05]),
                },
            ],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )
    workflow_parameters = SimpleNamespace(runtime_info={})
    structure = SimpleNamespace(supercell=np.array([1]))

    try:
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=_FakeClient(),
        )
    finally:
        monkeypatch.delenv("DASK_BACKEND", raising=False)

    task_submits = [kwargs for _func, kwargs in submits if kwargs.get("key", "").startswith("residual-")]
    assert [kwargs["workers"] for kwargs in task_submits] == [
        ["worker-a"],
        ["worker-a"],
        ["worker-b"],
    ]
    assert [kwargs["allow_other_workers"] for kwargs in task_submits] == [False, False, False]


def test_residual_partition_runtime_policy_reads_hysteresis_from_runtime_info_and_env(
    monkeypatch,
):
    runtime_policy = _residual_partition_runtime_policy(
        SimpleNamespace(
            runtime_info={
                "residual_partition_hysteresis_low": "0.7",
                "residual_partition_hysteresis_high": "1.3",
            }
        ),
        default_target_bytes=256,
        effective_nufft_workers=4,
    )

    assert runtime_policy["hysteresis_low_factor"] == pytest.approx(0.7)
    assert runtime_policy["hysteresis_high_factor"] == pytest.approx(1.3)

    monkeypatch.setenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_LOW", "0.6")
    monkeypatch.setenv("MOSAIC_RESIDUAL_PARTITION_HYSTERESIS_HIGH", "1.4")
    env_policy = _residual_partition_runtime_policy(
        SimpleNamespace(runtime_info={}),
        default_target_bytes=256,
        effective_nufft_workers=4,
    )

    assert env_policy["hysteresis_low_factor"] == pytest.approx(0.6)
    assert env_policy["hysteresis_high_factor"] == pytest.approx(1.4)


def test_residual_field_async_stage_passes_hysteresis_policy_into_partition_planner_and_logs_band(
    monkeypatch,
    tmp_path,
    caplog,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {"workers": {"worker-a": {"resources": {"nufft": 1}}}}

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            if kwargs.get("key", "").startswith("residual-"):
                planned_work_unit = args[0]
                return _FakeFuture(
                    ResidualFieldAccumulatorStatus(
                        artifact_key=planned_work_unit.artifact_key,
                        chunk_id=planned_work_unit.chunk_id,
                        parameter_digest=planned_work_unit.parameter_digest,
                        interval_ids=planned_work_unit.interval_ids,
                        partition_id=planned_work_unit.partition_id,
                        contribution_reciprocal_point_count=1,
                        total_reciprocal_points=1,
                    )
                )
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            return _FakeFuture(func(*args, **call_kwargs))

    planner_kwargs: dict[str, object] = {}
    fake_backend = _FakeLocalReducerBackend()

    def _capture_partition_plan(*args, **kwargs):
        planner_kwargs.update(kwargs)
        return {
            3: SimpleNamespace(
                target_partitions=2,
                dimensionality=2,
                point_count=2,
                estimated_rifft_points=2,
                estimated_bytes=112,
                target_partition_bytes=100,
                reason="byte-budget-hysteresis",
                rifft_points_per_atom=(1, 1),
                partition_rifft_points=(1, 1),
                partition_imbalance_ratio=1.0,
            )
        }

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: np.array(
            [
                ([0.0, 0.0], 3),
                ([1.0, 1.0], 3),
            ],
            dtype=[("coordinates", object), ("chunk_id", np.int64)],
        ).view(np.recarray),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_adaptive_partition_plan",
        _capture_partition_plan,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: {"chunk_id": kwargs["chunk_id"]},
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [
                {"chunk_id": 3, "coordinates": np.array([0.0, 0.0])},
                {"chunk_id": 3, "coordinates": np.array([1.0, 1.0])},
            ],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )

    caplog.set_level(logging.INFO, logger="core.residual_field.execution")
    try:
        run_residual_field_stage(
            workflow_parameters=SimpleNamespace(
                runtime_info={
                    "residual_partition_hysteresis_low": 0.7,
                    "residual_partition_hysteresis_high": 1.3,
                }
            ),
            structure=SimpleNamespace(supercell=np.array([1])),
            artifacts=artifacts,
            client=_FakeClient(),
        )
    finally:
        monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert planner_kwargs["hysteresis_low_factor"] == pytest.approx(0.7)
    assert planner_kwargs["hysteresis_high_factor"] == pytest.approx(1.3)
    assert any(
        "Residual-field partition plan | chunk=3" in message
        and "hysteresis_band=70-130" in message
        and "reason=byte-budget-hysteresis" in message
        for message in caplog.messages
    )


def test_residual_field_async_distributed_stage_flushes_validates_and_logs_metrics(
    monkeypatch,
    tmp_path,
    caplog,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    submits = []
    finalized = []
    fake_backend = _FakeDistributedOwnerLocalBackend(
        metrics_by_target={
            (3, None): {
                "total_checkpoint_bytes_written": 128,
                "total_checkpoint_writes": 2,
                "total_checkpoint_wall_seconds": 0.5,
            }
        }
    )
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {"workers": {"worker-a": {"resources": {"nufft": 1}}}}

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            submits.append((func, kwargs))
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            return _FakeFuture(func(*args, **call_kwargs))

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.run_residual_field_interval_chunk_task",
        lambda work_unit, *args, **kwargs: fake_backend.accept_local_contribution(
            work_unit,
            grid_shape_nd=np.array([[1]], dtype=np.int64),
            total_reciprocal_points=1,
            contribution_reciprocal_points=1,
            amplitudes_delta=np.array([1.0 + 0.0j]),
            amplitudes_average=np.array([1.0 + 0.0j]),
            point_ids=np.array([0], dtype=np.int64),
            output_dir=str(tmp_path),
            scratch_root=str(tmp_path / "scratch"),
            db_path=str(tmp_path / "state.db"),
            total_expected_partials=1,
        )
        or ResidualFieldAccumulatorStatus(
            artifact_key=work_unit.artifact_key,
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            interval_ids=work_unit.interval_ids,
            partition_id=work_unit.partition_id,
            contribution_reciprocal_point_count=1,
            total_reciprocal_points=1,
        ),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.flush_process_local_residual_reducer_target",
        lambda template_backend, **kwargs: fake_backend.flush_local_reducer_target(**kwargs),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.inspect_process_local_residual_reducer_target",
        lambda template_backend, **kwargs: fake_backend.inspect_local_reducer_target(**kwargs),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda template_backend, **kwargs: finalized.append(kwargs["chunk_id"]) or {"chunk_id": kwargs["chunk_id"]},
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )

    with caplog.at_level(logging.INFO):
        run_residual_field_stage(
            workflow_parameters=SimpleNamespace(runtime_info={}),
            structure=SimpleNamespace(supercell=np.array([1])),
            artifacts=artifacts,
            client=_FakeClient(),
        )

    assert finalized == [3]
    assert any("finalize checkpoints | backend=durable_shared_restartable" in rec.message for rec in caplog.records)
    assert any("Residual-field partition report | target=" in rec.message for rec in caplog.records)
    assert any(
        kwargs.get("workers") == ["worker-a"] and kwargs.get("allow_other_workers") is False
        for _func, kwargs in submits
        if kwargs.get("key", "").startswith("residual-")
    )


def test_residual_field_async_distributed_stage_remaps_missing_owner_for_finalize_steps(
    monkeypatch,
    tmp_path,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    scheduler_states = [
        {"worker-a": {"resources": {"nufft": 1}}, "worker-b": {"resources": {"nufft": 1}}},
        {"worker-a": {"resources": {"nufft": 1}}, "worker-b": {"resources": {"nufft": 1}}},
        {"worker-b": {"resources": {"nufft": 1}}},
        {"worker-b": {"resources": {"nufft": 1}}},
        {"worker-b": {"resources": {"nufft": 1}}},
    ]
    submits = []
    fake_backend = _FakeDistributedOwnerLocalBackend()
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)

    def _flush_helper(template_backend, **kwargs):
        return fake_backend.flush_local_reducer_target(**kwargs)

    def _inspect_helper(template_backend, **kwargs):
        return fake_backend.inspect_local_reducer_target(**kwargs)

    def _finalize_helper(template_backend, **kwargs):
        return {"chunk_id": kwargs["chunk_id"]}

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def __init__(self):
            self.scheduler_call_count = 0

        def scheduler_info(self):
            index = min(self.scheduler_call_count, len(scheduler_states) - 1)
            self.scheduler_call_count += 1
            return {"workers": scheduler_states[index]}

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            submits.append((func.__name__, kwargs))
            call_kwargs = dict(kwargs)
            for reserved_key in (
                "key",
                "pure",
                "workers",
                "allow_other_workers",
                "resources",
                "retries",
            ):
                call_kwargs.pop(reserved_key, None)
            if kwargs.get("key", "").startswith("residual-"):
                return _FakeFuture(
                    fake_backend.accept_local_contribution(
                        args[0],
                        grid_shape_nd=np.array([[1]], dtype=np.int64),
                        total_reciprocal_points=1,
                        contribution_reciprocal_points=1,
                        amplitudes_delta=np.array([1.0 + 0.0j]),
                        amplitudes_average=np.array([1.0 + 0.0j]),
                        point_ids=np.array([0], dtype=np.int64),
                        output_dir=str(tmp_path),
                        scratch_root=str(tmp_path / "scratch"),
                        db_path=str(tmp_path / "state.db"),
                        total_expected_partials=2,
                    )
                    or ResidualFieldAccumulatorStatus(
                        artifact_key=args[0].artifact_key,
                        chunk_id=args[0].chunk_id,
                        parameter_digest=args[0].parameter_digest,
                        interval_ids=args[0].interval_ids,
                        partition_id=args[0].partition_id,
                        contribution_reciprocal_point_count=1,
                        total_reciprocal_points=1,
                    )
                )
            return _FakeFuture(func(*args, **call_kwargs))

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, future.result()) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.flush_process_local_residual_reducer_target",
        _flush_helper,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.inspect_process_local_residual_reducer_target",
        _inspect_helper,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        _finalize_helper,
    )

    run_residual_field_stage(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        structure=SimpleNamespace(supercell=np.array([1])),
        artifacts=SimpleNamespace(
            db_manager=SimpleNamespace(
                get_unsaved_interval_chunks=lambda: [(1, 3)],
                get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
                db_path=str(tmp_path / "state.db"),
            ),
            padded_intervals=[{"h_range": (0.0, 0.0)}],
            output_dir=str(tmp_path),
            transient_interval_payloads={},
        ),
        client=_FakeClient(),
    )

    assert any(
        func_name == "_flush_helper" and kwargs.get("workers") == ["worker-b"]
        for func_name, kwargs in submits
    )
    assert any(
        func_name == "_inspect_helper" and kwargs.get("workers") == ["worker-b"]
        for func_name, kwargs in submits
    )
    assert any(
        func_name == "_finalize_helper" and kwargs.get("workers") == ["worker-b"]
        for func_name, kwargs in submits
    )


def test_residual_field_async_local_max_inflight_is_capped_to_worker_capacity(monkeypatch):
    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scheduler_info(self):
            return {
                "workers": {
                    "worker-1": {"resources": {"nufft": 1}},
                    "worker-2": {"resources": {"nufft": 1}},
                    "worker-3": {"resources": {"nufft": 1}},
                    "worker-4": {"resources": {"nufft": 1}},
                }
            }

    monkeypatch.setenv("DASK_BACKEND", "local")
    try:
        assert _cap_async_max_inflight(client=_FakeClient(), requested=5000) == 4
    finally:
        monkeypatch.delenv("DASK_BACKEND", raising=False)


def test_task_reducer_backend_clone_drops_live_local_accumulators():
    backend = build_residual_field_reducer_backend(
        "local_restartable",
        shard_storage_root_override="/tmp/shared-shards",
        local_accumulator_max_ram_bytes=123456,
    )
    backend._local_accumulators[(3, "abc123")] = object()

    task_backend = _build_task_reducer_backend(backend)

    assert task_backend is not backend
    assert task_backend.layout.kind == "local_restartable"
    assert task_backend.shard_storage_root_override == "/tmp/shared-shards"
    assert task_backend.local_accumulator_max_ram_bytes == 123456
    assert task_backend._local_accumulators == {}


def test_residual_field_async_stage_logs_main_process_progress_when_enabled(
    monkeypatch,
    tmp_path,
    caplog,
):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, data, **kwargs):
            return data

        def submit(self, func, *args, **kwargs):
            return _FakeFuture(
                ResidualFieldAccumulatorStatus(
                    artifact_key=work_unit.artifact_key,
                    chunk_id=3,
                    parameter_digest="abc123",
                    interval_ids=(1,),
                    contribution_reciprocal_point_count=1,
                    total_reciprocal_points=1,
                )
            )

    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    fake_backend = _FakeLocalReducerBackend()
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.yield_futures_with_results",
        lambda futures, client: ((future, True) for future in futures),
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [
                {
                    "chunk_id": 3,
                    "coordinates": np.array([0.0]),
                    "dist_from_atom_center": np.array([0.1]),
                    "step_in_frac": np.array([0.05]),
                }
            ],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )
    workflow_parameters = SimpleNamespace(runtime_info={})
    structure = SimpleNamespace(supercell=np.array([1]))

    configure_progress(force_progress=None, task_progress=True)
    caplog.set_level(logging.INFO, logger="core.residual_field.execution")
    try:
        run_residual_field_stage(
            workflow_parameters=workflow_parameters,
            structure=structure,
            artifacts=artifacts,
            client=_FakeClient(),
        )
    finally:
        configure_progress(force_progress=None, task_progress=None)
        monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert any("Residual-field start" in message for message in caplog.messages)
    assert any("Residual-field queue" in message for message in caplog.messages)
    assert any("Residual-field progress" in message for message in caplog.messages)
    assert fake_backend.accepted == []


def test_residual_field_sync_stage_uses_worker_owned_local_reducer_boundary(
    monkeypatch,
    tmp_path,
):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )
    fake_backend = _FakeLocalReducerBackend()
    rec = np.array(
        [([0.0], [0.1], [0.05], 3)],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
            ("chunk_id", np.int64),
        ],
    ).view(np.recarray)
    finalized = []

    def _forbid_accept_partial(*args, **kwargs):
        raise AssertionError("driver-side accept_partial should not run in worker-owned mode")

    fake_backend.accept_partial = _forbid_accept_partial

    monkeypatch.setattr(
        "core.residual_field.execution.resolve_residual_field_reducer_backend",
        lambda *args, **kwargs: fake_backend,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.resolve_worker_scratch_root",
        lambda preferred, stage: str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.build_residual_field_work_units",
        lambda *args, **kwargs: [work_unit],
    )
    monkeypatch.setattr(
        "core.residual_field.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.point_list_to_recarray",
        lambda *args, **kwargs: rec,
    )
    monkeypatch.setattr(
        "core.residual_field.execution.run_residual_field_interval_chunk_task",
        lambda *args, **kwargs: ResidualFieldAccumulatorStatus(
            artifact_key=work_unit.artifact_key,
            chunk_id=3,
            parameter_digest="abc123",
            interval_ids=(1,),
            contribution_reciprocal_point_count=1,
            total_reciprocal_points=1,
        ),
    )
    monkeypatch.setattr(
        "core.residual_field.execution.finalize_process_local_residual_chunk",
        lambda *args, **kwargs: finalized.append(kwargs["chunk_id"]) or None,
    )
    monkeypatch.setattr(
        "core.residual_field.execution._validate_local_durable_coverage_or_raise",
        lambda **kwargs: None,
    )

    artifacts = SimpleNamespace(
        db_manager=SimpleNamespace(
            get_unsaved_interval_chunks=lambda: [(1, 3)],
            get_point_data_for_chunk=lambda chunk_id: [{"chunk_id": 3}],
            db_path=str(tmp_path / "state.db"),
        ),
        padded_intervals=[{"h_range": (0.0, 0.0)}],
        output_dir=str(tmp_path),
        transient_interval_payloads={},
    )
    workflow_parameters = SimpleNamespace(
        runtime_info={"residual_local_owner_reducer": True}
    )
    structure = SimpleNamespace(supercell=np.array([1]))

    run_residual_field_stage(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        client=None,
    )

    assert finalized == [3]


def test_residual_field_interval_chunk_task_uses_batched_inverse(monkeypatch, tmp_path):
    interval_path = tmp_path / "interval_1.npz"
    np.savez(
        interval_path,
        irecip_id=np.array([1], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: calls.__setitem__("count", calls["count"] + 1) or np.array([[5.0 + 0.0j], [6.0 + 0.0j]]),
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        interval_path,
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert calls["count"] == 1
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([5.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))


def test_residual_field_interval_chunk_task_accepts_direct_interval_payloads(monkeypatch, tmp_path):
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    reducer_backend = _CapturingReducerBackend("manifest")

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[7.0 + 0.0j], [8.0 + 0.0j]]),
    )

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([7.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([8.0 + 0.0j]))


def test_residual_field_interval_chunk_task_returns_small_status_for_local_backend(
    monkeypatch,
    tmp_path,
):
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    local_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        client=None,
    )
    captured = {}
    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[9.0 + 0.0j], [10.0 + 0.0j]]),
    )

    class _WorkerLocalBackend:
        def local_intervals_already_durable(self, work_unit, *, output_dir):
            return False

        def accept_local_contribution(self, work_unit, **kwargs):
            captured["accepted"] = ResidualFieldLocalAccumulatorPartial(
                work_unit=work_unit,
                point_ids=kwargs["point_ids"],
                grid_shape_nd=kwargs["grid_shape_nd"],
                total_reciprocal_points=kwargs["total_reciprocal_points"],
                contribution_reciprocal_points=kwargs["contribution_reciprocal_points"],
                amplitudes_delta=kwargs["amplitudes_delta"],
                amplitudes_average=kwargs["amplitudes_average"],
            )

    monkeypatch.setattr(
        "core.residual_field.tasks.get_process_local_residual_field_backend",
        lambda template_backend: _WorkerLocalBackend(),
    )

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        scratch_root=str(tmp_path / "scratch"),
        reducer_backend=local_backend,
        total_expected_partials=4,
        owner_local_reducer=True,
        quiet_logs=True,
    )

    assert isinstance(result, ResidualFieldAccumulatorStatus)
    np.testing.assert_allclose(captured["accepted"].amplitudes_delta, np.array([9.0 + 0.0j]))
    np.testing.assert_allclose(captured["accepted"].amplitudes_average, np.array([10.0 + 0.0j]))
    assert not (tmp_path / "residual_shards").exists()


def test_residual_field_interval_chunk_task_returns_small_status_for_distributed_owner_local_backend(
    monkeypatch,
    tmp_path,
):
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    distributed_backend = _FakeDistributedOwnerLocalBackend()
    captured = {}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[9.0 + 0.0j], [10.0 + 0.0j]]),
    )

    class _WorkerDistributedBackend:
        def local_intervals_already_durable(self, work_unit, *, output_dir):
            return False

        def accept_local_contribution(self, work_unit, **kwargs):
            captured["accepted"] = ResidualFieldLocalAccumulatorPartial(
                work_unit=work_unit,
                point_ids=kwargs["point_ids"],
                grid_shape_nd=kwargs["grid_shape_nd"],
                total_reciprocal_points=kwargs["total_reciprocal_points"],
                contribution_reciprocal_points=kwargs["contribution_reciprocal_points"],
                amplitudes_delta=kwargs["amplitudes_delta"],
                amplitudes_average=kwargs["amplitudes_average"],
            )

    monkeypatch.setattr(
        "core.residual_field.tasks.get_process_local_residual_field_backend",
        lambda template_backend: _WorkerDistributedBackend(),
    )

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        scratch_root=str(tmp_path / "scratch"),
        reducer_backend=distributed_backend,
        total_expected_partials=4,
        owner_local_reducer=True,
        quiet_logs=True,
    )

    assert isinstance(result, ResidualFieldAccumulatorStatus)
    np.testing.assert_allclose(captured["accepted"].amplitudes_delta, np.array([9.0 + 0.0j]))
    np.testing.assert_allclose(captured["accepted"].amplitudes_average, np.array([10.0 + 0.0j]))


def test_residual_field_interval_chunk_task_returns_small_status_for_owner_local_reducer(
    monkeypatch,
    tmp_path,
):
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    local_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        client=None,
    )
    captured = {}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[9.0 + 0.0j], [10.0 + 0.0j]]),
    )

    class _WorkerLocalBackend:
        def accept_local_contribution(self, work_unit, **kwargs):
            captured["accepted"] = ResidualFieldLocalAccumulatorPartial(
                work_unit=work_unit,
                point_ids=kwargs["point_ids"],
                grid_shape_nd=kwargs["grid_shape_nd"],
                total_reciprocal_points=kwargs["total_reciprocal_points"],
                contribution_reciprocal_points=kwargs["contribution_reciprocal_points"],
                amplitudes_delta=kwargs["amplitudes_delta"],
                amplitudes_average=kwargs["amplitudes_average"],
            )

    monkeypatch.setattr(
        "core.residual_field.tasks.get_process_local_residual_field_backend",
        lambda template_backend: _WorkerLocalBackend(),
    )

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        scratch_root=str(tmp_path / "scratch"),
        reducer_backend=local_backend,
        total_expected_partials=4,
        owner_local_reducer=True,
        quiet_logs=True,
    )

    assert isinstance(result, ResidualFieldAccumulatorStatus)
    np.testing.assert_allclose(
        captured["accepted"].amplitudes_delta,
        np.array([9.0 + 0.0j]),
    )
    np.testing.assert_allclose(
        captured["accepted"].amplitudes_average,
        np.array([10.0 + 0.0j]),
    )


def test_residual_field_interval_chunk_task_slices_partition_atoms_for_owner_local_reducer(
    monkeypatch,
    tmp_path,
):
    atoms = np.array(
        [
            ([0.0], [0.1], [0.05]),
            ([1.0], [0.2], [0.05]),
        ],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    local_backend = resolve_residual_field_reducer_backend(
        workflow_parameters=SimpleNamespace(runtime_info={}),
        client=None,
    )
    captured = {}

    def _capture_grid(chunk_data):
        captured["chunk_data_len"] = len(chunk_data)
        return np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        _capture_grid,
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[9.0 + 0.0j], [10.0 + 0.0j]]),
    )

    class _WorkerLocalBackend:
        def local_intervals_already_durable(self, work_unit, *, output_dir):
            return False

        def accept_local_contribution(self, work_unit, **kwargs):
            return None

    monkeypatch.setattr(
        "core.residual_field.tasks.get_process_local_residual_field_backend",
        lambda template_backend: _WorkerLocalBackend(),
    )

    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    ).with_partition(partition_id=1, point_start=1, point_stop=2)

    result = run_residual_field_interval_chunk_task(
        work_unit,
        IntervalTask(
            1,
            "All",
            np.array([[0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
        ),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        scratch_root=str(tmp_path / "scratch"),
        reducer_backend=local_backend,
        total_expected_partials=4,
        owner_local_reducer=True,
        quiet_logs=True,
    )

    assert isinstance(result, ResidualFieldAccumulatorStatus)
    assert captured["chunk_data_len"] == 1


def test_residual_field_interval_chunk_task_uses_super_batch_for_same_geometry(monkeypatch, tmp_path):
    interval_path_1 = tmp_path / "interval_1.npz"
    interval_path_2 = tmp_path / "interval_2.npz"
    for path, interval_id, q_amp in (
        (interval_path_1, 1, 2.0 + 0.0j),
        (interval_path_2, 2, 4.0 + 0.0j),
    ):
        np.savez(
            path,
            irecip_id=np.array([interval_id], dtype=np.int64),
            element=np.array(["All"]),
            q_grid=np.array([[0.0]], dtype=np.float64),
            q_amp=np.array([q_amp]),
            q_amp_av=np.array([1.0 + 0.0j]),
        )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    calls = {"count": 0}

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: calls.__setitem__("count", calls["count"] + 1)
        or np.array(
            [
                [1.0 + 0.0j],
                [2.0 + 0.0j],
                [3.0 + 0.0j],
                [4.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(1, 2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        (interval_path_1, interval_path_2),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert calls["count"] == 1
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([4.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))


def test_residual_field_interval_chunk_task_groups_mixed_q_grid_batches(monkeypatch, tmp_path):
    interval_path_1 = tmp_path / "interval_1.npz"
    interval_path_2 = tmp_path / "interval_2.npz"
    np.savez(
        interval_path_1,
        irecip_id=np.array([1], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    np.savez(
        interval_path_2,
        irecip_id=np.array([2], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[1.0]], dtype=np.float64),
        q_amp=np.array([4.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}
    calls = []

    monkeypatch.setattr(
        "core.residual_field.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )

    def fake_super_batch(**kwargs):
        calls.append(np.asarray(kwargs["q_coords"]).copy())
        if np.allclose(kwargs["q_coords"], np.array([[0.0]], dtype=np.float64)):
            return np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128)
        return np.array([[3.0 + 0.0j], [4.0 + 0.0j]], dtype=np.complex128)

    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        fake_super_batch,
    )
    reducer_backend = _CapturingReducerBackend("manifest")

    result = run_residual_field_interval_chunk_task(
        ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(1, 2),
            chunk_id=3,
            parameter_digest="abc123",
            output_dir=str(tmp_path),
        ),
        (interval_path_1, interval_path_2),
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        reducer_backend=reducer_backend,
        quiet_logs=True,
    )

    assert result == "manifest"
    assert len(calls) == 2
    captured.update(reducer_backend.calls[0])
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([4.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 + 0.0j]))
