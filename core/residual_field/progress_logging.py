"""Progress-bar formatting and partition-effectiveness logging for the residual-field stage.

Pure leaf utilities: formatting, metrics aggregation, and logger calls.
None of these import from core.residual_field.execution.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.planning import (
    _RESIDUAL_GRID_VALUE_BYTES_PER_POINT,
    _weighted_partition_split,
)

__all__ = [
    "_build_planned_target_metrics",
    "_format_elapsed_eta",
    "_format_progress_bar",
    "_log_async_residual_progress",
    "_log_owner_local_finalize_metrics",
    "_log_partition_effectiveness_report",
    "_planned_partition_imbalance_ratio",
    "_should_log_async_progress",
    "_work_unit_interval_label",
]

logger = logging.getLogger(__name__)


def _format_progress_bar(count: int, total: int, *, width: int = 20) -> str:
    total = max(int(total), 1)
    count = max(0, min(int(count), total))
    filled = int(round((count / float(total)) * width))
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def _work_unit_interval_label(work_unit: ResidualFieldWorkUnit) -> str:
    interval_ids = work_unit.interval_ids or (
        (work_unit.interval_id,) if work_unit.interval_id is not None else ()
    )
    return ",".join(str(interval_id) for interval_id in interval_ids) if interval_ids else "n/a"


def _format_elapsed_eta(elapsed_seconds: float, completed: int, total: int) -> str:
    """Format elapsed time and estimated remaining time."""
    def _fmt(seconds: float) -> str:
        seconds = max(0.0, seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60.0
        return f"{hours:.1f}h"

    parts = [f"elapsed={_fmt(elapsed_seconds)}"]
    if completed > 0 and completed < total:
        rate = completed / max(elapsed_seconds, 0.001)
        remaining = (total - completed) / rate
        parts.append(f"eta={_fmt(remaining)}")
        parts.append(f"rate={rate:.1f}/s")
    return " | ".join(parts)


def _log_async_residual_progress(
    *,
    enabled: bool,
    event: str,
    work_unit: ResidualFieldWorkUnit,
    completed: int,
    total: int,
    submitted: int,
    running: int,
    detail: str | None = None,
    start_time: float | None = None,
) -> None:
    if not enabled:
        return
    current = int(submitted if event == "queue" else completed)
    progress_bar_text = _format_progress_bar(current, total)
    percent = (100.0 * current / float(max(int(total), 1)))
    suffix = f" | {detail}" if detail else ""
    timing = ""
    if start_time is not None and event == "progress":
        timing = f" | {_format_elapsed_eta(time.monotonic() - start_time, completed, total)}"
    logger.info(
        "Residual-field %s %s %d/%d (%.0f%%) | running=%d | chunk=%d | intervals=%s%s%s",
        event,
        progress_bar_text,
        current,
        int(total),
        percent,
        int(running),
        int(work_unit.chunk_id),
        _work_unit_interval_label(work_unit),
        timing,
        suffix,
    )


def _should_log_async_progress(
    *,
    phase: str,
    count: int,
    total: int,
    force: bool = False,
) -> bool:
    if force or total <= 0:
        return True
    if count <= 1 or count >= total:
        return True
    target_updates = 4 if phase == "queue" else 20
    stride = max(1, int(math.ceil(total / float(target_updates))))
    return count % stride == 0


def _planned_partition_imbalance_ratio(
    *,
    rifft_points_per_atom: tuple[int, ...],
    target_partitions: int,
) -> float:
    weights = np.asarray(rifft_points_per_atom, dtype=np.int64)
    if weights.size == 0 or target_partitions <= 1:
        return 1.0
    selections = _weighted_partition_split(
        np.arange(weights.shape[0], dtype=np.int64),
        weights,
        int(target_partitions),
    )
    partition_weights = [
        int(np.sum(weights[selection], dtype=np.int64))
        for selection in selections
        if selection.size > 0
    ]
    if not partition_weights:
        return 1.0
    min_weight = min(partition_weights)
    max_weight = max(partition_weights)
    if min_weight <= 0:
        return float("inf") if max_weight > 0 else 1.0
    return float(max_weight) / float(min_weight)


def _build_planned_target_metrics(
    partition_plans: dict[int, object],
) -> dict[tuple[int, int | None], dict[str, object]]:
    planned_metrics: dict[tuple[int, int | None], dict[str, object]] = {}
    for chunk_id, plan in partition_plans.items():
        weights = np.asarray(
            getattr(
                plan,
                "rifft_points_per_atom",
                np.ones(int(plan.point_count), dtype=np.int64),
            ),
            dtype=np.int64,
        )
        target_partitions = int(getattr(plan, "target_partitions", 1))
        if target_partitions <= 1:
            planned_metrics[(int(chunk_id), None)] = {
                "planned_partition_count": 1,
                "planned_rifft_points": int(np.sum(weights, dtype=np.int64)),
                "planned_estimated_bytes": int(getattr(plan, "estimated_bytes", 0)),
                "target_partition_bytes": int(getattr(plan, "target_partition_bytes", 0)),
                "planned_imbalance_ratio": 1.0,
            }
            continue
        selections = _weighted_partition_split(
            np.arange(weights.shape[0], dtype=np.int64),
            weights,
            target_partitions,
        )
        imbalance_ratio = _planned_partition_imbalance_ratio(
            rifft_points_per_atom=tuple(int(value) for value in weights.tolist()),
            target_partitions=target_partitions,
        )
        for partition_id, selection in enumerate(selections):
            planned_rifft_points = int(np.sum(weights[selection], dtype=np.int64))
            planned_estimated_bytes = (
                planned_rifft_points * int(_RESIDUAL_GRID_VALUE_BYTES_PER_POINT)
            ) + (int(selection.size) * int(getattr(plan, "dimensionality", 1)) * 8)
            planned_metrics[(int(chunk_id), int(partition_id))] = {
                "planned_partition_count": int(target_partitions),
                "planned_rifft_points": int(planned_rifft_points),
                "planned_estimated_bytes": int(planned_estimated_bytes),
                "target_partition_bytes": int(getattr(plan, "target_partition_bytes", 0)),
                "planned_imbalance_ratio": float(imbalance_ratio),
            }
    return planned_metrics


def _log_partition_effectiveness_report(
    *,
    planned_target_metrics: dict[tuple[int, int | None], dict[str, object]],
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None],
) -> None:
    if not planned_target_metrics or not inspected_target_states:
        return
    for target_key in sorted(planned_target_metrics):
        planned = planned_target_metrics.get(target_key) or {}
        actual = inspected_target_states.get(target_key) or {}
        checkpoint_metrics = actual.get("checkpoint_metrics") if isinstance(actual, dict) else {}
        if not isinstance(checkpoint_metrics, dict):
            checkpoint_metrics = {}
        actual_checkpoint_bytes = int(
            checkpoint_metrics.get(
                "latest_checkpoint_bytes_written",
                checkpoint_metrics.get("total_checkpoint_bytes_written", 0),
            )
        )
        actual_checkpoint_writes = int(checkpoint_metrics.get("total_checkpoint_writes", 0))
        actual_checkpoint_wall = float(checkpoint_metrics.get("total_checkpoint_wall_seconds", 0.0))
        target_partition_bytes = int(planned.get("target_partition_bytes", 0))
        logger.info(
            "Residual-field partition report | target=%s | planned_rifft_points=%d | planned_bytes=%d | target_bytes=%d | actual_checkpoint_bytes=%d | actual_checkpoint_writes=%d | actual_checkpoint_wall=%.3fs | imbalance=%.3f | over_budget=%s",
            target_key,
            int(planned.get("planned_rifft_points", 0)),
            int(planned.get("planned_estimated_bytes", 0)),
            target_partition_bytes,
            actual_checkpoint_bytes,
            actual_checkpoint_writes,
            actual_checkpoint_wall,
            float(planned.get("planned_imbalance_ratio", 1.0)),
            str(bool(target_partition_bytes > 0 and actual_checkpoint_bytes > target_partition_bytes)).lower(),
        )


def _log_owner_local_finalize_metrics(
    *,
    inspected_target_states: dict[tuple[int, int | None], dict[str, object] | None],
    backend_kind: str,
) -> None:
    if not inspected_target_states:
        return
    total_bytes = 0
    total_writes = 0
    total_wall_seconds = 0.0
    saw_metrics = False
    for target_key in sorted(inspected_target_states):
        target_state = inspected_target_states.get(target_key) or {}
        checkpoint_metrics = target_state.get("checkpoint_metrics")
        if not isinstance(checkpoint_metrics, dict):
            checkpoint_metrics = target_state
        checkpoint_bytes = target_state.get(
            "total_checkpoint_bytes_written",
            checkpoint_metrics.get("total_checkpoint_bytes_written")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_bytes_written"),
        )
        checkpoint_writes = target_state.get(
            "total_checkpoint_writes",
            checkpoint_metrics.get("total_checkpoint_writes")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_writes"),
        )
        checkpoint_wall_seconds = target_state.get(
            "total_checkpoint_wall_seconds",
            checkpoint_metrics.get("total_checkpoint_wall_seconds")
            if checkpoint_metrics is not target_state
            else target_state.get("checkpoint_wall_seconds"),
        )
        if (
            checkpoint_bytes is None
            and checkpoint_writes is None
            and checkpoint_wall_seconds is None
        ):
            continue
        saw_metrics = True
        target_bytes = int(checkpoint_bytes or 0)
        target_writes = int(checkpoint_writes or 0)
        target_wall_seconds = float(checkpoint_wall_seconds or 0.0)
        total_bytes += target_bytes
        total_writes += target_writes
        total_wall_seconds += target_wall_seconds
        logger.info(
            "Residual-field finalize checkpoints | backend=%s | target=%s | writes=%d | bytes=%d | wall=%.3fs",
            backend_kind,
            target_key,
            target_writes,
            target_bytes,
            target_wall_seconds,
        )
    if saw_metrics:
        logger.info(
            "Residual-field finalize checkpoints total | backend=%s | targets=%d | writes=%d | bytes=%d | wall=%.3fs",
            backend_kind,
            int(len(inspected_target_states)),
            total_writes,
            total_bytes,
            total_wall_seconds,
        )
