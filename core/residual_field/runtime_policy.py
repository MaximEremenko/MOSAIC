"""Runtime policy readers for the residual-field stage.

All functions here are pure leaf utilities: they read workflow_parameters /
environment variables and return plain Python values.  None of them import
from core.residual_field.execution, so there are no import cycles.
"""

from __future__ import annotations

import os

from core.runtime import (
    nufft_task_resources,
    resolve_nufft_execution_settings,
)
from core.residual_field.backend import ResidualFieldReducerBackend

__all__ = [
    "DEFAULT_RESIDUAL_PARTITION_TARGET_BYTES",
    "_cleanup_residual_attempts_enabled",
    "_distributed_owner_affinity_enabled",
    "_distributed_owner_local_reducer_supported",
    "_memory_backpressure_poll_seconds",
    "_memory_backpressure_threshold",
    "_owner_local_reducer_enabled",
    "_residual_attempt_cleanup_policy",
    "_residual_nufft_policy",
    "_residual_nufft_prefetch_factor",
    "_residual_nufft_resources",
    "_residual_nufft_settings",
    "_residual_partition_runtime_policy",
    "_residual_rifft_payload_reuse_enabled",
    "_worker_owned_local_reducer_enabled",
]

DEFAULT_RESIDUAL_PARTITION_TARGET_BYTES = 256 * 1024 * 1024


def _worker_owned_local_reducer_enabled(workflow_parameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    override = None
    if hasattr(runtime_info, "get"):
        override = runtime_info.get("residual_local_owner_reducer")
    if override is None:
        override = os.getenv("MOSAIC_RESIDUAL_LOCAL_OWNER_REDUCER")
    if override is None:
        return True
    if isinstance(override, str):
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(override)


def _distributed_owner_affinity_enabled(workflow_parameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    override = None
    if hasattr(runtime_info, "get"):
        override = runtime_info.get("residual_distributed_owner_affinity")
    if override is None:
        override = os.getenv("MOSAIC_RESIDUAL_DISTRIBUTED_OWNER_AFFINITY")
    if override is None:
        return True
    if isinstance(override, str):
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(override)


def _distributed_owner_local_reducer_supported(
    reducer_backend: ResidualFieldReducerBackend,
    *,
    reducer_runtime_state,
) -> bool:
    support_override = getattr(
        reducer_backend,
        "distributed_owner_local_reducer_supported",
        None,
    )
    if support_override is None:
        support_override = getattr(
            reducer_backend,
            "supports_distributed_owner_local_reducer",
            None,
        )
    if support_override is not None:
        return bool(support_override)
    return (
        callable(getattr(reducer_backend, "accept_local_contribution", None))
        and callable(getattr(reducer_backend, "inspect_local_reducer_target", None))
        and callable(getattr(reducer_backend, "flush_local_reducer_target", None))
        and getattr(reducer_runtime_state, "durable_truth_unit", None)
        == "committed_local_snapshot_generation"
        and getattr(reducer_runtime_state, "durable_checkpoint_storage_role", None)
        in {"durable-local-snapshot-generation", "durable-shared-generation"}
    )


def _owner_local_reducer_enabled(
    *,
    reducer_backend: ResidualFieldReducerBackend,
    worker_owned_local_reducer: bool,
    distributed_owner_local_reducer: bool,
) -> bool:
    return bool(worker_owned_local_reducer or distributed_owner_local_reducer)


def _residual_partition_runtime_policy(
    workflow_parameters,
    *,
    default_target_bytes: int,
    effective_nufft_workers: int,
) -> dict[str, int | float]:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    default_target_bytes = (
        int(default_target_bytes)
        if int(default_target_bytes) > 0
        else DEFAULT_RESIDUAL_PARTITION_TARGET_BYTES
    )

    def _get_int(name: str, default: int) -> int:
        value = None
        if hasattr(runtime_info, "get"):
            value = runtime_info.get(name)
        if value is None:
            env_name = f"MOSAIC_{name.upper()}"
            value = os.getenv(env_name)
        return int(value) if value is not None else int(default)

    def _get_float(name: str, default: float) -> float:
        value = None
        if hasattr(runtime_info, "get"):
            value = runtime_info.get(name)
        if value is None:
            env_name = f"MOSAIC_{name.upper()}"
            value = os.getenv(env_name)
        return float(value) if value is not None else float(default)

    return {
        "target_partition_bytes": max(1, _get_int("residual_partition_target_bytes", int(default_target_bytes))),
        "target_partition_bytes_3d": max(
            1,
            _get_int(
                "residual_partition_target_bytes_3d",
                max(1, int(default_target_bytes) // 2),
            ),
        ),
        "max_partitions_per_chunk": _get_int(
            "residual_max_partitions_per_chunk",
            0,  # 0 = auto: let byte budget drive partition count
        ),
        "min_points_per_partition": max(
            1,
            _get_int("residual_min_points_per_partition", 1),
        ),
        "hysteresis_low_factor": _get_float(
            "residual_partition_hysteresis_low",
            0.8,
        ),
        "hysteresis_high_factor": _get_float(
            "residual_partition_hysteresis_high",
            1.2,
        ),
    }


def _residual_nufft_prefetch_factor(workflow_parameters) -> int:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    value = None
    if hasattr(runtime_info, "get"):
        value = runtime_info.get("residual_nufft_prefetch_factor")
    if value is None:
        value = os.getenv("MOSAIC_RESIDUAL_NUFFT_PREFETCH_FACTOR")
    try:
        factor = int(value) if value is not None else 2
    except (TypeError, ValueError):
        factor = 2
    return max(1, min(8, factor))


def _residual_rifft_payload_reuse_enabled(workflow_parameters) -> bool:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    value = None
    if hasattr(runtime_info, "get"):
        value = runtime_info.get("residual_reuse_rifft_payload")
    if value is None:
        value = os.getenv("MOSAIC_RESIDUAL_REUSE_RIFFT_PAYLOAD")
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _residual_nufft_settings(workflow_parameters):
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    if not hasattr(runtime_info, "get"):
        runtime_info = {}
    requested = runtime_info.get("residual_nufft_policy")
    if requested is None:
        requested = runtime_info.get("nufft_execution_policy")
    if requested is None:
        requested = runtime_info.get("nufft_policy")
    eps = runtime_info.get("residual_nufft_eps", runtime_info.get("nufft_eps", 1e-12))
    dtype = runtime_info.get("residual_dtype", runtime_info.get("nufft_dtype", "complex128"))
    return resolve_nufft_execution_settings(requested, eps=eps, dtype=dtype)


def _residual_nufft_policy(workflow_parameters) -> str:
    return _residual_nufft_settings(workflow_parameters).execution_policy


def _residual_nufft_resources(workflow_parameters) -> dict[str, int]:
    return nufft_task_resources(_residual_nufft_policy(workflow_parameters))


def _memory_backpressure_threshold() -> float:
    raw = os.getenv("MOSAIC_RESIDUAL_MEMORY_BACKPRESSURE_PCT")
    try:
        threshold = float(raw) if raw is not None else 0.72
    except ValueError:
        threshold = 0.72
    return max(0.0, min(1.0, threshold))


def _memory_backpressure_poll_seconds() -> float:
    raw = os.getenv("MOSAIC_RESIDUAL_MEMORY_BACKPRESSURE_POLL_SECONDS")
    try:
        seconds = float(raw) if raw is not None else 5.0
    except ValueError:
        seconds = 5.0
    return max(0.0, seconds)


def _cleanup_residual_attempts_enabled(workflow_parameters) -> bool:
    return _residual_attempt_cleanup_policy(workflow_parameters) == "delete_reclaimable"


def _residual_attempt_cleanup_policy(workflow_parameters) -> str:
    runtime_policy = workflow_parameters.runtime_info.get(
        "residual_attempt_cleanup_policy"
    )
    if runtime_policy is not None:
        value = str(runtime_policy).strip().lower()
        if value in {"off", "keep"}:
            return "off"
        if value in {"delete_reclaimable", "cleanup"}:
            return "delete_reclaimable"
    runtime_value = workflow_parameters.runtime_info.get("cleanup_residual_attempts")
    if runtime_value is not None:
        return "delete_reclaimable" if bool(runtime_value) else "off"
    return (
        "delete_reclaimable"
        if os.getenv("MOSAIC_CLEANUP_RESIDUAL_SHARDS", "0") == "1"
        else "off"
    )
