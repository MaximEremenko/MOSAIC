from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Literal, cast


NufftExecutionPolicy = Literal[
    "auto",
    "gpu-required",
    "cpu-only",
    "allow-fallback",
]

VALID_NUFFT_POLICIES: tuple[NufftExecutionPolicy, ...] = (
    "auto",
    "gpu-required",
    "cpu-only",
    "allow-fallback",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_GPU_LAUNCH_WORKER_COMMANDS = {"dask-cuda-worker"}


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    raw = env.get(name)
    return raw is not None and str(raw).strip().lower() in _TRUE_VALUES


def _env_int(env: Mapping[str, str], name: str, default: int = 0) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def normalize_nufft_policy(value: object) -> NufftExecutionPolicy:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in VALID_NUFFT_POLICIES:
        return cast(NufftExecutionPolicy, normalized)
    valid = ", ".join(VALID_NUFFT_POLICIES)
    raise ValueError(f"NUFFT execution policy must be one of: {valid}.")


def gpu_launch_requested(*, env: Mapping[str, str] | None = None) -> bool:
    environment = _env(env)
    backend = str(environment.get("DASK_BACKEND", "")).strip().lower()
    worker_command = str(environment.get("DASK_WORKER_COMMAND", "")).strip().lower()
    return (
        backend == "cuda-local"
        or worker_command in _GPU_LAUNCH_WORKER_COMMANDS
        or _env_int(environment, "GPUS_PER_JOB", 0) > 0
    )


def resolve_nufft_policy(
    requested: object | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> NufftExecutionPolicy:
    """Resolve NUFFT execution policy with runtime environment precedence.

    Precedence:
      1. ``MOSAIC_NUFFT_CPU_ONLY=1`` forces ``cpu-only``.
      2. Explicit caller policy, when supplied.
      3. ``MOSAIC_NUFFT_EXECUTION_POLICY``.
      4. GPU launch environment implies ``gpu-required``.
      5. Default ``auto``.
    """
    environment = _env(env)
    if _env_truthy(environment, "MOSAIC_NUFFT_CPU_ONLY"):
        return "cpu-only"
    if requested is not None:
        return normalize_nufft_policy(requested)
    raw_policy = environment.get("MOSAIC_NUFFT_EXECUTION_POLICY")
    if raw_policy is not None and str(raw_policy).strip():
        return normalize_nufft_policy(raw_policy)
    if gpu_launch_requested(env=environment):
        return "gpu-required"
    return "auto"


def nufft_execute_kwargs(policy: object) -> dict[str, bool]:
    resolved = normalize_nufft_policy(policy)
    return {
        "prefer_cpu": resolved == "cpu-only",
        "gpu_only": resolved in {"gpu-required", "allow-fallback"},
    }


def nufft_task_retries(policy: object, default_retries: int) -> int:
    resolved = normalize_nufft_policy(policy)
    if resolved in {"gpu-required", "allow-fallback"}:
        return 0
    return int(default_retries)


def is_nufft_gpu_resource_failure(error: object) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "cuda",
            "cudart",
            "cufft",
            "cufinufft",
            "cupy",
            "gpu execution forced",
            "no cuda device",
            "out of memory",
            "memory allocation",
            "cuda_error_out_of_memory",
            "cudaerroroutofmemory",
            "insufficient resources",
            "too many resources requested",
            "launch-resource-exhausted",
            "budget",
            "driver shutting down",
            "device-side assert",
            "illegal memory access",
        )
    )


def should_resubmit_cpu_fallback(
    policy: object,
    failure: object,
    *,
    already_resubmitted: bool,
) -> bool:
    return (
        normalize_nufft_policy(policy) == "allow-fallback"
        and not bool(already_resubmitted)
        and is_nufft_gpu_resource_failure(failure)
    )


__all__ = [
    "NufftExecutionPolicy",
    "VALID_NUFFT_POLICIES",
    "gpu_launch_requested",
    "is_nufft_gpu_resource_failure",
    "normalize_nufft_policy",
    "nufft_execute_kwargs",
    "nufft_task_retries",
    "resolve_nufft_policy",
    "should_resubmit_cpu_fallback",
]
