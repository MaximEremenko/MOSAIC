from __future__ import annotations

import pytest

from core.runtime.nufft_policy import (
    gpu_launch_requested,
    is_nufft_gpu_resource_failure,
    normalize_nufft_policy,
    nufft_execute_kwargs,
    nufft_task_retries,
    resolve_nufft_execution_settings,
    resolve_nufft_policy,
    should_resubmit_cpu_fallback,
)


def test_resolve_nufft_policy_honors_cpu_only_before_all_other_sources():
    env = {
        "MOSAIC_NUFFT_CPU_ONLY": "1",
        "MOSAIC_NUFFT_EXECUTION_POLICY": "gpu-required",
        "DASK_WORKER_COMMAND": "dask-cuda-worker",
        "GPUS_PER_JOB": "4",
    }

    assert resolve_nufft_policy("gpu-required", env=env) == "cpu-only"


def test_resolve_nufft_policy_uses_explicit_and_env_before_gpu_launch():
    assert (
        resolve_nufft_policy(
            "cpu_only",
            env={
                "MOSAIC_NUFFT_EXECUTION_POLICY": "gpu-required",
                "DASK_WORKER_COMMAND": "dask-cuda-worker",
            },
        )
        == "cpu-only"
    )
    assert (
        resolve_nufft_policy(
            env={
                "MOSAIC_NUFFT_EXECUTION_POLICY": "allow_fallback",
                "DASK_WORKER_COMMAND": "dask-cuda-worker",
            }
        )
        == "allow-fallback"
    )


def test_resolve_nufft_policy_gpu_launch_defaults_to_gpu_required():
    assert gpu_launch_requested(env={"DASK_BACKEND": "cuda-local"}) is True
    assert gpu_launch_requested(env={"DASK_WORKER_COMMAND": "dask-cuda-worker"}) is True
    assert gpu_launch_requested(env={"GPUS_PER_JOB": "1"}) is True
    assert resolve_nufft_policy(env={"GPUS_PER_JOB": "1"}) == "gpu-required"


def test_nufft_execute_kwargs_encode_fail_fast_and_cpu_modes():
    assert nufft_execute_kwargs("auto") == {
        "prefer_cpu": False,
        "gpu_only": False,
    }
    assert nufft_execute_kwargs("gpu-required") == {
        "prefer_cpu": False,
        "gpu_only": True,
    }
    assert nufft_execute_kwargs("allow-fallback") == {
        "prefer_cpu": False,
        "gpu_only": True,
    }
    assert nufft_execute_kwargs("cpu-only") == {
        "prefer_cpu": True,
        "gpu_only": False,
    }


def test_nufft_task_retries_and_fallback_policy():
    assert nufft_task_retries("gpu-required", 4) == 0
    assert nufft_task_retries("allow-fallback", 4) == 0
    assert nufft_task_retries("auto", 4) == 4
    assert should_resubmit_cpu_fallback(
        "allow-fallback",
        RuntimeError("CUDA out of memory"),
        already_resubmitted=False,
    )
    assert not should_resubmit_cpu_fallback(
        "allow-fallback",
        RuntimeError("CUDA out of memory"),
        already_resubmitted=True,
    )
    assert not should_resubmit_cpu_fallback(
        "auto",
        RuntimeError("CUDA out of memory"),
        already_resubmitted=False,
    )


def test_nufft_gpu_failure_classifier_and_invalid_policy():
    assert is_nufft_gpu_resource_failure("CUDA_ERROR_OUT_OF_MEMORY")
    assert not is_nufft_gpu_resource_failure("plain validation failure")
    with pytest.raises(ValueError):
        normalize_nufft_policy("sometimes-gpu")


def test_resolved_execution_settings_make_auto_concrete_cpu():
    settings = resolve_nufft_execution_settings(
        "auto",
        eps=1e-9,
        dtype="complex128",
        env={},
    )

    assert settings.requested_policy == "auto"
    assert settings.execution_policy == "cpu-only"
    assert settings.backend == "cpu"
    assert settings.execute_kwargs == {"prefer_cpu": True, "gpu_only": False}
    assert settings.identity_payload()["eps"] == 1e-9


def test_resolved_execution_settings_make_gpu_required_fail_fast_contract():
    settings = resolve_nufft_execution_settings(
        "allow-fallback",
        eps=1e-12,
        dtype="complex128",
        env={},
    )

    assert settings.requested_policy == "allow-fallback"
    assert settings.execution_policy == "gpu-required"
    assert settings.backend == "cuda"
    assert settings.execute_kwargs == {"prefer_cpu": False, "gpu_only": True}


def test_resolved_execution_settings_reject_non_complex128_dtype():
    with pytest.raises(ValueError, match="complex128"):
        resolve_nufft_execution_settings("cpu-only", dtype="complex64")
