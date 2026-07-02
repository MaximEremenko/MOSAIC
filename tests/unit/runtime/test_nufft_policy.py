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


def test_resolved_execution_settings_auto_attempts_gpu_with_cpu_fallback():
    settings = resolve_nufft_execution_settings(
        "auto",
        eps=1e-9,
        dtype="complex128",
        env={},
    )

    assert settings.requested_policy == "auto"
    # ``auto`` now attempts GPU first but ALWAYS permits a silent CPU fallback:
    # the wrapper probes the device at execution time and runs on CPU when the
    # GPU is unavailable/exhausted, so this stays safe on CPU-only hosts. The
    # execution-policy label stays ``auto`` (not a strict GPU policy) so no hard
    # admission gate or ``gpu`` task-resource is imposed, and the recorded
    # identity stays the conservative ``cpu`` until a realized device is stamped.
    assert settings.execution_policy == "auto"
    assert settings.backend == "cpu"
    assert settings.execute_kwargs == {"prefer_cpu": False, "gpu_only": False}
    assert settings.identity_payload()["eps"] == 1e-9


def test_resolved_execution_settings_gpu_required_binds_cuda_identity():
    settings = resolve_nufft_execution_settings(
        "gpu-required",
        eps=1e-12,
        dtype="complex128",
        env={},
    )

    assert settings.requested_policy == "gpu-required"
    assert settings.execution_policy == "gpu-required"
    # GPU enforced with no fallback -> cuda identity is definitively true.
    assert settings.backend == "cuda"
    assert settings.realized_backend_known is True
    assert settings.execute_kwargs == {"prefer_cpu": False, "gpu_only": True}


def test_allow_fallback_does_not_bind_cuda_identity_until_realized():
    settings = resolve_nufft_execution_settings(
        "allow-fallback",
        eps=1e-12,
        dtype="complex128",
        env={},
    )

    # allow-fallback still *attempts* GPU first ...
    assert settings.requested_policy == "allow-fallback"
    assert settings.execution_policy == "allow-fallback"
    assert settings.execute_kwargs == {"prefer_cpu": False, "gpu_only": True}
    # ... but it must NOT stamp a cuda identity, because the wrapper may
    # silently fall back to CPU. Recorded backend is the conservative cpu and
    # the realized device is explicitly not yet known.
    assert settings.backend == "cpu"
    assert settings.realized_backend_known is False
    assert settings.identity_payload()["backend"] == "cpu"
    assert settings.identity_payload()["realized_backend_known"] is False


def test_allow_fallback_records_realized_backend():
    settings = resolve_nufft_execution_settings(
        "allow-fallback",
        eps=1e-12,
        dtype="complex128",
        env={},
    )

    # Realized as CPU (the GPU did not run): identity must read cpu.
    realized_cpu = settings.with_realized_backend("cpu")
    assert realized_cpu.backend == "cpu"
    assert realized_cpu.realized_backend_known is True
    assert realized_cpu.identity_payload()["backend"] == "cpu"

    # Realized as CUDA (the GPU actually ran): only now may identity read cuda.
    realized_cuda = settings.with_realized_backend("cuda")
    assert realized_cuda.backend == "cuda"
    assert realized_cuda.realized_backend_known is True
    assert realized_cuda.identity_payload()["backend"] == "cuda"


def test_with_realized_backend_fails_closed_on_device_bound_contradiction():
    gpu_required = resolve_nufft_execution_settings(
        "gpu-required", eps=1e-12, dtype="complex128", env={}
    )
    # A device-bound gpu-required run must never be re-recorded as cpu.
    with pytest.raises(ValueError, match="contradicts device-bound"):
        gpu_required.with_realized_backend("cpu")
    # Re-stamping the same device is a no-op (idempotent).
    assert gpu_required.with_realized_backend("cuda") is gpu_required

    cpu_only = resolve_nufft_execution_settings(
        "cpu-only", eps=1e-12, dtype="complex128", env={}
    )
    with pytest.raises(ValueError, match="contradicts device-bound"):
        cpu_only.with_realized_backend("cuda")


def test_resolved_execution_settings_reject_non_complex128_dtype():
    with pytest.raises(ValueError, match="complex128"):
        resolve_nufft_execution_settings("cpu-only", dtype="complex64")
