from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Mapping

from core.runtime.dask_resources import require_scheduler_resource_capacity
from core.runtime.nufft_policy import normalize_nufft_policy


class GPUAdmissionError(RuntimeError):
    """Raised when a requested GPU runtime cannot be proven safe."""


@dataclass(frozen=True)
class GPUWorkerReport:
    address: str
    host: str
    cuda_visible_devices: str | None
    nthreads: int | None
    cufinufft_available: bool
    cupy_version: str | None
    cufinufft_version: str | None
    cuda_device_uuid: str | None
    error: str | None = None


@dataclass(frozen=True)
class GPUAdmissionReport:
    policy: str
    gpu_required: bool
    resource_slots: int
    workers: tuple[GPUWorkerReport, ...]


def nufft_task_resources(policy: object) -> dict[str, int]:
    resolved = normalize_nufft_policy(policy)
    resources = {"nufft": 1}
    if resolved in {"gpu-required", "allow-fallback"}:
        resources["gpu"] = 1
    return resources


def _probe_worker_gpu() -> dict[str, Any]:
    address = "local"
    nthreads: int | None = None
    try:
        from distributed import get_worker

        worker = get_worker()
        address = str(getattr(worker, "address", "worker"))
        nthreads = int(getattr(worker, "nthreads", 0) or 0) or None
    except Exception:
        pass

    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    report: dict[str, Any] = {
        "address": address,
        "host": socket.gethostname(),
        "cuda_visible_devices": cuda_visible,
        "nthreads": nthreads,
        "cufinufft_available": False,
        "cupy_version": None,
        "cufinufft_version": None,
        "cuda_device_uuid": None,
        "error": None,
    }
    try:
        import cupy as cp  # type: ignore
        import cufinufft  # type: ignore

        report["cupy_version"] = getattr(cp, "__version__", None)
        report["cufinufft_version"] = getattr(cufinufft, "__version__", None)
        device = cp.cuda.Device()
        attrs = device.attributes
        uuid = attrs.get("Uuid") if isinstance(attrs, Mapping) else None
        report["cuda_device_uuid"] = None if uuid is None else str(uuid)
        # Force a light runtime touch without allocating a large plan.
        cp.cuda.runtime.getDeviceCount()
        report["cufinufft_available"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _worker_reports(client) -> tuple[GPUWorkerReport, ...]:
    if client is None or not hasattr(client, "run"):
        raw_values = (_probe_worker_gpu(),)
    else:
        try:
            raw = client.run(_probe_worker_gpu)
            raw_values = tuple(raw.values()) if isinstance(raw, Mapping) else (raw,)
        except Exception as exc:
            raw_values = (
                {
                    "address": "scheduler",
                    "host": socket.gethostname(),
                    "cuda_visible_devices": None,
                    "nthreads": None,
                    "cufinufft_available": False,
                    "cupy_version": None,
                    "cufinufft_version": None,
                    "cuda_device_uuid": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    return tuple(
        GPUWorkerReport(
            address=str(item.get("address", "unknown")),
            host=str(item.get("host", "unknown")),
            cuda_visible_devices=item.get("cuda_visible_devices"),
            nthreads=(
                None if item.get("nthreads") is None else int(item.get("nthreads"))
            ),
            cufinufft_available=bool(item.get("cufinufft_available")),
            cupy_version=item.get("cupy_version"),
            cufinufft_version=item.get("cufinufft_version"),
            cuda_device_uuid=item.get("cuda_device_uuid"),
            error=item.get("error"),
        )
        for item in raw_values
        if isinstance(item, Mapping)
    )


def _configured_thread_override() -> bool:
    raw = os.getenv("MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _visible_device_tokens(report: GPUWorkerReport) -> tuple[str, ...]:
    visible = str(report.cuda_visible_devices or "").strip()
    if not visible:
        # WSL/local CUDA commonly leaves CUDA_VISIBLE_DEVICES unset while one
        # process owns the default visible GPU. Treat that as one all-visible
        # token so a single worker is accepted and multiple workers on the same
        # host are still rejected below.
        return ("<all-visible>",)
    return tuple(token.strip() for token in visible.split(",") if token.strip())


def require_gpu_admission(
    client,
    *,
    policy: object,
    required_gpu_tasks: int,
    allow_multi_worker_per_gpu: bool = False,
) -> GPUAdmissionReport:
    resolved = normalize_nufft_policy(policy)
    gpu_required = resolved in {"gpu-required", "allow-fallback"} or int(required_gpu_tasks) > 0
    if not gpu_required:
        return GPUAdmissionReport(
            policy=resolved,
            gpu_required=False,
            resource_slots=0,
            workers=(),
        )

    capacity = require_scheduler_resource_capacity(
        client,
        "gpu",
        context="GPU NUFFT admission",
        minimum_slots=max(1, int(required_gpu_tasks)),
    )
    reports = _worker_reports(client)
    failures: list[str] = []
    active_tokens: dict[tuple[str, str], str] = {}
    for report in reports:
        if (
            report.nthreads is not None
            and report.nthreads != 1
            and not _configured_thread_override()
        ):
            failures.append(
                f"{report.address}: GPU worker has {report.nthreads} CPU threads; "
                "set one thread or MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER=1"
            )
        if not report.cufinufft_available:
            failures.append(
                f"{report.address}: cuFINUFFT probe failed"
                + (f" ({report.error})" if report.error else "")
            )
        # Dask-CUDA rotates the complete CUDA_VISIBLE_DEVICES list per worker
        # and assigns the first token to logical device 0. The remaining tokens
        # being visible is expected and does not mean the worker owns them.
        token = _visible_device_tokens(report)[0]
        key = (report.host, token)
        if key in active_tokens and not allow_multi_worker_per_gpu:
            failures.append(
                f"{report.address}: active CUDA device {token!r} on {report.host} "
                f"is also assigned to {active_tokens[key]}"
            )
        active_tokens[key] = report.address
    if failures:
        raise GPUAdmissionError("GPU admission failed: " + "; ".join(failures))
    return GPUAdmissionReport(
        policy=resolved,
        gpu_required=True,
        resource_slots=int(capacity.effective_runnable_slots),
        workers=reports,
    )


def runtime_provenance_for_attempt(
    *,
    fs_capability_digest: str | None = None,
    scheduler_kind: str = "local",
    nufft_policy: object = "auto",
    resource_requirements: Mapping[str, Any] | None = None,
    cuda_probe: bool = False,
) -> dict[str, Any]:
    report = _probe_worker_gpu() if cuda_probe else {
        "address": "local",
        "host": socket.gethostname(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nthreads": None,
        "cufinufft_available": False,
        "cupy_version": None,
        "cufinufft_version": None,
        "cuda_device_uuid": None,
        "error": None,
    }
    return {
        "fs_capability_digest": fs_capability_digest,
        "scheduler_kind": str(scheduler_kind),
        "nufft_policy": normalize_nufft_policy(nufft_policy),
        "resource_requirements": dict(resource_requirements or {"nufft": 1}),
        "worker_host": str(report.get("host", socket.gethostname())),
        "worker_address": str(report.get("address", "local")),
        "cpu_thread_policy": {
            "worker_nthreads": report.get("nthreads"),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
            "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
            "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        },
        "cuda": {
            "cuda_visible_devices": report.get("cuda_visible_devices"),
            "cuda_device_uuid": report.get("cuda_device_uuid"),
            "cupy_version": report.get("cupy_version"),
            "cufinufft_version": report.get("cufinufft_version"),
            "cufinufft_available": bool(report.get("cufinufft_available")),
            "probe_error": report.get("error"),
        },
    }


__all__ = [
    "GPUAdmissionError",
    "GPUAdmissionReport",
    "GPUWorkerReport",
    "nufft_task_resources",
    "require_gpu_admission",
    "runtime_provenance_for_attempt",
]
