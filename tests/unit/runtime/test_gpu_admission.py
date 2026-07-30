from __future__ import annotations

import pytest

from core.runtime.gpu_admission import (
    GPUAdmissionError,
    GPUWorkerReport,
    nufft_task_resources,
    require_gpu_admission,
    runtime_provenance_for_attempt,
)


class FakeClient:
    def __init__(self, workers):
        self._workers = workers

    def scheduler_info(self):
        return {"workers": self._workers}


def _worker(address="worker-a", cuda_visible_devices="0", nthreads=1):
    return GPUWorkerReport(
        address=address,
        host="host-a",
        cuda_visible_devices=cuda_visible_devices,
        nthreads=nthreads,
        cufinufft_available=True,
        cupy_version="1",
        cufinufft_version="2",
        cuda_device_uuid=f"uuid-{address}",
        error=None,
    )


def test_nufft_task_resources_adds_gpu_only_for_gpu_policies():
    assert nufft_task_resources("cpu-only") == {"nufft": 1}
    assert nufft_task_resources("auto") == {"nufft": 1}
    assert nufft_task_resources("gpu-required") == {"nufft": 1, "gpu": 1}
    assert nufft_task_resources("allow-fallback") == {"nufft": 1, "gpu": 1}


def test_gpu_admission_requires_scheduler_gpu_resource(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (_worker(),),
    )

    with pytest.raises(RuntimeError, match="'gpu'"):
        require_gpu_admission(
            FakeClient({"worker-a": {"resources": {"nufft": 1}}}),
            policy="gpu-required",
            required_gpu_tasks=1,
        )


def test_gpu_admission_accepts_pinned_single_thread_worker(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (_worker(),),
    )

    report = require_gpu_admission(
        FakeClient({"worker-a": {"resources": {"gpu": 1, "nufft": 1}}}),
        policy="gpu-required",
        required_gpu_tasks=1,
    )

    assert report.gpu_required is True
    assert report.resource_slots == 1
    assert report.workers[0].cuda_visible_devices == "0"


def test_gpu_admission_accepts_single_local_worker_without_cuda_visible_devices(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (_worker(cuda_visible_devices=None),),
    )

    report = require_gpu_admission(
        FakeClient({"worker-a": {"resources": {"gpu": 1, "nufft": 1}}}),
        policy="gpu-required",
        required_gpu_tasks=1,
    )

    assert report.gpu_required is True
    assert report.resource_slots == 1
    assert report.workers[0].cuda_visible_devices is None


def test_gpu_admission_rejects_duplicate_visible_device(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (
            _worker(address="worker-a", cuda_visible_devices="0"),
            _worker(address="worker-b", cuda_visible_devices="0"),
        ),
    )

    with pytest.raises(GPUAdmissionError, match="also assigned"):
        require_gpu_admission(
            FakeClient(
                {
                    "worker-a": {"resources": {"gpu": 1, "nufft": 1}},
                    "worker-b": {"resources": {"gpu": 1, "nufft": 1}},
                }
            ),
            policy="gpu-required",
            required_gpu_tasks=1,
        )


def test_gpu_admission_rejects_duplicate_all_visible_local_workers(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (
            _worker(address="worker-a", cuda_visible_devices=None),
            _worker(address="worker-b", cuda_visible_devices=None),
        ),
    )

    with pytest.raises(GPUAdmissionError, match="also assigned"):
        require_gpu_admission(
            FakeClient(
                {
                    "worker-a": {"resources": {"gpu": 1, "nufft": 1}},
                    "worker-b": {"resources": {"gpu": 1, "nufft": 1}},
                }
            ),
            policy="gpu-required",
            required_gpu_tasks=1,
        )


def test_gpu_admission_accepts_dask_cuda_rotated_visibility(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.gpu_admission._worker_reports",
        lambda client: (
            _worker(address="worker-a", cuda_visible_devices="0,1,2,3"),
            _worker(address="worker-b", cuda_visible_devices="1,2,3,0"),
            _worker(address="worker-c", cuda_visible_devices="2,3,0,1"),
            _worker(address="worker-d", cuda_visible_devices="3,0,1,2"),
        ),
    )

    report = require_gpu_admission(
        FakeClient(
            {
                address: {"resources": {"gpu": 1, "nufft": 1}}
                for address in ("worker-a", "worker-b", "worker-c", "worker-d")
            }
        ),
        policy="gpu-required",
        required_gpu_tasks=1,
    )

    assert len(report.workers) == 4


def test_runtime_provenance_contains_required_attempt_fields():
    provenance = runtime_provenance_for_attempt(
        fs_capability_digest="f" * 64,
        scheduler_kind="dask",
        nufft_policy="cpu-only",
        resource_requirements={"nufft": 1},
    )

    assert provenance["fs_capability_digest"] == "f" * 64
    assert provenance["scheduler_kind"] == "dask"
    assert provenance["nufft_policy"] == "cpu-only"
    assert provenance["resource_requirements"] == {"nufft": 1}
    assert "worker_host" in provenance
    assert "cuda" in provenance
