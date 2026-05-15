import pytest

from core.runtime.dask_resources import (
    require_scheduler_resource_capacity,
    scheduler_resource_capacity,
    wait_for_scheduler_resource_capacity,
)


class FakeClient:
    def __init__(self, workers):
        self._workers = workers

    def scheduler_info(self):
        return {"workers": self._workers}


def test_require_scheduler_resource_capacity_returns_capacity_and_worker_count():
    client = FakeClient(
        {
            "worker-a": {"resources": {"nufft": 1}},
            "worker-b": {"resources": {"nufft": 2.5}},
        }
    )

    assert require_scheduler_resource_capacity(
        client,
        "nufft",
        context="test scheduling",
    ).effective_runnable_slots == 3

    capacity = scheduler_resource_capacity(client, "nufft")
    assert capacity.worker_count == 2
    assert capacity.eligible_worker_addresses == ("worker-a", "worker-b")
    assert capacity.effective_runnable_slots == 3
    assert capacity.total_declared_capacity == 3.5


def test_fractional_worker_resources_do_not_aggregate_across_workers():
    client = FakeClient(
        {
            "worker-a": {"resources": {"nufft": 0.5}},
            "worker-b": {"resources": {"nufft": 0.5}},
        }
    )

    capacity = scheduler_resource_capacity(client, "nufft")

    assert capacity.worker_count == 2
    assert capacity.total_declared_capacity == 1.0
    assert capacity.eligible_worker_addresses == ()
    assert capacity.effective_runnable_slots == 0
    with pytest.raises(RuntimeError, match="Fractional worker resources cannot be aggregated"):
        require_scheduler_resource_capacity(
            client,
            "nufft",
            context="fractional test",
        )


def test_require_scheduler_resource_capacity_raises_clear_error_for_zero_capacity():
    client = FakeClient(
        {
            "worker-a": {"resources": {}},
            "worker-b": {"resources": {"nufft": 0}},
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        require_scheduler_resource_capacity(
            client,
            "nufft",
            context="test scheduling",
        )

    message = str(exc_info.value)
    assert "'nufft'" in message
    assert "test scheduling" in message
    assert "observed worker count=2" in message
    assert "resources={'nufft': N}" in message
    assert "MOSAIC's Dask client factory" in message


def test_require_scheduler_resource_capacity_fails_zero_workers_by_default():
    client = FakeClient({})

    with pytest.raises(RuntimeError, match="zero workers"):
        require_scheduler_resource_capacity(
            client,
            "nufft",
            context="test scheduling",
        )


def test_require_scheduler_resource_capacity_deferred_waits_for_workers(monkeypatch):
    snapshots = [
        {},
        {"worker-a": {"resources": {"nufft": 1}}},
    ]
    calls = []

    class ChangingClient:
        def scheduler_info(self):
            calls.append(1)
            return {"workers": snapshots[min(len(calls) - 1, len(snapshots) - 1)]}

    monkeypatch.setenv("MOSAIC_DASK_RESOURCE_READINESS", "deferred")
    monkeypatch.setattr("core.runtime.dask_resources.time.sleep", lambda seconds: None)

    capacity = require_scheduler_resource_capacity(
        ChangingClient(),
        "nufft",
        context="test scheduling",
    )

    assert len(calls) == 2
    assert capacity.worker_count == 1
    assert capacity.effective_runnable_slots == 1
    assert capacity.eligible_worker_addresses == ("worker-a",)


def test_wait_for_scheduler_resource_capacity_polls_until_runnable_slot(monkeypatch):
    calls = []
    snapshots = [
        {},
        {"worker-a": {"resources": {"nufft": 1}}},
    ]

    class ChangingClient:
        def scheduler_info(self):
            calls.append(1)
            return {"workers": snapshots[min(len(calls) - 1, len(snapshots) - 1)]}

    monkeypatch.setattr("core.runtime.dask_resources.time.sleep", lambda seconds: None)

    capacity = wait_for_scheduler_resource_capacity(
        ChangingClient(),
        "nufft",
        context="jobqueue startup",
        timeout_seconds=1.0,
    )

    assert len(calls) == 2
    assert capacity.eligible_worker_addresses == ("worker-a",)
    assert capacity.effective_runnable_slots == 1
