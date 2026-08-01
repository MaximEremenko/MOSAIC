"""Dead-owner rescue at the flush/inspect/finalize barriers.

A future pinned with workers=[owner], allow_other_workers=False whose only
allowed worker died parks in no-worker state as 'pending' forever — the fold
drain loop already rescues this, but the post-drain barriers used a bare
as_completed and hung. _drain_owner_pinned_barrier must cancel dead-pinned
futures, resubmit them on a live worker, and abort if the whole cluster
stays dead past the horizon."""
from types import SimpleNamespace

import pytest

import core.residual_field.execution as execution


class _FakeFuture:
    def __init__(self, status="pending", result=None):
        self.status = status
        self._result = result
        self.cancel_calls = 0

    def done(self):
        return self.status in ("finished", "error", "cancelled")

    def result(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def cancel(self):
        self.cancel_calls += 1
        self.status = "cancelled"


class _FakeClient:
    """Just enough surface for the barrier: a real-looking event loop (so
    is_sync_client is False) and scheduler_info-driven worker liveness."""

    def __init__(self, workers):
        self.loop = SimpleNamespace(asyncio_loop=object())
        self._workers = workers

    def scheduler_info(self):
        return {"workers": {address: {} for address in self._workers}}


@pytest.fixture(autouse=True)
def _immediate_wait(monkeypatch):
    import distributed

    monkeypatch.setattr(distributed, "wait", lambda *a, **k: None)


def test_dead_owner_future_is_cancelled_and_remapped(monkeypatch):
    client = _FakeClient(["tcp://alive"])
    live_future = _FakeFuture(status="finished", result=True)
    parked_future = _FakeFuture(status="pending")  # pinned to the dead owner
    resubmitted = {}

    def resubmit(key, new_owner):
        resubmitted[key] = new_owner
        return _FakeFuture(status="finished", result=True)

    results = dict()
    for key, future, ok in execution._drain_owner_pinned_barrier(
        client=client,
        futures_by_key={"a": live_future, "b": parked_future},
        owner_by_key={"a": "tcp://alive", "b": "tcp://dead"},
        resubmit=resubmit,
        barrier_name="test",
        timeout_seconds=0.01,
    ):
        results[key] = ok

    assert parked_future.cancel_calls == 1
    assert resubmitted == {"b": "tcp://alive"}
    assert results == {"a": True, "b": True}


def test_cancelled_future_is_resubmitted_even_with_live_owner(monkeypatch):
    client = _FakeClient(["tcp://alive"])
    cancelled = _FakeFuture(status="cancelled")
    calls = []

    def resubmit(key, new_owner):
        calls.append((key, new_owner))
        return _FakeFuture(status="finished", result=True)

    results = dict(
        (key, ok)
        for key, _future, ok in execution._drain_owner_pinned_barrier(
            client=client,
            futures_by_key={"a": cancelled},
            owner_by_key={"a": "tcp://alive"},
            resubmit=resubmit,
            barrier_name="test",
            timeout_seconds=0.01,
        )
    )
    assert calls == [("a", "tcp://alive")]
    assert results == {"a": True}


def test_all_workers_dead_raises_past_horizon(monkeypatch):
    monkeypatch.setattr(execution, "_dead_cluster_horizon_seconds", lambda: 0.0)
    client = _FakeClient([])
    with pytest.raises(RuntimeError, match="no live workers"):
        list(
            execution._drain_owner_pinned_barrier(
                client=client,
                futures_by_key={"a": _FakeFuture(status="pending")},
                owner_by_key={"a": "tcp://dead"},
                resubmit=lambda key, owner: _FakeFuture(),
                barrier_name="test",
                timeout_seconds=0.01,
            )
        )
