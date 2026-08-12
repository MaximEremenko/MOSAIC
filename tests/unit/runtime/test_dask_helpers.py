from __future__ import annotations

import sys
from types import SimpleNamespace

import core.runtime.dask_helpers as dask_helpers


class _FakeFuture:
    def __init__(self, status: str) -> None:
        self.status = status


def test_yield_futures_with_results_treats_cancelled_and_error_as_failed(
    monkeypatch,
) -> None:
    finished = _FakeFuture("finished")
    cancelled = _FakeFuture("cancelled")
    errored = _FakeFuture("error")
    captured_kwargs: dict[str, object] = {}

    def fake_as_completed(futs, **kwargs):
        captured_kwargs.update(kwargs)
        assert futs == [finished, cancelled, errored]
        return iter(
            [
                (finished, {"manifest": 1}),
                (cancelled, RuntimeError("cancelled")),
                (errored, (RuntimeError, RuntimeError("boom"), None)),
            ]
        )

    monkeypatch.setattr(dask_helpers, "as_completed", fake_as_completed)

    results = list(
        dask_helpers.yield_futures_with_results(
            [finished, cancelled, errored],
            SimpleNamespace(loop="event-loop"),
        )
    )

    assert results == [
        (finished, True),
        (cancelled, False),
        (errored, False),
    ]
    assert captured_kwargs == {
        "with_results": True,
        "raise_errors": False,
        "loop": "event-loop",
    }


def test_yield_futures_with_results_treats_none_payload_as_failed(monkeypatch) -> None:
    finished = _FakeFuture("finished")

    def fake_as_completed(futs, **kwargs):
        assert futs == [finished]
        return iter([(finished, None)])

    monkeypatch.setattr(dask_helpers, "as_completed", fake_as_completed)

    assert list(dask_helpers.yield_futures_with_results([finished], None)) == [
        (finished, False)
    ]


def test_jobqueue_backend_forwards_worker_resources(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def scale(self, *, jobs):
            captured["jobs"] = jobs

    fake_jobqueue = SimpleNamespace(
        LSFCluster=_FakeCluster,
        OARCluster=_FakeCluster,
        PBSCluster=_FakeCluster,
        SGECluster=_FakeCluster,
        SLURMCluster=_FakeCluster,
    )

    monkeypatch.setitem(sys.modules, "dask_jobqueue", fake_jobqueue)
    monkeypatch.setattr(dask_helpers, "Client", lambda cluster: SimpleNamespace(cluster=cluster))
    monkeypatch.setattr(dask_helpers, "get_client", lambda: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setenv("DASK_SCHEDULER_IP", "127.0.0.1")

    dask_helpers.ensure_dask_client(
        backend="slurm",
        max_workers=2,
        threads_per_worker=4,
        resources={"nufft": 3},
        dashboard=False,
    )

    kwargs = captured["kwargs"]
    assert kwargs["worker_extra_args"] == ["--resources", "nufft=3"]
    assert "resources" not in kwargs
    assert captured["jobs"] == 2


def _fake_jobqueue(monkeypatch, captured: dict):
    class _FakeCluster:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def scale(self, *, jobs):
            captured["jobs"] = jobs

    fake_jobqueue = SimpleNamespace(
        LSFCluster=_FakeCluster,
        OARCluster=_FakeCluster,
        PBSCluster=_FakeCluster,
        SGECluster=_FakeCluster,
        SLURMCluster=_FakeCluster,
    )
    monkeypatch.setitem(sys.modules, "dask_jobqueue", fake_jobqueue)
    monkeypatch.setattr(
        dask_helpers, "Client", lambda cluster: SimpleNamespace(cluster=cluster)
    )
    monkeypatch.setattr(
        dask_helpers, "get_client", lambda: (_ for _ in ()).throw(ValueError())
    )
    monkeypatch.setenv("DASK_SCHEDULER_IP", "127.0.0.1")
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)


def test_jobqueue_slurm_keeps_gpu_directive_after_merge(monkeypatch):
    captured: dict[str, object] = {}
    _fake_jobqueue(monkeypatch, captured)

    dask_helpers.ensure_dask_client(
        backend="slurm",
        max_workers=2,
        threads_per_worker=4,
        gpu=4,
        dashboard=False,
        job_extra_directives=["--output=/logs/worker.o.%j"],
    )

    directives = captured["kwargs"]["job_extra_directives"]
    # concat-merge: default --gpus first, then user log directive
    assert "--gpus=4" in directives
    assert "--output=/logs/worker.o.%j" in directives
    assert not any(d in {"-cwd", "-V"} for d in directives)


def test_jobqueue_merge_preserves_scheduler_endpoint_defaults(monkeypatch):
    captured: dict[str, object] = {}
    _fake_jobqueue(monkeypatch, captured)

    dask_helpers.ensure_dask_client(
        backend="slurm",
        max_workers=1,
        threads_per_worker=2,
        dashboard=False,
        scheduler_options={"interface": "ib0"},
    )

    sched = captured["kwargs"]["scheduler_options"]
    assert sched["interface"] == "ib0"          # user key wins
    assert sched["port"] == 8786                # default endpoint preserved
    assert sched["host"] == "127.0.0.1:8786"


def test_merge_jobqueue_kwargs_concatenates_lists_user_last():
    merged = dask_helpers._merge_jobqueue_kwargs(
        {"job_extra_directives": ["--gpus=2"], "walltime": "01:00:00"},
        {"job_extra_directives": ["--output=o.%j", "--gpus=2"], "walltime": "02:00:00"},
    )
    assert merged["job_extra_directives"] == ["--gpus=2", "--output=o.%j"]
    assert merged["walltime"] == "02:00:00"     # scalar: user wins


def test_jobqueue_registers_heap_trim_and_worker_env(monkeypatch):
    captured: dict[str, object] = {}
    _fake_jobqueue(monkeypatch, captured)
    registered: list[object] = []
    monkeypatch.setattr(
        dask_helpers, "_register_heap_trim_plugin", registered.append
    )

    client = dask_helpers.ensure_dask_client(
        backend="slurm",
        max_workers=1,
        threads_per_worker=2,
        dashboard=False,
    )

    assert registered == [client]
    prologue = captured["kwargs"]["job_script_prologue"]
    assert "export MALLOC_ARENA_MAX=2" in prologue


def test_jobqueue_worker_env_respects_user_prologue(monkeypatch):
    captured: dict[str, object] = {}
    _fake_jobqueue(monkeypatch, captured)

    dask_helpers.ensure_dask_client(
        backend="slurm",
        max_workers=1,
        threads_per_worker=2,
        dashboard=False,
        job_script_prologue=["export MALLOC_ARENA_MAX=8"],
    )

    prologue = captured["kwargs"]["job_script_prologue"]
    assert prologue == ["export MALLOC_ARENA_MAX=8"]
