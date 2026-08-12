"""Tests for worker-heap hygiene: malloc_trim guard and cluster lifecycle."""
from __future__ import annotations

import ctypes
import json
import os

import pytest


# --- _final_cleanup tolerates missing glibc malloc_trim --------------------

def test_final_cleanup_survives_missing_malloc_trim(monkeypatch):
    """ctypes.CDLL('libc.so.6') raises OSError on non-glibc platforms."""
    import core.runtime.worker_hooks as wh

    monkeypatch.setattr(wh, "free_gpu_memory", lambda: None, raising=True)
    monkeypatch.setattr(
        wh, "_cleanup_process_local_reducers", lambda: None, raising=True
    )

    def _raise(*args, **kwargs):
        raise OSError("libc.so.6 not present on this platform")

    monkeypatch.setattr(ctypes, "CDLL", _raise)

    wh._final_cleanup()  # must not raise


def test_final_cleanup_survives_libc_without_malloc_trim(monkeypatch):
    """libc.so.6 loads but lacks malloc_trim (musl shims, ancient glibc)."""
    import core.runtime.worker_hooks as wh

    monkeypatch.setattr(wh, "free_gpu_memory", lambda: None, raising=True)
    monkeypatch.setattr(
        wh, "_cleanup_process_local_reducers", lambda: None, raising=True
    )

    class _FakeLib:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: _FakeLib())

    wh._final_cleanup()  # must not raise


def test_malloc_trim_noop_is_safe():
    """The public _malloc_trim wrapper must never raise."""
    from core.runtime.worker_hooks import _malloc_trim

    _malloc_trim()


def test_per_task_heap_trim_plugin_released_only(monkeypatch):
    """Plugin only fires on finish='released'; other states are no-ops."""
    from core.runtime.worker_hooks import _PerTaskHeapTrim

    plugin = _PerTaskHeapTrim()
    # Must not raise for any transition shape.
    plugin.transition("k", "processing", "memory")
    plugin.transition("k", "waiting", "processing")
    # Released -> the trim path runs; with no libc this still swallows.
    plugin.transition("k", "memory", "released")


# --- shutdown_dask reclaims cluster children -------------------------------

@pytest.mark.skipif(
    os.name == "nt",
    reason="LocalCluster child-process counts are unreliable on Windows CI",
)
def test_shutdown_dask_clears_singleton(monkeypatch):
    """After shutdown_dask() the module-level _CLIENT reference is cleared."""
    import core.runtime.dask_client as _dc

    _dc._CLIENT = object()  # simulate a live client singleton
    from core.runtime.dask_client import shutdown_dask

    shutdown_dask()
    assert _dc._CLIENT is None


def test_threaded_local_cluster_does_not_receive_worker_env(monkeypatch):
    """processes=False uses distributed.Worker, which has no env= kwarg."""
    import core.runtime.dask_helpers as dh

    captured = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, cluster):
            self.cluster = cluster

    monkeypatch.setattr(dh, "LocalCluster", _FakeCluster)
    monkeypatch.setattr(dh, "Client", _FakeClient)
    monkeypatch.setattr(dh, "_register_heap_trim_plugin", lambda client: None)

    dh.ensure_dask_client(
        backend="local",
        processes=False,
        dashboard=False,
        env={"CUSTOM": "1"},
        resources={"nufft": 1},
    )

    assert captured["processes"] is False
    assert "env" not in captured
    assert captured["memory_limit"] == 0


def test_threaded_local_cluster_preserves_explicit_memory_limit(monkeypatch):
    import core.runtime.dask_helpers as dh

    captured = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, cluster):
            self.cluster = cluster

    monkeypatch.setattr(dh, "LocalCluster", _FakeCluster)
    monkeypatch.setattr(dh, "Client", _FakeClient)
    monkeypatch.setattr(dh, "_register_heap_trim_plugin", lambda client: None)

    dh.ensure_dask_client(
        backend="local",
        processes=False,
        dashboard=False,
        memory_limit="8GiB",
    )

    assert captured["processes"] is False
    assert captured["memory_limit"] == "8GiB"


def test_process_local_cluster_receives_worker_env(monkeypatch):
    import core.runtime.dask_helpers as dh

    captured = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, cluster):
            self.cluster = cluster

    monkeypatch.setattr(dh, "LocalCluster", _FakeCluster)
    monkeypatch.setattr(dh, "Client", _FakeClient)
    monkeypatch.setattr(dh, "_register_heap_trim_plugin", lambda client: None)
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)

    dh.ensure_dask_client(
        backend="local",
        processes=True,
        dashboard=False,
        env={"CUSTOM": "1"},
        resources={"nufft": 1},
    )

    assert captured["processes"] is True
    assert captured["env"]["MALLOC_ARENA_MAX"] == "2"
    assert captured["env"]["CUSTOM"] == "1"


def test_local_cluster_processes_can_come_from_config(monkeypatch, tmp_path):
    import core.runtime.dask_helpers as dh

    captured = {}

    class _FakeCluster:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, cluster):
            self.cluster = cluster

    config_path = tmp_path / "dask.json"
    config_path.write_text(json.dumps({"processes": False}))
    monkeypatch.setenv("MOSAIC_DASK_CONFIG", str(config_path))
    monkeypatch.delenv("DASK_PROCESSES", raising=False)
    monkeypatch.setattr(dh, "LocalCluster", _FakeCluster)
    monkeypatch.setattr(dh, "Client", _FakeClient)
    monkeypatch.setattr(dh, "_register_heap_trim_plugin", lambda client: None)

    dh.ensure_dask_client(backend="local", processes=None, dashboard=False)

    assert captured["processes"] is False
    assert "env" not in captured
