from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

from core.runtime.worker_hooks import (
    _final_cleanup,
    chunk_mutex,
    handle_worker_gpu_failure,
    register_cleanup_plugin,
    resolve_worker_scratch_root,
)


def test_register_cleanup_plugin_skips_sync_clients():
    class FakeClient:
        def register_worker_plugin(self, plugin, name=None):
            raise AssertionError("register_worker_plugin should not be called for sync clients")

    called = {"sync": False}

    def fake_is_sync_client(client):
        called["sync"] = True
        return True

    assert register_cleanup_plugin(FakeClient(), is_sync_client=fake_is_sync_client) is False
    assert called["sync"] is True


def test_register_cleanup_plugin_registers_worker_plugin_for_async_clients():
    calls = []

    class FakeClient:
        def register_worker_plugin(self, plugin, name=None):
            calls.append((plugin.name, name))

    assert register_cleanup_plugin(FakeClient(), is_sync_client=lambda client: False) is True
    assert calls == [("cupy-cleanup", "cupy-cleanup")]


def test_handle_worker_gpu_failure_on_non_gpu_error_only_cleans_up(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "core.runtime.worker_hooks.free_gpu_memory",
        lambda: calls.append("freed"),
    )

    handled = handle_worker_gpu_failure(RuntimeError("plain failure"), logger=SimpleNamespace(warning=lambda *args, **kwargs: None))

    assert handled is False
    assert calls == ["freed"]


def test_handle_worker_gpu_failure_pool_cap_oom_does_not_demote(monkeypatch):
    """A CuPy POOL-cap OOM is a workload-sizing error, not a sick device; the
    kernel paths re-tile and retry. Demoting here is what turned the first
    hkl40 streaming run into a 21 h CPU crawl."""
    calls = []
    monkeypatch.setattr(
        "core.runtime.worker_hooks.free_gpu_memory",
        lambda: calls.append("freed"),
    )
    monkeypatch.setattr(
        "core.runtime.worker_hooks.set_cpu_only",
        lambda flag=True: calls.append(("cpu_only", flag)),
    )

    err = RuntimeError(
        "Out of memory allocating 1,693,802,496 bytes (allocated so far: "
        "1,693,802,496 bytes, limit set to: 2,576,311,910 bytes)."
    )
    handled = handle_worker_gpu_failure(
        err, logger=SimpleNamespace(warning=lambda *args, **kwargs: None)
    )

    assert handled is False
    assert ("cpu_only", True) not in calls
    assert "freed" in calls


def test_handle_worker_gpu_failure_device_fault_still_demotes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "core.runtime.worker_hooks.free_gpu_memory",
        lambda: calls.append("freed"),
    )
    monkeypatch.setattr(
        "core.runtime.worker_hooks.set_cpu_only",
        lambda flag=True: calls.append(("cpu_only", flag)),
    )

    err = RuntimeError("CUDA error: an illegal memory access was encountered")
    handled = handle_worker_gpu_failure(
        err, logger=SimpleNamespace(warning=lambda *args, **kwargs: None)
    )

    assert handled is True
    assert ("cpu_only", True) in calls


def test_final_cleanup_also_cleans_process_local_reducers(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "core.runtime.worker_hooks.free_gpu_memory",
        lambda: calls.append("gpu"),
    )
    monkeypatch.setattr(
        "core.runtime.worker_hooks._cleanup_process_local_reducers",
        lambda: calls.append("reducers"),
    )

    _final_cleanup()

    assert calls[:2] == ["gpu", "reducers"]


def test_resolve_worker_scratch_root_uses_preferred_or_env(monkeypatch, tmp_path):
    preferred = tmp_path / "preferred"
    env_root = tmp_path / "env_root"
    monkeypatch.setenv("MOSAIC_WORKER_SCRATCH_ROOT", str(env_root))

    resolved_preferred = resolve_worker_scratch_root(
        preferred=str(preferred),
        stage="residual_field",
    )
    resolved_env = resolve_worker_scratch_root(
        preferred=None,
        stage="residual_field",
    )

    # No worker segment: this resolves on the DRIVER, which has no worker
    # identity. worker_local_scratch_dir adds it on the worker.
    assert resolved_preferred == os.fspath(preferred / "mosaic" / "residual_field")
    assert resolved_env == os.fspath(env_root / "mosaic" / "residual_field")


def test_chunk_mutex_does_not_instantiate_distributed_lock_when_lock_root_available(
    monkeypatch,
    tmp_path,
):
    lock_calls = []

    class FakeClient:
        loop = SimpleNamespace(asyncio_loop=object())

    def fake_lock(*args, **kwargs):
        lock_calls.append((args, kwargs))
        raise AssertionError("dask.distributed.Lock should not guard artifact writes")

    distributed_module = ModuleType("distributed")
    distributed_module.get_client = lambda: FakeClient()
    dask_module = ModuleType("dask")
    dask_distributed_module = ModuleType("dask.distributed")
    dask_distributed_module.Lock = fake_lock
    dask_module.distributed = dask_distributed_module

    monkeypatch.setitem(sys.modules, "distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "dask", dask_module)
    monkeypatch.setitem(sys.modules, "dask.distributed", dask_distributed_module)

    with chunk_mutex(7, lock_root=tmp_path / "artifact-root"):
        pass

    assert lock_calls == []
