from __future__ import annotations

import os
from types import SimpleNamespace

from core.runtime.worker_hooks import (
    _final_cleanup,
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

    assert resolved_preferred == os.fspath(preferred / "mosaic" / "residual_field" / "local")
    assert resolved_env == os.fspath(env_root / "mosaic" / "residual_field" / "local")
