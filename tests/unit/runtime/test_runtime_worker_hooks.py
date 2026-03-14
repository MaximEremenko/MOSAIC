from __future__ import annotations

from types import SimpleNamespace

from core.runtime.worker_hooks import handle_worker_gpu_failure, register_cleanup_plugin


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
