import os
from pathlib import Path

from core.runtime.dask_client import default_log_dir, set_log_dir_for_run


def test_default_log_dir_uses_run_directory_when_env_not_set(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSAIC_LOG_DIR", raising=False)
    log_dir = default_log_dir(tmp_path / "run")
    assert log_dir == Path(tmp_path / "run" / "dask_logs")


def test_set_log_dir_for_run_sets_env_once(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSAIC_LOG_DIR", raising=False)
    run_dir = tmp_path / "run"
    log_dir = set_log_dir_for_run(run_dir)
    assert os.environ["MOSAIC_LOG_DIR"] == str(run_dir / "dask_logs")
    assert log_dir == run_dir / "dask_logs"


def test_set_log_dir_for_run_preserves_explicit_env(monkeypatch, tmp_path):
    explicit = tmp_path / "custom_logs"
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(explicit))
    log_dir = set_log_dir_for_run(tmp_path / "run")
    assert os.environ["MOSAIC_LOG_DIR"] == str(explicit)
    assert log_dir == explicit


def test_get_client_defaults_nufft_supply_to_one(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DASK_THREADS_PER_WORKER", "4")
    monkeypatch.delenv("MOSAIC_NUFFT_SLOTS_PER_WORKER", raising=False)
    monkeypatch.delenv("MOSAIC_DASK_GPU_RESOURCE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("GPUS_PER_JOB", "0")
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["threads_per_worker"] == 4
    assert captured["resources"] == {"nufft": 1}


def test_get_client_honors_processes_env(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DASK_PROCESSES", "0")
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["processes"] is False


def test_get_client_defers_processes_to_config_when_env_unset(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("DASK_PROCESSES", raising=False)
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["processes"] is None


def test_get_client_allows_explicit_nufft_supply_override(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MOSAIC_NUFFT_SLOTS_PER_WORKER", "2")
    monkeypatch.delenv("MOSAIC_DASK_GPU_RESOURCE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("GPUS_PER_JOB", "0")
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["resources"] == {"nufft": 2}


def test_get_client_declares_gpu_resource_when_requested(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MOSAIC_DASK_GPU_RESOURCE", "1")
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["resources"] == {"nufft": 1, "gpu": 1}
