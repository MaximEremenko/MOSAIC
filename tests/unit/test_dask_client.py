import os
from pathlib import Path

from core.infrastructure.runtime.dask_client import default_log_dir, set_log_dir_for_run


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
