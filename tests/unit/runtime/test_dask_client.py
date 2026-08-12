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


def _clear_auto_worker_env(monkeypatch):
    for var in (
        "DASK_MAX_WORKERS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "SLURM_GPUS",
        "SLURM_JOB_NUM_NODES",
    ):
        monkeypatch.delenv(var, raising=False)


def test_build_job_extra_slurm_has_no_sge_flags(monkeypatch, tmp_path):
    from core.runtime.dask_client import _build_job_extra

    for var in ("DASK_GPU", "DASK_PE", "DASK_HOST"):
        monkeypatch.delenv(var, raising=False)
    extras = _build_job_extra(tmp_path, "slurm")
    assert "-cwd" not in extras
    assert "-V" not in extras
    assert not any("$JOB_ID" in item for item in extras)
    assert f"--output={tmp_path}/worker.o.%j" in extras
    assert f"--error={tmp_path}/worker.e.%j" in extras


def test_build_job_extra_sge_keeps_legacy_flags(monkeypatch, tmp_path):
    from core.runtime.dask_client import _build_job_extra

    for var in ("DASK_GPU", "DASK_PE", "DASK_HOST"):
        monkeypatch.delenv(var, raising=False)
    extras = _build_job_extra(tmp_path, "sge")
    assert "-cwd" in extras
    assert "-V" in extras
    assert f"-o {tmp_path}/worker.o.$JOB_ID.$TASK_ID" in extras


def test_build_job_extra_keeps_env_passthrough(monkeypatch, tmp_path):
    from core.runtime.dask_client import _build_job_extra

    monkeypatch.setenv("DASK_GPU", "--gres=gpu:2")
    monkeypatch.delenv("DASK_PE", raising=False)
    monkeypatch.delenv("DASK_HOST", raising=False)
    extras = _build_job_extra(tmp_path, "slurm")
    assert "--gres=gpu:2" in extras


def test_resolve_max_workers_mpi_uses_world_size(monkeypatch):
    from core.runtime.dask_client import _resolve_max_workers

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "5")
    # rank 0 = scheduler, rank 1 = client
    assert _resolve_max_workers("mpi") == 3


def test_resolve_max_workers_mpi_falls_back_to_pmi_then_slurm(monkeypatch):
    from core.runtime.dask_client import _resolve_max_workers

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("PMI_SIZE", "8")
    assert _resolve_max_workers("mpi") == 6

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "3")
    assert _resolve_max_workers("mpi") == 1


def test_resolve_max_workers_mpi_never_below_one(monkeypatch):
    from core.runtime.dask_client import _resolve_max_workers

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")
    assert _resolve_max_workers("mpi") == 1


def test_resolve_max_workers_jobqueue_uses_scheduler_env(monkeypatch):
    from core.runtime.dask_client import _resolve_max_workers

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("SLURM_GPUS", "6")
    assert _resolve_max_workers("slurm") == 6

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
    assert _resolve_max_workers("slurm") == 3

    _clear_auto_worker_env(monkeypatch)
    assert _resolve_max_workers("slurm") == 4


def test_resolve_max_workers_explicit_integer_wins(monkeypatch):
    from core.runtime.dask_client import _resolve_max_workers

    _clear_auto_worker_env(monkeypatch)
    monkeypatch.setenv("DASK_MAX_WORKERS", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "5")
    assert _resolve_max_workers("mpi") == 7


def test_threads_per_worker_default_matches_helpers(monkeypatch, tmp_path):
    import core.runtime.dask_client as dask_client
    from core.runtime.dask_helpers import DEFAULT_DASK_THREADS_PER_WORKER

    captured = {}

    def fake_ensure(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(dask_client, "ensure_dask_client", fake_ensure)
    monkeypatch.setenv("MOSAIC_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("DASK_THREADS_PER_WORKER", raising=False)
    dask_client._CLIENT = None
    try:
        dask_client.get_client()
    finally:
        dask_client._CLIENT = None

    assert captured["threads_per_worker"] == DEFAULT_DASK_THREADS_PER_WORKER == 16
