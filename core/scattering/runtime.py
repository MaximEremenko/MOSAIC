from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from core.runtime import (
    DEFAULT_TASK_RETRIES,
    TIMER,
    chunk_mutex,
    is_sync_client as _is_sync_client,
    progress_bar as _tqdm,
    quiet_loggers,
    resolve_nufft_execution_settings,
    timed as _timed,
    yield_futures_with_results as _yield_futures_with_results,
)
from core.runtime.worker_hooks import CuPyCleanup
from core.scattering.tasks import IntervalPayloadRef
from core.storage.fingerprint import file_sha256


@contextmanager
def _quiet_db_info():
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        yield


def _current_worker_addresses(client) -> list[str]:
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return []
    return sorted(str(address) for address in workers)


def _require_scheduler_resource_capacity(client, resource_name: str) -> None:
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return
    if not workers:
        return
    total = 0.0
    for worker in workers.values():
        resources = worker.get("resources", {}) or {}
        try:
            total += float(resources.get(resource_name, 0.0))
        except (TypeError, ValueError):
            pass
    if total <= 0:
        raise RuntimeError(
            f"Dask scheduler reports zero total {resource_name!r} resource capacity; "
            "MOSAIC NUFFT tasks would stay queued or overbook. Configure worker "
            f"resources with {resource_name}=N."
        )


def _runtime_info(parameters: Dict[str, Any]) -> dict[str, Any]:
    runtime_info = parameters.get("runtime_info") or {}
    return runtime_info if isinstance(runtime_info, dict) else {}


def _nufft_execution_settings(parameters: Dict[str, Any]):
    runtime_info = _runtime_info(parameters)
    requested = runtime_info.get("scattering_nufft_policy")
    if requested is None:
        requested = runtime_info.get("nufft_execution_policy")
    if requested is None:
        requested = runtime_info.get("nufft_policy")
    eps = runtime_info.get("scattering_nufft_eps", runtime_info.get("nufft_eps", 1e-12))
    dtype = runtime_info.get("scattering_dtype", runtime_info.get("nufft_dtype", "complex128"))
    return resolve_nufft_execution_settings(requested, eps=eps, dtype=dtype)


def _call_accepts_kwarg(func, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _add_nufft_task_kwargs(func, kwargs: dict[str, Any], nufft_settings) -> None:
    for name, value in (
        ("nufft_eps", nufft_settings.eps),
        ("nufft_prefer_cpu", nufft_settings.prefer_cpu),
        ("nufft_gpu_only", nufft_settings.gpu_only),
    ):
        if _call_accepts_kwarg(func, name):
            kwargs[name] = value


def _interval_payload_input(
    interval_id: int | None,
    path: Path | str,
) -> IntervalPayloadRef | Path:
    payload_path = Path(path)
    if not payload_path.exists():
        return payload_path
    return IntervalPayloadRef(
        path=str(payload_path),
        file_sha256=file_sha256(payload_path),
        interval_id=None if interval_id is None else int(interval_id),
    )


__all__ = [
    "CuPyCleanup",
    "DEFAULT_TASK_RETRIES",
    "TIMER",
    "_add_nufft_task_kwargs",
    "_call_accepts_kwarg",
    "_current_worker_addresses",
    "_interval_payload_input",
    "_is_sync_client",
    "_nufft_execution_settings",
    "_quiet_db_info",
    "_require_scheduler_resource_capacity",
    "_runtime_info",
    "_timed",
    "_tqdm",
    "_yield_futures_with_results",
    "chunk_mutex",
]
