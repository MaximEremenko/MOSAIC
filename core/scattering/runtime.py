from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

from core.runtime import (
    quiet_loggers,
    resolve_nufft_execution_settings,
)


@contextmanager
def _quiet_db_info():
    with quiet_loggers("core.storage.database_manager", "DatabaseManager"):
        yield


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


# Nine further names were re-exported here for the stage2_replacement
# module deleted in 085fb82. Nothing has imported them through this module
# since; every consumer takes them from core.runtime directly.
__all__ = [
    "_nufft_execution_settings",
    "_quiet_db_info",
    "_require_scheduler_resource_capacity",
    "_runtime_info",
]
