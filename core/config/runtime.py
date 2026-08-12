from __future__ import annotations

import os

from core.config.values import as_bool, first_present
from core.models import RuntimeSettings, WorkflowParameters
from core.scattering.form_factors.contracts import ScatteringWeightSelection


def apply_runtime_settings(runtime_settings: RuntimeSettings) -> None:
    os.environ["DASK_WORKER_DASHBOARD"] = "1" if runtime_settings.worker_dashboard else "0"
    os.environ["DASK_BACKEND"] = runtime_settings.backend
    os.environ["DASK_MAX_WORKERS"] = str(runtime_settings.max_workers)
    os.environ["DASK_THREADS_PER_WORKER"] = str(runtime_settings.threads_per_worker)
    os.environ["DASK_PROCESSES"] = "1" if runtime_settings.processes else "0"


def resolve_scattering_weight_settings(
    workflow_parameters: WorkflowParameters,
) -> ScatteringWeightSelection:
    return ScatteringWeightSelection(
        kind=workflow_parameters.runtime_info.scattering_weights.kind,
        calculator=workflow_parameters.runtime_info.scattering_weights.calculator,
    )


def _resolve_max_workers_setting(raw) -> "int | str":
    """Portable worker count: an integer is explicit; absent or "auto" defers
    to the runtime, which sizes one worker per visible GPU on cuda-local
    (node-size independent — correct on a 1-GPU laptop and an 8-GPU node)."""
    if raw is None:
        return "auto"
    text = str(raw).strip().lower()
    if text in {"auto", ""}:
        return "auto"
    return int(raw)


def resolve_runtime_settings(parameters: dict) -> RuntimeSettings:
    runtime = first_present(parameters, ("runtime", "runtime_info", "runtimeInfo")) or {}
    dask = first_present(runtime, ("dask", "dask_info", "daskInfo")) or {}
    return RuntimeSettings(
        worker_dashboard=as_bool(
            first_present(dask, ("worker_dashboard", "dashboard", "dask_worker_dashboard")),
            default=False,
        ),
        backend=str(first_present(dask, ("backend", "dask_backend")) or "local"),
        max_workers=_resolve_max_workers_setting(
            first_present(dask, ("max_workers", "dask_max_workers"))
        ),
        threads_per_worker=int(
            first_present(dask, ("threads_per_worker", "dask_threads_per_worker")) or 16
        ),
        processes=as_bool(
            first_present(dask, ("processes", "dask_processes")),
            default=False,
        ),
        wait_timeout=str(
            first_present(
                dask, ("worker_wait_timeout", "wait_timeout", "dask_wait_timeout")
            )
            or "120s"
        ),
    )
