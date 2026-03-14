from __future__ import annotations

import logging
import sys
from multiprocessing import freeze_support
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.application.configuration import ParameterLoadingService
from core.application.workflow import WorkflowService
from core.infrastructure.runtime import (
    get_client,
    set_log_dir_for_run,
    setup_logging,
    shutdown_dask,
)


def main(run_file: str = "run_parameters.json") -> None:
    parameter_loading_service = ParameterLoadingService()
    run_settings, workflow_parameters = parameter_loading_service.load(run_file)
    parameter_loading_service.apply_runtime_settings(run_settings.runtime)
    run_dir = Path(workflow_parameters.struct_info["working_directory"]).resolve()
    set_log_dir_for_run(run_dir)

    setup_logging()
    log = logging.getLogger("app")
    log.info("Using input parameters: %s", run_settings.input_parameters_path)
    log.info("Resolved configuration root: %s", run_settings.config_root)
    log.info(
        "Runtime settings: backend=%s max_workers=%d threads_per_worker=%d processes=%s",
        run_settings.runtime.backend,
        run_settings.runtime.max_workers,
        run_settings.runtime.threads_per_worker,
        run_settings.runtime.processes,
    )

    client = get_client()
    if client is not None:
        client.wait_for_workers(
            run_settings.runtime.max_workers,
            timeout=run_settings.runtime.wait_timeout,
        )

    try:
        WorkflowService().run(
            run_settings=run_settings,
            workflow_parameters=workflow_parameters,
            client=client,
        )
    finally:
        shutdown_dask()


if __name__ == "__main__":
    freeze_support()
    main()
