from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing import freeze_support
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.config import ParameterLoadingService
from core import __version__
from core.workflow import build_default_workflow_service
from core.runtime import (
    configure_progress,
    get_client,
    set_log_dir_for_run,
    setup_logging,
    short_path,
    shutdown_dask,
)


def main(run_file: str = "run_parameters.json") -> None:
    parameter_loading_service = ParameterLoadingService()
    run_settings, workflow_parameters = parameter_loading_service.load(run_file)
    parameter_loading_service.apply_runtime_settings(run_settings.runtime)
    run_dir = Path(workflow_parameters.struct_info.working_directory).resolve()
    set_log_dir_for_run(run_dir)

    runtime_info = workflow_parameters.runtime_info.to_mapping()
    progress_cfg = runtime_info.get("progress") or {}
    if not isinstance(progress_cfg, dict):
        progress_cfg = {}
    force_progress = progress_cfg.get(
        "force",
        runtime_info.get("force_progress"),
    )
    task_progress = progress_cfg.get(
        "task_logs",
        runtime_info.get("task_progress"),
    )
    configure_progress(
        force_progress=(
            None if force_progress is None else bool(force_progress)
        ),
        task_progress=(
            None if task_progress is None else bool(task_progress)
        ),
    )

    setup_logging()
    log = logging.getLogger("app")
    log.info("Using input parameters: %s", short_path(run_settings.input_parameters_path))
    log.info("Resolved configuration root: %s", short_path(run_settings.config_root))
    log.info(
        "Runtime settings: backend=%s max_workers=%d threads_per_worker=%d processes=%s",
        run_settings.runtime.backend,
        run_settings.runtime.max_workers,
        run_settings.runtime.threads_per_worker,
        run_settings.runtime.processes,
    )

    client = get_client()
    if client is not None:
        if force_progress is not None or task_progress is not None:
            try:
                client.run(
                    configure_progress,
                    force_progress=(
                        None if force_progress is None else bool(force_progress)
                    ),
                    task_progress=(
                        None if task_progress is None else bool(task_progress)
                    ),
                )
            except Exception:
                pass
        client.wait_for_workers(
            run_settings.runtime.max_workers,
            timeout=run_settings.runtime.wait_timeout,
        )

    try:
        build_default_workflow_service().run(
            run_settings=run_settings,
            workflow_parameters=workflow_parameters,
            client=client,
        )
    finally:
        shutdown_dask()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description="Run the MOSAIC scientific-stage workflow.",
    )
    parser.add_argument(
        "run_file",
        nargs="?",
        default="run_parameters.json",
        help="Path to run_parameters.json. Defaults to ./run_parameters.json.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_support()
    main(args.run_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
