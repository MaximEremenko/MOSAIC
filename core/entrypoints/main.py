from __future__ import annotations

import argparse
import json
import logging
import os
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


def _runtime_requests_gpu(runtime_info: dict) -> bool:
    for key in (
        "nufft_policy",
        "nufft_execution_policy",
        "scattering_nufft_policy",
        "residual_nufft_policy",
    ):
        raw = runtime_info.get(key)
        if raw is None:
            continue
        if str(raw).strip().lower().replace("_", "-") in {
            "gpu-required",
            "allow-fallback",
        }:
            return True
    return False


def main(
    run_file: str = "run_parameters.json",
    *,
    db_path: str | None = None,
    no_db_cache: bool = False,
) -> None:
    parameter_loading_service = ParameterLoadingService()
    run_settings, workflow_parameters = parameter_loading_service.load(run_file)
    parameter_loading_service.apply_runtime_settings(run_settings.runtime)
    run_dir = Path(workflow_parameters.struct_info.working_directory).resolve()
    set_log_dir_for_run(run_dir)

    runtime_info = workflow_parameters.runtime_info.to_mapping()
    if _runtime_requests_gpu(runtime_info):
        os.environ.setdefault("MOSAIC_DASK_GPU_RESOURCE", "1")
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
        "Runtime settings: backend=%s max_workers=%s threads_per_worker=%d processes=%s",
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
        expected_workers = run_settings.runtime.max_workers
        if not isinstance(expected_workers, int):
            # "auto" resolves at cluster build (one worker per visible GPU);
            # ask the same resolver so the wait matches what was spawned.
            from core.runtime.dask_client import _resolve_max_workers

            expected_workers = _resolve_max_workers(run_settings.runtime.backend)
        client.wait_for_workers(
            expected_workers,
            timeout=run_settings.runtime.wait_timeout,
        )

    try:
        build_default_workflow_service().run(
            run_settings=run_settings,
            workflow_parameters=workflow_parameters,
            client=client,
            db_path=db_path,
            no_db_cache=no_db_cache,
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
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite cache path, 'local', or ':memory:'. Defaults to output_dir cache DB.",
    )
    parser.add_argument(
        "--no-db-cache",
        action="store_true",
        help="Use manifest-only in-memory cache state and do not open SQLite.",
    )
    return parser


def build_publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic publish",
        description="Publish a completed private MOSAIC run to compatibility files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Processed output directory containing the private .mosaic run namespace.",
    )
    parser.add_argument(
        "--run-digest",
        required=True,
        help="Run digest to publish.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing valid public manifest from a different run.",
    )
    return parser


def publish_main(
    *,
    output_dir: str,
    run_digest: str,
    replace: bool = False,
) -> None:
    from core.workflow.publisher import publish_run

    publish_run(output_dir=output_dir, run_digest=run_digest, replace=replace)


def build_cleanup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic cleanup",
        description="Clean manifest-proven reclaimable artifacts for a MOSAIC run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Processed output directory containing the private .mosaic run namespace.",
    )
    parser.add_argument(
        "--run-digest",
        required=True,
        help="Run digest to clean.",
    )
    parser.add_argument(
        "--temp-file-grace-seconds",
        type=float,
        default=0.0,
        help="Keep temporary files newer than this many seconds.",
    )
    parser.add_argument(
        "--remove-superseded-run",
        action="store_true",
        help="Remove the private run namespace only when public_manifest.json proves it is not active.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the cleanup report as JSON.",
    )
    return parser


def cleanup_main(
    *,
    output_dir: str,
    run_digest: str,
    temp_file_grace_seconds: float = 0.0,
    remove_superseded_run: bool = False,
    as_json: bool = False,
) -> None:
    from core.storage.cleanup import cleanup_run_artifacts

    report = cleanup_run_artifacts(
        output_dir=output_dir,
        run_digest=run_digest,
        temp_file_grace_seconds=temp_file_grace_seconds,
        remove_superseded_run=remove_superseded_run,
    )
    payload = {
        "removed_paths": list(report.removed_paths),
        "retained_paths": list(report.retained_paths),
        "skipped_reasons": list(report.skipped_reasons),
    }
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    sys.stdout.write(
        "Cleanup removed "
        f"{len(report.removed_paths)} path(s), retained {len(report.retained_paths)} "
        f"path(s), skipped {len(report.skipped_reasons)} action(s).\n"
    )


def cli(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "publish":
        args = build_publish_parser().parse_args(argv[1:])
        publish_main(
            output_dir=args.output_dir,
            run_digest=args.run_digest,
            replace=args.replace,
        )
        return 0
    if argv and argv[0] == "cleanup":
        args = build_cleanup_parser().parse_args(argv[1:])
        cleanup_main(
            output_dir=args.output_dir,
            run_digest=args.run_digest,
            temp_file_grace_seconds=args.temp_file_grace_seconds,
            remove_superseded_run=args.remove_superseded_run,
            as_json=args.json,
        )
        return 0
    args = build_parser().parse_args(argv)
    freeze_support()
    main(args.run_file, db_path=args.db_path, no_db_cache=args.no_db_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
