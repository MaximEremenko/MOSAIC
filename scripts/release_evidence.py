#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        ended_at = datetime.now(timezone.utc)
        return {
            "command": command,
            "returncode": completed.returncode,
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "stdout_tail": completed.stdout[-8000:],
            "stderr_tail": completed.stderr[-8000:],
        }
    except FileNotFoundError as exc:
        ended_at = datetime.now(timezone.utc)
        return {
            "command": command,
            "returncode": 127,
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        ended_at = datetime.now(timezone.utc)
        return {
            "command": command,
            "returncode": 124,
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }


def _git_evidence(repo_root: Path) -> dict[str, Any]:
    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout_seconds=30)
    status = _run_command(["git", "status", "--short", "--branch"], cwd=repo_root, timeout_seconds=30)
    porcelain = _run_command(["git", "status", "--porcelain"], cwd=repo_root, timeout_seconds=30)
    return {
        "sha": sha["stdout_tail"].strip() if sha["returncode"] == 0 else None,
        "status_short": status["stdout_tail"].splitlines(),
        "dirty": bool(porcelain["stdout_tail"].strip()),
        "commands": {
            "rev_parse_head": sha,
            "status_short_branch": status,
            "status_porcelain": porcelain,
        },
    }


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dist_hashes(dist_dir: Path) -> list[dict[str, Any]]:
    if not dist_dir.exists():
        return []
    entries = []
    for path in sorted(dist_dir.iterdir()):
        if path.is_file() and path.suffix in {".gz", ".whl", ".zip"}:
            entries.append(
                {
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return entries


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_evidence_file(
    source: Path,
    target: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if not source.exists():
        return {
            "source": source.as_posix(),
            "copied_to": None,
            "present": False,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    try:
        copied_to = target.relative_to(repo_root).as_posix()
    except ValueError:
        copied_to = target.as_posix()
    return {
        "source": source.as_posix(),
        "copied_to": copied_to,
        "present": True,
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
    }


def _copy_optional_input(
    raw_path: str | None,
    *,
    repo_root: Path,
    evidence_dir: Path,
    target_name: str,
) -> dict[str, Any] | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return _copy_evidence_file(
        path,
        evidence_dir / target_name,
        repo_root=repo_root,
    )


def _copy_run_artifacts(
    *,
    repo_root: Path,
    evidence_dir: Path,
    output_dir: Path,
    run_digest: str,
) -> dict[str, Any]:
    output_root = output_dir.resolve()
    run_root = output_root / ".mosaic" / "runs" / str(run_digest)
    copied: dict[str, Any] = {
        "output_dir": output_root.as_posix(),
        "run_digest": str(run_digest),
        "run_root": run_root.as_posix(),
        "run_root_present": run_root.exists(),
        "files": {},
        "stage_commits": [],
        "chunk_commits": [],
    }
    copied["files"]["fs_capability"] = _copy_evidence_file(
        run_root / "fs_capability.json",
        evidence_dir / "fs_capability.json",
        repo_root=repo_root,
    )
    copied["files"]["public_manifest"] = _copy_evidence_file(
        output_root / "public_manifest.json",
        evidence_dir / "public_manifest.json",
        repo_root=repo_root,
    )
    copied["files"]["decoder_commit"] = _copy_evidence_file(
        run_root / "decoding" / "decoder_commit.json",
        evidence_dir / "decoder_commit.json",
        repo_root=repo_root,
    )
    for path in sorted(run_root.glob("*/stage_commit.json")):
        stage = path.parent.name
        copied["stage_commits"].append(
            _copy_evidence_file(
                path,
                evidence_dir / "stage_commits" / f"{stage}_stage_commit.json",
                repo_root=repo_root,
            )
        )
    for path in sorted(run_root.glob("*/chunks/chunk_*/chunk_commit.json")):
        stage = path.parents[2].name
        chunk = path.parent.name
        copied["chunk_commits"].append(
            _copy_evidence_file(
                path,
                evidence_dir / "chunk_commits" / f"{stage}_{chunk}_chunk_commit.json",
                repo_root=repo_root,
            )
        )
    return copied


def _standard_commands(repo_root: Path) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = [
        ("pip_check", [sys.executable, "-m", "pip", "check"]),
    ]
    dist_paths = sorted(str(path) for path in (repo_root / "dist").iterdir()) if (repo_root / "dist").exists() else []
    if dist_paths:
        commands.append(("twine_check", [sys.executable, "-m", "twine", "check", *dist_paths]))
    mosaic_cli = shutil.which("mosaic")
    if mosaic_cli is not None:
        commands.append(("mosaic_help", [mosaic_cli, "--help"]))
    else:
        commands.append(("mosaic_help", ["mosaic", "--help"]))
    commands.append(("smoke_example", [sys.executable, "scripts/smoke_example.py"]))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a MOSAIC release evidence JSON bundle."
    )
    parser.add_argument("--output-dir", default="release_evidence")
    parser.add_argument("--label", default=None)
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--hpc-smoke-json", action="append", default=[])
    parser.add_argument("--run-output-dir", default=None)
    parser.add_argument("--run-digest", default=None)
    parser.add_argument("--restart-recovery-log", default=None)
    parser.add_argument("--cross-host-visibility-log", default=None)
    parser.add_argument("--gpu-validation-report", default=None)
    parser.add_argument("--hpc-runbook-signoff", default=None)
    parser.add_argument("--run-standard-commands", action="store_true")
    parser.add_argument("--fail-on-command-error", action="store_true")
    parser.add_argument("--command-timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    label = args.label or _utc_timestamp()
    evidence_dir = (repo_root / args.output_dir / label).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = (repo_root / args.dist_dir).resolve()

    command_results: dict[str, Any] = {}
    if args.run_standard_commands:
        for label_name, command in _standard_commands(repo_root):
            command_results[label_name] = _run_command(
                command,
                cwd=repo_root,
                timeout_seconds=max(1, int(args.command_timeout_seconds)),
            )

    hpc_smoke_outputs = []
    for raw_path in args.hpc_smoke_json:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        hpc_smoke_outputs.append(
            {
                "path": path.as_posix(),
                "content": _load_json_file(path),
            }
        )

    run_artifacts = None
    if args.run_output_dir is not None or args.run_digest is not None:
        if args.run_output_dir is None or args.run_digest is None:
            parser.error("--run-output-dir and --run-digest must be provided together.")
        run_output_dir = Path(args.run_output_dir)
        if not run_output_dir.is_absolute():
            run_output_dir = repo_root / run_output_dir
        run_artifacts = _copy_run_artifacts(
            repo_root=repo_root,
            evidence_dir=evidence_dir,
            output_dir=run_output_dir,
            run_digest=str(args.run_digest),
        )

    optional_evidence_files = {
        "restart_recovery_log": _copy_optional_input(
            args.restart_recovery_log,
            repo_root=repo_root,
            evidence_dir=evidence_dir,
            target_name="restart_recovery.log",
        ),
        "cross_host_visibility_log": _copy_optional_input(
            args.cross_host_visibility_log,
            repo_root=repo_root,
            evidence_dir=evidence_dir,
            target_name="cross_host_visibility.log",
        ),
        "gpu_validation_report": _copy_optional_input(
            args.gpu_validation_report,
            repo_root=repo_root,
            evidence_dir=evidence_dir,
            target_name="gpu_validation_report.json",
        ),
        "hpc_runbook_signoff": _copy_optional_input(
            args.hpc_runbook_signoff,
            repo_root=repo_root,
            evidence_dir=evidence_dir,
            target_name="hpc_runbook_signoff.txt",
        ),
    }

    evidence = {
        "schema": "mosaic.release_evidence",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": repo_root.as_posix(),
        "git": _git_evidence(repo_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "installed_packages": _installed_packages(),
        "dist_files": _dist_hashes(dist_dir),
        "command_results": command_results,
        "hpc_smoke_outputs": hpc_smoke_outputs,
        "run_artifacts": run_artifacts,
        "optional_evidence_files": optional_evidence_files,
    }
    evidence_path = evidence_dir / "release_evidence.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")

    failed_commands = {
        name: result
        for name, result in command_results.items()
        if int(result.get("returncode", 1)) != 0
    }
    print(json.dumps({"ok": not failed_commands, "path": evidence_path.as_posix()}, indent=2))
    if failed_commands and args.fail_on_command_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
