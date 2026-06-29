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
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Release evidence bundle layout
# ---------------------------------------------------------------------------
#
# ``final_plan.md`` section 13 defines an explicit, auditable bundle layout::
#
#     release_evidence/<version>/
#       commit.txt
#       ci_urls.txt
#       dist_sha256.txt
#       pip_freeze.txt
#       environment.json
#       fs_capability.json
#       public_manifest.json
#       stage_commits/
#       chunk_commits/
#       decoder_commit.json
#       restart_recovery.log
#       cross_host_visibility.log
#       gpu_validation_report.json
#       hpc_runbook_signoff.txt
#
# This script emits that layout wherever inputs are available and keeps
# ``release_evidence.json`` as the canonical *index*: it lists every expected
# path with present/missing status, the hash where applicable, and whether the
# item is required for the claimed release scope. The bundle is auditable
# without opening ``release_evidence.json`` (each piece of evidence is a real
# file on disk); the index exists so an auditor can verify completeness and so a
# release gate can fail when required evidence is missing.

# Scopes describe how much of the release is being claimed. An evidence item is
# "required" when the claimed scope appears in its ``required_scopes`` set.
#
#   local   - packaging / source-tree evidence derivable on the build host.
#   runtime - adds evidence produced by a real MOSAIC run (durable run state).
#   hpc     - adds multi-node / GPU site evidence that cannot be derived locally.
#   full    - every listed item is required (a complete release candidate).
SCOPES = ("local", "runtime", "hpc", "full")


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
            "stdout": completed.stdout,
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
            "stdout": "",
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        ended_at = datetime.now(timezone.utc)
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "command": command,
            "returncode": 124,
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "stdout": stdout,
            "stdout_tail": stdout[-8000:],
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
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return entries


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pip_freeze(repo_root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    result = _run_command(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=repo_root,
        timeout_seconds=timeout_seconds,
    )
    return result


def _environment_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "schema": "mosaic.release_evidence.environment",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": repo_root.as_posix(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "installed_packages": _installed_packages(),
    }


class BundleWriter:
    """Materialise the explicit bundle layout and build the index entries.

    Every expected layout item is registered through one of the ``record_*``
    helpers. Each call appends one entry to :attr:`index` describing where the
    file lives, whether it is present, its hash, and whether it is required for
    the claimed scope. Items that cannot be produced (no input supplied, or an
    input path that does not exist) are recorded as explicitly missing rather
    than silently omitted.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        evidence_dir: Path,
        scope: str,
    ) -> None:
        self.repo_root = repo_root
        self.evidence_dir = evidence_dir
        self.scope = scope
        self.index: list[dict[str, Any]] = []

    # -- low level ---------------------------------------------------------
    def _relpath(self, target: Path) -> str:
        try:
            return target.relative_to(self.evidence_dir).as_posix()
        except ValueError:
            return target.as_posix()

    def _is_required(self, required_scopes: Iterable[str]) -> bool:
        required = set(required_scopes)
        return self.scope == "full" or self.scope in required

    def _base_entry(
        self,
        rel_path: str,
        *,
        kind: str,
        required_scopes: Iterable[str],
    ) -> dict[str, Any]:
        return {
            "path": rel_path,
            "kind": kind,
            "required": self._is_required(required_scopes),
            "required_scopes": sorted(set(required_scopes)),
        }

    def _present_entry(
        self,
        target: Path,
        *,
        kind: str,
        required_scopes: Iterable[str],
        source: str | None,
    ) -> dict[str, Any]:
        entry = self._base_entry(
            self._relpath(target),
            kind=kind,
            required_scopes=required_scopes,
        )
        entry.update(
            {
                "present": True,
                "missing_reason": None,
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
                "source": source,
            }
        )
        return entry

    def _missing_entry(
        self,
        rel_path: str,
        *,
        kind: str,
        required_scopes: Iterable[str],
        missing_reason: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        entry = self._base_entry(rel_path, kind=kind, required_scopes=required_scopes)
        entry.update(
            {
                "present": False,
                "missing_reason": missing_reason,
                "sha256": None,
                "size_bytes": None,
                "source": source,
            }
        )
        return entry

    # -- recorders ---------------------------------------------------------
    def write_text(
        self,
        rel_path: str,
        content: str | None,
        *,
        required_scopes: Iterable[str],
        missing_reason: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Write locally-derived text evidence, or record it as missing."""
        if content is None:
            entry = self._missing_entry(
                rel_path,
                kind="text",
                required_scopes=required_scopes,
                missing_reason=missing_reason or "not available",
                source=source,
            )
            self.index.append(entry)
            return entry
        target = self.evidence_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content if content.endswith("\n") else content + "\n"
        target.write_text(text, encoding="utf-8")
        entry = self._present_entry(
            target,
            kind="text",
            required_scopes=required_scopes,
            source=source,
        )
        self.index.append(entry)
        return entry

    def write_json(
        self,
        rel_path: str,
        payload: Any,
        *,
        required_scopes: Iterable[str],
        source: str | None = None,
    ) -> dict[str, Any]:
        target = self.evidence_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        entry = self._present_entry(
            target,
            kind="json",
            required_scopes=required_scopes,
            source=source,
        )
        self.index.append(entry)
        return entry

    def copy_input(
        self,
        rel_path: str,
        source_path: Path | None,
        *,
        kind: str,
        required_scopes: Iterable[str],
        missing_reason: str,
    ) -> dict[str, Any]:
        """Copy an externally-supplied input into the bundle, or mark missing.

        ``source_path is None`` means the input was never supplied on the CLI
        (e.g. an HPC artifact that requires a real site run). A supplied path
        that does not exist is also recorded as missing, with the source noted
        so the auditor can see what was attempted.
        """
        if source_path is None:
            entry = self._missing_entry(
                rel_path,
                kind=kind,
                required_scopes=required_scopes,
                missing_reason=missing_reason,
            )
            self.index.append(entry)
            return entry
        if not source_path.exists():
            entry = self._missing_entry(
                rel_path,
                kind=kind,
                required_scopes=required_scopes,
                missing_reason="supplied path does not exist",
                source=source_path.as_posix(),
            )
            self.index.append(entry)
            return entry
        target = self.evidence_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        entry = self._present_entry(
            target,
            kind=kind,
            required_scopes=required_scopes,
            source=source_path.as_posix(),
        )
        self.index.append(entry)
        return entry

    def record_directory(
        self,
        rel_path: str,
        members: list[dict[str, Any]],
        *,
        required_scopes: Iterable[str],
        missing_reason: str,
    ) -> dict[str, Any]:
        """Record a directory of copied artifacts (stage/chunk commits).

        The directory is created even when empty so the layout is stable. It is
        considered present only when at least one member was copied in.
        """
        target = self.evidence_dir / rel_path
        target.mkdir(parents=True, exist_ok=True)
        entry = self._base_entry(
            rel_path.rstrip("/") + "/",
            kind="directory",
            required_scopes=required_scopes,
        )
        present = bool(members)
        entry.update(
            {
                "present": present,
                "missing_reason": None if present else missing_reason,
                "member_count": len(members),
                "members": members,
            }
        )
        self.index.append(entry)
        return entry


def _resolve_input(raw_path: str | None, repo_root: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _read_ci_urls(args_urls: list[str], ci_urls_file: Path | None) -> str | None:
    urls: list[str] = []
    if ci_urls_file is not None and ci_urls_file.exists():
        for line in ci_urls_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                urls.append(stripped)
    for raw in args_urls:
        stripped = raw.strip()
        if stripped:
            urls.append(stripped)
    if not urls:
        return None
    return "\n".join(urls)


def _stage_chunk_members(
    writer: BundleWriter,
    run_root: Path,
    *,
    required_scopes: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_members: list[dict[str, Any]] = []
    chunk_members: list[dict[str, Any]] = []
    if not run_root.exists():
        return stage_members, chunk_members
    for path in sorted(run_root.glob("*/stage_commit.json")):
        stage = path.parent.name
        rel = f"stage_commits/{stage}_stage_commit.json"
        target = writer.evidence_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        stage_members.append(
            {
                "path": rel,
                "source": path.as_posix(),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    for path in sorted(run_root.glob("*/chunks/chunk_*/chunk_commit.json")):
        stage = path.parents[2].name
        chunk = path.parent.name
        rel = f"chunk_commits/{stage}_{chunk}_chunk_commit.json"
        target = writer.evidence_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        chunk_members.append(
            {
                "path": rel,
                "source": path.as_posix(),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    return stage_members, chunk_members


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


def build_bundle(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Materialise the bundle layout and return the canonical index payload."""
    label = args.label or _utc_timestamp()
    evidence_dir = (repo_root / args.output_dir / label).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = (repo_root / args.dist_dir).resolve()
    timeout = max(1, int(args.command_timeout_seconds))

    writer = BundleWriter(repo_root=repo_root, evidence_dir=evidence_dir, scope=args.scope)

    # -- Locally derivable, always-required-for-local evidence -------------
    git = _git_evidence(repo_root)
    commit_sha = git["sha"]
    commit_text = None
    if commit_sha:
        dirty_suffix = " (dirty)" if git["dirty"] else ""
        commit_text = f"{commit_sha}{dirty_suffix}"
    writer.write_text(
        "commit.txt",
        commit_text,
        required_scopes={"local", "runtime", "hpc"},
        missing_reason="git rev-parse HEAD failed",
        source="git rev-parse HEAD",
    )

    ci_urls_file = _resolve_input(args.ci_urls_file, repo_root)
    ci_urls = _read_ci_urls(args.ci_url, ci_urls_file)
    writer.write_text(
        "ci_urls.txt",
        ci_urls,
        required_scopes={"local", "runtime", "hpc"},
        missing_reason="no CI run URLs supplied (--ci-url / --ci-urls-file)",
        source="--ci-url / --ci-urls-file",
    )

    dist_files = _dist_hashes(dist_dir)
    dist_text = None
    if dist_files:
        dist_text = "\n".join(f"{entry['sha256']}  {entry['name']}" for entry in dist_files)
    writer.write_text(
        "dist_sha256.txt",
        dist_text,
        required_scopes={"local"},
        missing_reason=f"no distribution artifacts under {dist_dir.as_posix()}",
        source=f"sha256({dist_dir.as_posix()}/*)",
    )

    freeze = _pip_freeze(repo_root, timeout_seconds=timeout)
    freeze_text = freeze["stdout"] if freeze["returncode"] == 0 else None
    writer.write_text(
        "pip_freeze.txt",
        freeze_text,
        required_scopes={"local"},
        missing_reason="pip freeze failed",
        source="python -m pip freeze --all",
    )

    writer.write_json(
        "environment.json",
        _environment_payload(repo_root),
        required_scopes={"local"},
        source="platform/python introspection",
    )

    # -- Run evidence (requires a real run output dir) ---------------------
    run_root: Path | None = None
    run_output_dir: Path | None = None
    if args.run_output_dir is not None or args.run_digest is not None:
        if args.run_output_dir is None or args.run_digest is None:
            raise SystemExit("--run-output-dir and --run-digest must be provided together.")
        run_output_dir = _resolve_input(args.run_output_dir, repo_root)
        assert run_output_dir is not None
        run_output_dir = run_output_dir.resolve()
        run_root = run_output_dir / ".mosaic" / "runs" / str(args.run_digest)

    run_missing = "no run output dir supplied (--run-output-dir / --run-digest)"

    writer.copy_input(
        "fs_capability.json",
        (run_root / "fs_capability.json") if run_root is not None else None,
        kind="json",
        required_scopes={"runtime", "hpc"},
        missing_reason=run_missing,
    )
    writer.copy_input(
        "public_manifest.json",
        (run_output_dir / "public_manifest.json") if run_output_dir is not None else None,
        kind="json",
        required_scopes={"runtime", "hpc"},
        missing_reason=run_missing,
    )
    writer.copy_input(
        "decoder_commit.json",
        (run_root / "decoding" / "decoder_commit.json") if run_root is not None else None,
        kind="json",
        required_scopes={"runtime"},
        missing_reason=run_missing,
    )

    stage_members, chunk_members = (
        _stage_chunk_members(writer, run_root, required_scopes={"runtime"})
        if run_root is not None
        else ([], [])
    )
    writer.record_directory(
        "stage_commits",
        stage_members,
        required_scopes={"runtime"},
        missing_reason=run_missing if run_root is None else "no stage_commit.json files under run root",
    )
    writer.record_directory(
        "chunk_commits",
        chunk_members,
        required_scopes={"runtime"},
        missing_reason=run_missing if run_root is None else "no chunk_commit.json files under run root",
    )

    # -- HPC / GPU site evidence (cannot be derived locally) ---------------
    writer.copy_input(
        "restart_recovery.log",
        _resolve_input(args.restart_recovery_log, repo_root),
        kind="log",
        required_scopes={"hpc"},
        missing_reason="requires a real HPC restart/recovery run (--restart-recovery-log)",
    )
    writer.copy_input(
        "cross_host_visibility.log",
        _resolve_input(args.cross_host_visibility_log, repo_root),
        kind="log",
        required_scopes={"hpc"},
        missing_reason="requires a multi-host run (--cross-host-visibility-log)",
    )
    writer.copy_input(
        "gpu_validation_report.json",
        _resolve_input(args.gpu_validation_report, repo_root),
        kind="json",
        required_scopes={"hpc"},
        missing_reason="requires a GPU validation run (--gpu-validation-report)",
    )
    writer.copy_input(
        "hpc_runbook_signoff.txt",
        _resolve_input(args.hpc_runbook_signoff, repo_root),
        kind="text",
        required_scopes={"hpc"},
        missing_reason="requires a human HPC runbook signoff (--hpc-runbook-signoff)",
    )

    # -- Optional standard commands + hpc smoke outputs --------------------
    command_results: dict[str, Any] = {}
    if args.run_standard_commands:
        for name, command in _standard_commands(repo_root):
            command_results[name] = _run_command(command, cwd=repo_root, timeout_seconds=timeout)

    hpc_smoke_outputs = []
    for raw_path in args.hpc_smoke_json:
        path = _resolve_input(raw_path, repo_root)
        assert path is not None
        hpc_smoke_outputs.append(
            {
                "path": path.as_posix(),
                "present": path.exists(),
                "content": _load_json_file(path) if path.exists() else None,
            }
        )

    missing_required = [
        entry for entry in writer.index if entry["required"] and not entry["present"]
    ]
    failed_commands = {
        name: result
        for name, result in command_results.items()
        if int(result.get("returncode", 1)) != 0
    }

    index = {
        "schema": "mosaic.release_evidence",
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": repo_root.as_posix(),
        "evidence_dir": evidence_dir.as_posix(),
        "label": label,
        "scope": args.scope,
        "git": git,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "dist_files": dist_files,
        "bundle_index": writer.index,
        "missing_required": [entry["path"] for entry in missing_required],
        "complete_for_scope": not missing_required,
        "command_results": command_results,
        "hpc_smoke_outputs": hpc_smoke_outputs,
    }

    evidence_path = evidence_dir / "release_evidence.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "index": index,
        "evidence_path": evidence_path,
        "missing_required": missing_required,
        "failed_commands": failed_commands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the MOSAIC release evidence bundle (explicit layout + index)."
    )
    parser.add_argument("--output-dir", default="release_evidence")
    parser.add_argument("--label", default=None)
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="local",
        help=(
            "Release scope being claimed. Items required for this scope that are "
            "missing fail the gate (unless --no-fail-on-missing-required)."
        ),
    )
    parser.add_argument(
        "--ci-url",
        action="append",
        default=[],
        help="A CI run URL to record in ci_urls.txt (repeatable).",
    )
    parser.add_argument(
        "--ci-urls-file",
        default=None,
        help="Path to a newline-delimited file of CI run URLs.",
    )
    parser.add_argument("--hpc-smoke-json", action="append", default=[])
    parser.add_argument("--run-output-dir", default=None)
    parser.add_argument("--run-digest", default=None)
    parser.add_argument("--restart-recovery-log", default=None)
    parser.add_argument("--cross-host-visibility-log", default=None)
    parser.add_argument("--gpu-validation-report", default=None)
    parser.add_argument("--hpc-runbook-signoff", default=None)
    parser.add_argument("--run-standard-commands", action="store_true")
    parser.add_argument("--fail-on-command-error", action="store_true")
    parser.add_argument(
        "--no-fail-on-missing-required",
        dest="fail_on_missing_required",
        action="store_false",
        help="Do not fail the gate when required evidence is missing for the scope.",
    )
    parser.set_defaults(fail_on_missing_required=True)
    parser.add_argument("--command-timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    result = build_bundle(args, repo_root)

    missing_required = result["missing_required"]
    failed_commands = result["failed_commands"]
    ok = not missing_required and not failed_commands
    print(
        json.dumps(
            {
                "ok": ok,
                "scope": args.scope,
                "path": result["evidence_path"].as_posix(),
                "complete_for_scope": not missing_required,
                "missing_required": [entry["path"] for entry in missing_required],
                "failed_commands": sorted(failed_commands),
            },
            indent=2,
        )
    )

    if missing_required and args.fail_on_missing_required:
        return 2
    if failed_commands and args.fail_on_command_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
