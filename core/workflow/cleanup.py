from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from core.residual_field.commit import (
    RESIDUAL_FIELD_STAGE,
    ResidualChunkCommitManifest,
    ResidualCommitCandidateManifest,
    ResidualStageCommitManifest,
    discover_residual_attempts,
    discover_residual_commit_candidates,
    validate_residual_commit_candidate,
)
from core.scattering.commit import (
    SCATTERING_STAGE,
    ScatteringChunkCommitManifest,
    ScatteringCommitCandidateManifest,
    ScatteringStageCommitManifest,
    discover_scattering_attempts,
    discover_scattering_commit_candidates,
    validate_scattering_commit_candidate,
)
from core.storage.atomic import assert_path_contained
from core.storage.attempt_store import chunk_commit_path, run_root, stage_commit_path
from core.storage.manifest import read_manifest
from core.workflow.publisher import public_manifest_path, validate_public_manifest_files


@dataclass(frozen=True)
class CleanupReport:
    removed_paths: tuple[str, ...]
    retained_paths: tuple[str, ...]
    skipped_reasons: tuple[str, ...]


def _relative(path: Path, *, output_dir: str | Path) -> str:
    return assert_path_contained(path, output_dir=output_dir).relative_to(
        Path(output_dir).resolve()
    ).as_posix()


def _remove_path(path: Path, *, output_dir: str | Path, removed: list[str]) -> None:
    contained = assert_path_contained(path, output_dir=output_dir)
    if contained.is_dir():
        shutil.rmtree(contained)
    else:
        contained.unlink()
    removed.append(_relative(contained, output_dir=output_dir))


def _is_temp_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".tmp") or ".tmp-" in name or ".tmp." in name


def _cleanup_temp_files(
    *,
    output_dir: str | Path,
    run_digest: str,
    grace_seconds: float,
    removed: list[str],
    retained: list[str],
) -> None:
    root = run_root(output_dir, run_digest)
    if not root.exists():
        return
    now = time.time()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_temp_file(path):
            continue
        if "failure" in path.parts:
            retained.append(_relative(path, output_dir=output_dir))
            continue
        age = now - path.stat().st_mtime
        if age < grace_seconds:
            retained.append(_relative(path, output_dir=output_dir))
            continue
        _remove_path(path, output_dir=output_dir, removed=removed)


def _cleanup_scattering_chunk(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    removed: list[str],
    retained: list[str],
    skipped: list[str],
) -> None:
    commit_path = chunk_commit_path(output_dir, run_digest, SCATTERING_STAGE, chunk_id)
    commit = read_manifest(
        commit_path,
        codec=ScatteringChunkCommitManifest,
        output_dir=output_dir,
    )
    selected = read_manifest(
        Path(output_dir) / commit.candidate_manifest_path,
        codec=ScatteringCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_scattering_commit_candidate(selected, output_dir=output_dir)

    try:
        candidates = discover_scattering_commit_candidates(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=chunk_id,
        )
        valid_candidates = tuple(
            validate_scattering_commit_candidate(candidate, output_dir=output_dir)
            for candidate in candidates
        )
    except Exception as exc:
        skipped.append(f"scattering chunk {chunk_id}: retained candidates after validation error: {exc}")
        return

    # P11/W2.3 ADDRESS-BASED GC: the durable chunk commit already names the
    # winning candidate (``selected.candidate_id``). Identity is address-based --
    # ``candidate_id`` is derived from the chunk plus the device-independent
    # work-unit addresses of its selected attempts, NOT from payload bytes -- and
    # per-attempt byte integrity is verified separately at load (validation above).
    # So any OTHER valid candidate is SUPERSEDED and safe to delete REGARDLESS of
    # payload bytes. (This replaces the pre-P11 bitwise gate that retained ALL
    # candidates whenever their payload_sha256 differed; genuinely divergent /
    # invalid candidates are rejected upstream by the agreement gate before
    # cleanup, and validation errors above still skip/retain.)
    for candidate in valid_candidates:
        if candidate.candidate_id == selected.candidate_id:
            continue
        candidate_dir = Path(output_dir) / candidate.payload_path
        _remove_path(candidate_dir.parent, output_dir=output_dir, removed=removed)

    selected_attempt_keys = {
        (
            int(item["interval_id"]),
            str(item["work_unit_digest"]),
            str(item["payload_sha256"]),
        )
        for item in selected.selected_attempts
    }
    selected_attempt_ids = {
        (
            int(item["interval_id"]),
            str(item["work_unit_digest"]),
            str(item["attempt_id"]),
        )
        for item in selected.selected_attempts
    }
    try:
        attempts = discover_scattering_attempts(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=chunk_id,
        )
    except Exception as exc:
        skipped.append(f"scattering chunk {chunk_id}: retained attempts after validation error: {exc}")
        return
    for attempt in attempts:
        identity = (int(attempt.interval_id), attempt.work_unit_digest, attempt.attempt_id)
        payload_key = (
            int(attempt.interval_id),
            attempt.work_unit_digest,
            attempt.payload_sha256,
        )
        if identity in selected_attempt_ids:
            continue
        attempt_path = Path(output_dir) / attempt.payload_path
        if payload_key in selected_attempt_keys:
            _remove_path(attempt_path.parent, output_dir=output_dir, removed=removed)
        else:
            retained.append(_relative(attempt_path.parent, output_dir=output_dir))


def _cleanup_residual_chunk(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    removed: list[str],
    retained: list[str],
    skipped: list[str],
) -> None:
    commit_path = chunk_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id)
    commit = read_manifest(
        commit_path,
        codec=ResidualChunkCommitManifest,
        output_dir=output_dir,
    )
    selected = read_manifest(
        Path(output_dir) / commit.candidate_manifest_path,
        codec=ResidualCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_residual_commit_candidate(selected, output_dir=output_dir)

    try:
        candidates = discover_residual_commit_candidates(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=chunk_id,
        )
        valid_candidates = tuple(
            validate_residual_commit_candidate(candidate, output_dir=output_dir)
            for candidate in candidates
        )
    except Exception as exc:
        skipped.append(f"residual chunk {chunk_id}: retained candidates after validation error: {exc}")
        return

    # P11/W2.3 ADDRESS-BASED GC: the durable chunk commit already names the
    # winning candidate (``selected.candidate_id``). Identity is address-based --
    # ``candidate_id`` is derived from the chunk plus the device-independent
    # work-unit addresses of its selected attempts, NOT from payload bytes -- and
    # per-attempt byte integrity is verified separately at load (validation above).
    # So any OTHER valid candidate is SUPERSEDED and safe to delete REGARDLESS of
    # payload bytes. (This replaces the pre-P11 bitwise gate that retained ALL
    # candidates whenever their payload_sha256 differed; genuinely divergent /
    # invalid candidates are rejected upstream by the agreement gate before
    # cleanup, and validation errors above still skip/retain.)
    for candidate in valid_candidates:
        if candidate.candidate_id == selected.candidate_id:
            continue
        candidate_dir = Path(output_dir) / candidate.payload_path
        _remove_path(candidate_dir.parent, output_dir=output_dir, removed=removed)

    selected_attempt_keys = {
        (
            int(item["partition_id"]),
            str(item["work_unit_digest"]),
            str(item["payload_sha256"]),
        )
        for item in selected.selected_attempts
    }
    selected_attempt_ids = {
        (
            int(item["partition_id"]),
            str(item["work_unit_digest"]),
            str(item["attempt_id"]),
        )
        for item in selected.selected_attempts
    }
    try:
        attempts = discover_residual_attempts(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=chunk_id,
        )
    except Exception as exc:
        skipped.append(f"residual chunk {chunk_id}: retained attempts after validation error: {exc}")
        return
    for attempt in attempts:
        identity = (int(attempt.partition_id), attempt.work_unit_digest, attempt.attempt_id)
        payload_key = (
            int(attempt.partition_id),
            attempt.work_unit_digest,
            attempt.payload_sha256,
        )
        if identity in selected_attempt_ids:
            continue
        attempt_path = Path(output_dir) / attempt.payload_path
        if payload_key in selected_attempt_keys:
            _remove_path(attempt_path.parent, output_dir=output_dir, removed=removed)
        else:
            retained.append(_relative(attempt_path.parent, output_dir=output_dir))


def _cleanup_committed_stage(
    *,
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    removed: list[str],
    retained: list[str],
    skipped: list[str],
) -> None:
    commit_path = stage_commit_path(output_dir, run_digest, stage)
    if not commit_path.exists():
        skipped.append(f"{stage}: no stage_commit.json; retained attempts and candidates")
        return
    if stage == SCATTERING_STAGE:
        stage_commit = read_manifest(
            commit_path,
            codec=ScatteringStageCommitManifest,
            output_dir=output_dir,
        )
        for chunk_id in stage_commit.chunk_ids:
            _cleanup_scattering_chunk(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_id=int(chunk_id),
                removed=removed,
                retained=retained,
                skipped=skipped,
            )
        return
    if stage == RESIDUAL_FIELD_STAGE:
        stage_commit = read_manifest(
            commit_path,
            codec=ResidualStageCommitManifest,
            output_dir=output_dir,
        )
        for chunk_id in stage_commit.chunk_ids:
            _cleanup_residual_chunk(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_id=int(chunk_id),
                removed=removed,
                retained=retained,
                skipped=skipped,
            )
        return
    raise ValueError(f"Unsupported cleanup stage: {stage!r}")


def _cleanup_superseded_run(
    *,
    output_dir: str | Path,
    run_digest: str,
    remove_superseded_run: bool,
    removed: list[str],
    skipped: list[str],
) -> None:
    if not remove_superseded_run:
        return
    active_manifest = validate_public_manifest_files(
        public_manifest_path(output_dir),
        output_dir=output_dir,
    )
    if active_manifest.run_digest == str(run_digest):
        skipped.append("active public run is retained")
        return
    root = run_root(output_dir, run_digest)
    if root.exists():
        _remove_path(root, output_dir=output_dir, removed=removed)


def cleanup_run_artifacts(
    output_dir: str | Path,
    run_digest: str,
    *,
    temp_file_grace_seconds: float = 0,
    remove_superseded_run: bool = False,
) -> CleanupReport:
    removed: list[str] = []
    retained: list[str] = []
    skipped: list[str] = []
    output_root = Path(output_dir).resolve()

    _cleanup_temp_files(
        output_dir=output_root,
        run_digest=run_digest,
        grace_seconds=float(temp_file_grace_seconds),
        removed=removed,
        retained=retained,
    )
    _cleanup_committed_stage(
        output_dir=output_root,
        run_digest=run_digest,
        stage=SCATTERING_STAGE,
        removed=removed,
        retained=retained,
        skipped=skipped,
    )
    _cleanup_committed_stage(
        output_dir=output_root,
        run_digest=run_digest,
        stage=RESIDUAL_FIELD_STAGE,
        removed=removed,
        retained=retained,
        skipped=skipped,
    )
    _cleanup_superseded_run(
        output_dir=output_root,
        run_digest=run_digest,
        remove_superseded_run=remove_superseded_run,
        removed=removed,
        skipped=skipped,
    )
    return CleanupReport(
        removed_paths=tuple(sorted(removed)),
        retained_paths=tuple(sorted(retained)),
        skipped_reasons=tuple(sorted(skipped)),
    )


__all__ = [
    "CleanupReport",
    "cleanup_run_artifacts",
]
