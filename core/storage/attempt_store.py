from __future__ import annotations

from pathlib import Path

from core.storage.atomic import assert_path_contained


def _clean_component(value: str, *, label: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{label} must not be empty.")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"{label} contains an invalid path segment: {text!r}")
    return text


def _chunk_dir_name(chunk_id: int) -> str:
    return f"chunk_{int(chunk_id)}"


def _attempt_shard(work_unit_digest: str) -> str:
    clean = _clean_component(work_unit_digest, label="work_unit_digest")
    if len(clean) < 2:
        raise ValueError("work_unit_digest must contain at least two characters for sharding.")
    return clean[:2]


def run_root(output_dir: str | Path, run_digest: str) -> Path:
    root = Path(output_dir) / ".mosaic" / "runs" / _clean_component(run_digest, label="run_digest")
    return assert_path_contained(root, output_dir=output_dir)


def run_manifest_path(output_dir: str | Path, run_digest: str) -> Path:
    return run_root(output_dir, run_digest) / "run_manifest.json"


def qspace_plan_path(output_dir: str | Path, run_digest: str) -> Path:
    return run_root(output_dir, run_digest) / "qspace_plan.json"


def fs_capability_path(output_dir: str | Path, run_digest: str) -> Path:
    return run_root(output_dir, run_digest) / "fs_capability.json"


def stage_root(output_dir: str | Path, run_digest: str, stage: str) -> Path:
    return assert_path_contained(
        run_root(output_dir, run_digest) / _clean_component(stage, label="stage"),
        output_dir=output_dir,
    )


def stage_plan_path(output_dir: str | Path, run_digest: str, stage: str) -> Path:
    return stage_root(output_dir, run_digest, stage) / "stage_plan.json"


def stage_commit_path(output_dir: str | Path, run_digest: str, stage: str) -> Path:
    return stage_root(output_dir, run_digest, stage) / "stage_commit.json"


def chunk_root(output_dir: str | Path, run_digest: str, stage: str, chunk_id: int) -> Path:
    return assert_path_contained(
        stage_root(output_dir, run_digest, stage) / "chunks" / _chunk_dir_name(chunk_id),
        output_dir=output_dir,
    )


def chunk_commit_path(output_dir: str | Path, run_digest: str, stage: str, chunk_id: int) -> Path:
    return chunk_root(output_dir, run_digest, stage, chunk_id) / "chunk_commit.json"


def work_unit_attempts_root(
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    work_unit_digest: str,
) -> Path:
    digest = _clean_component(work_unit_digest, label="work_unit_digest")
    return assert_path_contained(
        chunk_root(output_dir, run_digest, stage, chunk_id)
        / "attempts"
        / _attempt_shard(digest)
        / digest,
        output_dir=output_dir,
    )


def attempt_root(
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    work_unit_digest: str,
    attempt_id: str,
) -> Path:
    return assert_path_contained(
        work_unit_attempts_root(output_dir, run_digest, stage, chunk_id, work_unit_digest)
        / f"attempt_{_clean_component(attempt_id, label='attempt_id')}",
        output_dir=output_dir,
    )


def attempt_payload_path(*args, **kwargs) -> Path:
    return attempt_root(*args, **kwargs) / "payload.hdf5"


def attempt_manifest_path(*args, **kwargs) -> Path:
    return attempt_root(*args, **kwargs) / "attempt.json"


def commit_attempts_root(output_dir: str | Path, run_digest: str, stage: str, chunk_id: int) -> Path:
    return chunk_root(output_dir, run_digest, stage, chunk_id) / "commit_attempts"


def commit_attempt_root(
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    commit_id: str,
) -> Path:
    return commit_attempts_root(output_dir, run_digest, stage, chunk_id) / (
        f"commit_{_clean_component(commit_id, label='commit_id')}"
    )


def commit_candidate_payload_path(*args, **kwargs) -> Path:
    return commit_attempt_root(*args, **kwargs) / "chunk_payload.hdf5"


def commit_candidate_manifest_path(*args, **kwargs) -> Path:
    return commit_attempt_root(*args, **kwargs) / "commit_candidate.json"


def relative_to_output(path: str | Path, *, output_dir: str | Path) -> str:
    contained = assert_path_contained(path, output_dir=output_dir)
    return contained.relative_to(Path(output_dir).resolve()).as_posix()


__all__ = [
    "chunk_commit_path",
    "attempt_manifest_path",
    "attempt_payload_path",
    "attempt_root",
    "chunk_root",
    "commit_attempt_root",
    "commit_attempts_root",
    "commit_candidate_manifest_path",
    "commit_candidate_payload_path",
    "fs_capability_path",
    "qspace_plan_path",
    "relative_to_output",
    "run_manifest_path",
    "run_root",
    "stage_commit_path",
    "stage_plan_path",
    "stage_root",
    "work_unit_attempts_root",
]
