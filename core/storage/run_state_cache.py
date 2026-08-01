from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.decoding.commit import DecoderCommitManifest
from core.residual_field.commit import (
    RESIDUAL_FIELD_STAGE,
    ResidualChunkCommitManifest,
    ResidualCommitCandidateManifest,
    ResidualNoOutputManifest,
    ResidualStageCommitManifest,
    ResidualStagePlanManifest,
    discover_residual_attempts,
    require_residual_no_output_manifest,
    validate_residual_commit_candidate,
    write_residual_stage_commit,
)
from core.scattering.commit import (
    SCATTERING_STAGE,
    ScatteringChunkCommitManifest,
    ScatteringCommitCandidateManifest,
    ScatteringStageCommitManifest,
    ScatteringStagePlanManifest,
    validate_scattering_commit_candidate,
    write_scattering_stage_commit,
)
from core.storage.database_manager import DatabaseManagerProtocol
from core.storage.attempt_store import (
    chunk_commit_path,
    run_manifest_path,
    qspace_plan_path,
    stage_commit_path,
    stage_plan_path,
)
from core.storage.fingerprint import file_sha256
from core.storage.manifest import ManifestError, read_manifest, validate_manifest_payload
from core.storage.performance import write_performance_metrics


@dataclass(frozen=True)
class ChunkState:
    stage: str
    chunk_id: int
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int
    candidate_manifest_path: str
    candidate_payload_path: str
    interval_ids: tuple[int, ...] = ()
    partition_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class StageState:
    stage: str
    stage_plan_present: bool
    stage_commit_present: bool
    complete: bool
    expected_chunk_ids: tuple[int, ...]
    valid_chunk_ids: tuple[int, ...]
    missing_chunk_ids: tuple[int, ...]
    selected_payloads: tuple[ChunkState, ...]
    invalid_manifests: tuple[str, ...]
    no_output: bool = False


@dataclass(frozen=True)
class RunStateSnapshot:
    output_dir: str
    run_digest: str
    run_manifest_valid: bool
    qspace_plan_valid: bool
    scattering: StageState
    residual_field: StageState
    decoder_commit_valid: bool
    invalid_manifests: tuple[str, ...]

    @property
    def complete_stages(self) -> tuple[str, ...]:
        values = []
        if self.scattering.complete:
            values.append(SCATTERING_STAGE)
        if self.residual_field.complete or self.residual_field.no_output:
            values.append(RESIDUAL_FIELD_STAGE)
        if self.decoder_commit_valid:
            values.append("decoding")
        return tuple(values)


def _read_json_manifest(path: Path, *, expected_schema: str, output_dir: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_manifest_payload(
        payload,
        expected_schema=expected_schema,
        expected_version=1,
        output_dir=output_dir,
    )
    return True


def _expected_scattering_chunks(plan: ScatteringStagePlanManifest | None) -> tuple[int, ...]:
    if plan is None:
        return ()
    return tuple(sorted(int(item["chunk_id"]) for item in plan.expected_by_chunk))


def _expected_residual_chunks(plan: ResidualStagePlanManifest | None) -> tuple[int, ...]:
    if plan is None:
        return ()
    return tuple(sorted(int(item["chunk_id"]) for item in plan.expected_by_chunk))


def _validate_payload_path(
    output_dir: Path,
    *,
    relative_path: str,
    expected_file_sha256: str,
    expected_nbytes: int,
) -> None:
    path = output_dir / relative_path
    if not path.exists():
        raise RuntimeError(f"Missing payload file: {relative_path}")
    if file_sha256(path) != expected_file_sha256:
        raise RuntimeError(f"Payload file hash mismatch: {relative_path}")
    if int(path.stat().st_size) != int(expected_nbytes):
        raise RuntimeError(f"Payload byte-size mismatch: {relative_path}")


def _scattering_chunk_state(
    *,
    output_dir: Path,
    run_digest: str,
    chunk_id: int,
) -> ChunkState:
    commit = read_manifest(
        chunk_commit_path(output_dir, run_digest, SCATTERING_STAGE, chunk_id),
        codec=ScatteringChunkCommitManifest,
        output_dir=output_dir,
    )
    if commit.run_digest != str(run_digest):
        raise RuntimeError("Scattering chunk commit run identity mismatch.")
    candidate = read_manifest(
        output_dir / commit.candidate_manifest_path,
        codec=ScatteringCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_scattering_commit_candidate(candidate, output_dir=output_dir)
    if candidate.candidate_id != commit.selected_candidate_id:
        raise RuntimeError("Scattering chunk commit selected candidate mismatch.")
    if candidate.payload_path != commit.candidate_payload_path:
        raise RuntimeError("Scattering chunk commit payload path mismatch.")
    if candidate.payload_sha256 != commit.payload_sha256:
        raise RuntimeError("Scattering chunk commit payload hash mismatch.")
    _validate_payload_path(
        output_dir,
        relative_path=commit.candidate_payload_path,
        expected_file_sha256=commit.file_sha256,
        expected_nbytes=commit.payload_nbytes,
    )
    return ChunkState(
        stage=SCATTERING_STAGE,
        chunk_id=int(chunk_id),
        payload_sha256=commit.payload_sha256,
        file_sha256=commit.file_sha256,
        payload_nbytes=int(commit.payload_nbytes),
        candidate_manifest_path=commit.candidate_manifest_path,
        candidate_payload_path=commit.candidate_payload_path,
        interval_ids=tuple(int(item) for item in candidate.contributing_interval_ids),
    )


def _residual_chunk_state(
    *,
    output_dir: Path,
    run_digest: str,
    chunk_id: int,
) -> ChunkState:
    commit = read_manifest(
        chunk_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id),
        codec=ResidualChunkCommitManifest,
        output_dir=output_dir,
    )
    if commit.run_digest != str(run_digest):
        raise RuntimeError("Residual chunk commit run identity mismatch.")
    candidate = read_manifest(
        output_dir / commit.candidate_manifest_path,
        codec=ResidualCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_residual_commit_candidate(candidate, output_dir=output_dir)
    if candidate.candidate_id != commit.selected_candidate_id:
        raise RuntimeError("Residual chunk commit selected candidate mismatch.")
    if candidate.payload_path != commit.candidate_payload_path:
        raise RuntimeError("Residual chunk commit payload path mismatch.")
    if candidate.payload_sha256 != commit.payload_sha256:
        raise RuntimeError("Residual chunk commit payload hash mismatch.")
    _validate_payload_path(
        output_dir,
        relative_path=commit.candidate_payload_path,
        expected_file_sha256=commit.file_sha256,
        expected_nbytes=commit.payload_nbytes,
    )
    attempts_by_key = {
        (attempt.work_unit_digest, attempt.attempt_id): attempt
        for attempt in discover_residual_attempts(
            output_dir=output_dir,
            run_digest=run_digest,
            chunk_id=int(chunk_id),
        )
    }
    interval_ids: set[int] = set()
    for selected in candidate.selected_attempts:
        attempt = attempts_by_key.get(
            (str(selected["work_unit_digest"]), str(selected["attempt_id"]))
        )
        if attempt is not None:
            interval_ids.update(int(item) for item in attempt.interval_ids)
    return ChunkState(
        stage=RESIDUAL_FIELD_STAGE,
        chunk_id=int(chunk_id),
        payload_sha256=commit.payload_sha256,
        file_sha256=commit.file_sha256,
        payload_nbytes=int(commit.payload_nbytes),
        candidate_manifest_path=commit.candidate_manifest_path,
        candidate_payload_path=commit.candidate_payload_path,
        interval_ids=tuple(sorted(interval_ids)),
        partition_ids=tuple(int(item) for item in candidate.partition_ids),
    )


def _scan_scattering_stage(output_dir: Path, run_digest: str) -> StageState:
    invalid: list[str] = []
    plan = None
    try:
        path = stage_plan_path(output_dir, run_digest, SCATTERING_STAGE)
        if path.exists():
            plan = read_manifest(path, codec=ScatteringStagePlanManifest, output_dir=output_dir)
    except Exception as exc:
        invalid.append(f"scattering/stage_plan.json: {type(exc).__name__}: {exc}")
    expected_chunks = _expected_scattering_chunks(plan)

    valid_payloads: list[ChunkState] = []
    valid_ids: set[int] = set()
    for chunk_id in expected_chunks:
        path = chunk_commit_path(output_dir, run_digest, SCATTERING_STAGE, chunk_id)
        if not path.exists():
            continue
        try:
            state = _scattering_chunk_state(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_id=int(chunk_id),
            )
            valid_payloads.append(state)
            valid_ids.add(int(chunk_id))
        except Exception as exc:
            invalid.append(f"scattering/chunk_{chunk_id}/chunk_commit.json: {type(exc).__name__}: {exc}")

    stage_commit_present = stage_commit_path(output_dir, run_digest, SCATTERING_STAGE).exists()
    complete = False
    if stage_commit_present:
        try:
            commit = read_manifest(
                stage_commit_path(output_dir, run_digest, SCATTERING_STAGE),
                codec=ScatteringStageCommitManifest,
                output_dir=output_dir,
            )
            write_scattering_stage_commit(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_ids=commit.chunk_ids,
            )
            complete = not invalid and tuple(sorted(commit.chunk_ids)) == expected_chunks
        except Exception as exc:
            invalid.append(f"scattering/stage_commit.json: {type(exc).__name__}: {exc}")
            complete = False

    missing = tuple(chunk_id for chunk_id in expected_chunks if chunk_id not in valid_ids)
    return StageState(
        stage=SCATTERING_STAGE,
        stage_plan_present=plan is not None,
        stage_commit_present=stage_commit_present,
        complete=bool(complete and not missing),
        expected_chunk_ids=expected_chunks,
        valid_chunk_ids=tuple(sorted(valid_ids)),
        missing_chunk_ids=missing,
        selected_payloads=tuple(sorted(valid_payloads, key=lambda item: item.chunk_id)),
        invalid_manifests=tuple(invalid),
    )


def _scan_residual_stage(output_dir: Path, run_digest: str) -> StageState:
    invalid: list[str] = []
    no_output = False
    try:
        no_output_path = output_dir / ".mosaic" / "runs" / run_digest / RESIDUAL_FIELD_STAGE / "no_output.json"
        if no_output_path.exists():
            require_residual_no_output_manifest(output_dir=output_dir, run_digest=run_digest)
            no_output = True
    except Exception as exc:
        invalid.append(f"residual_field/no_output.json: {type(exc).__name__}: {exc}")

    plan = None
    try:
        path = stage_plan_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE)
        if path.exists():
            plan = read_manifest(path, codec=ResidualStagePlanManifest, output_dir=output_dir)
    except Exception as exc:
        invalid.append(f"residual_field/stage_plan.json: {type(exc).__name__}: {exc}")
    expected_chunks = _expected_residual_chunks(plan)

    valid_payloads: list[ChunkState] = []
    valid_ids: set[int] = set()
    for chunk_id in expected_chunks:
        path = chunk_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id)
        if not path.exists():
            continue
        try:
            state = _residual_chunk_state(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_id=int(chunk_id),
            )
            valid_payloads.append(state)
            valid_ids.add(int(chunk_id))
        except Exception as exc:
            invalid.append(f"residual_field/chunk_{chunk_id}/chunk_commit.json: {type(exc).__name__}: {exc}")

    stage_commit_present = stage_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE).exists()
    complete = False
    if stage_commit_present:
        try:
            commit = read_manifest(
                stage_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE),
                codec=ResidualStageCommitManifest,
                output_dir=output_dir,
            )
            write_residual_stage_commit(
                output_dir=output_dir,
                run_digest=run_digest,
                chunk_ids=commit.chunk_ids,
            )
            complete = not invalid and tuple(sorted(commit.chunk_ids)) == expected_chunks
        except Exception as exc:
            invalid.append(f"residual_field/stage_commit.json: {type(exc).__name__}: {exc}")
            complete = False

    missing = tuple(chunk_id for chunk_id in expected_chunks if chunk_id not in valid_ids)
    return StageState(
        stage=RESIDUAL_FIELD_STAGE,
        stage_plan_present=plan is not None,
        stage_commit_present=stage_commit_present,
        complete=bool(complete and not missing),
        expected_chunk_ids=expected_chunks,
        valid_chunk_ids=tuple(sorted(valid_ids)),
        missing_chunk_ids=missing,
        selected_payloads=tuple(sorted(valid_payloads, key=lambda item: item.chunk_id)),
        invalid_manifests=tuple(invalid),
        no_output=no_output,
    )


def scan_run_state(output_dir: str | Path, run_digest: str) -> RunStateSnapshot:
    output_root = Path(output_dir).resolve()
    invalid: list[str] = []
    run_valid = False
    qspace_valid = False
    try:
        path = run_manifest_path(output_root, run_digest)
        if path.exists():
            run_valid = _read_json_manifest(
                path,
                expected_schema="mosaic.run_manifest",
                output_dir=output_root,
            )
    except Exception as exc:
        invalid.append(f"run_manifest.json: {type(exc).__name__}: {exc}")
    try:
        path = qspace_plan_path(output_root, run_digest)
        if path.exists():
            qspace_valid = _read_json_manifest(
                path,
                expected_schema="mosaic.qspace_plan",
                output_dir=output_root,
            )
    except Exception as exc:
        invalid.append(f"qspace_plan.json: {type(exc).__name__}: {exc}")

    scattering = _scan_scattering_stage(output_root, run_digest)
    residual = _scan_residual_stage(output_root, run_digest)
    invalid.extend(scattering.invalid_manifests)
    invalid.extend(residual.invalid_manifests)

    decoder_valid = False
    try:
        decoder_path = output_root / ".mosaic" / "runs" / run_digest / "decoding" / "decoder_commit.json"
        if decoder_path.exists():
            read_manifest(decoder_path, codec=DecoderCommitManifest, output_dir=output_root)
            decoder_valid = True
    except Exception as exc:
        invalid.append(f"decoding/decoder_commit.json: {type(exc).__name__}: {exc}")

    return RunStateSnapshot(
        output_dir=str(output_root),
        run_digest=str(run_digest),
        run_manifest_valid=run_valid,
        qspace_plan_valid=qspace_valid,
        scattering=scattering,
        residual_field=residual,
        decoder_commit_valid=decoder_valid,
        invalid_manifests=tuple(invalid),
    )


def pending_scattering_interval_chunks(
    snapshot: RunStateSnapshot,
    all_interval_chunk_pairs: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    all_pairs = sorted({(int(interval_id), int(chunk_id)) for interval_id, chunk_id in all_interval_chunk_pairs})
    if snapshot.scattering.complete:
        return []
    if snapshot.scattering.stage_plan_present:
        complete_chunks = set(snapshot.scattering.valid_chunk_ids)
        return [
            (interval_id, chunk_id)
            for interval_id, chunk_id in all_pairs
            if int(chunk_id) not in complete_chunks
        ]
    return all_pairs


def pending_residual_interval_chunks(
    snapshot: RunStateSnapshot,
    expected_interval_chunks: Iterable[tuple[int, int]],
    *,
    output_dir: str | Path | None = None,
    residual_parameter_digest: str | None = None,
) -> list[tuple[int, int]]:
    all_pairs = sorted({(int(interval_id), int(chunk_id)) for interval_id, chunk_id in expected_interval_chunks})
    if snapshot.residual_field.complete or snapshot.residual_field.no_output:
        return []
    complete_chunks = set(snapshot.residual_field.valid_chunk_ids)
    # Streaming work units leave no payload manifests for the snapshot to
    # see; their durable completion lives in COMMITTED reducer progress
    # manifests (with final artifacts verified present). Without this a
    # COMPLETED case re-derives its entire residual stage on retry.
    credited: set[tuple[int, int]] = set()
    if output_dir is not None and residual_parameter_digest:
        for chunk_id, interval_ids in _committed_streaming_residual_credits(
            output_dir, str(residual_parameter_digest)
        ):
            credited.update(
                (int(interval_id), int(chunk_id)) for interval_id in interval_ids
            )
    return [
        (interval_id, chunk_id)
        for interval_id, chunk_id in all_pairs
        if int(chunk_id) not in complete_chunks
        and (interval_id, chunk_id) not in credited
    ]


def _committed_streaming_residual_credits(
    output_dir: str | Path,
    parameter_digest: str,
) -> list[tuple[int, tuple[int, ...]]]:
    """(chunk_id, incorporated interval ids) from COMMITTED reducer
    progress manifests of the CURRENT parameter digest whose final
    artifacts all exist on disk.

    Streaming residual work units return status-only results and write no
    payload manifests, so the payload overlay in the rebuild cannot
    re-credit them; without this source, the reset wipes a COMPLETED
    case's saved rows and a retry re-derives the entire residual stage
    (observed: ~53 min of GPU recompute on an already-published hkl40
    case). Digest scoping comes from the manifest FILENAME, so a stale
    config family can never credit rows for the current one; missing
    final artifacts leave rows unsaved (manifests stay the authority).

    Parsing goes through the typed manifest loader (one place owns the
    status/pending semantics), and final-artifact presence is re-derived
    from output_dir instead of the write-time-absolute paths embedded in
    the manifest — a moved or remounted output directory keeps its credits
    instead of silently re-deriving the whole stage."""
    from core.contracts import CompletionStatus
    from core.residual_field.artifacts import (
        build_residual_field_output_artifact_refs,
    )
    from core.residual_field.manifest_io import (
        load_residual_field_reducer_progress_manifest,
    )

    credits: list[tuple[int, tuple[int, ...]]] = []
    root = Path(output_dir) / "residual_checkpoints"
    if not root.is_dir():
        return credits
    pattern = f"chunk_*/reducer_progress_params_{parameter_digest}.manifest.json"
    for manifest_path in sorted(root.glob(pattern)):
        try:
            manifest = load_residual_field_reducer_progress_manifest(manifest_path)
        except Exception:
            continue  # unreadable or schema-invalid manifests never credit
        if manifest.completion_status is not CompletionStatus.COMMITTED:
            continue
        if manifest.pending_interval_ids or manifest.pending_shard_keys:
            continue
        expected_refs = build_residual_field_output_artifact_refs(
            str(output_dir), int(manifest.chunk_id)
        )
        if not expected_refs or not all(
            Path(ref.path).exists() for ref in expected_refs
        ):
            continue
        interval_ids = tuple(
            int(value) for value in manifest.incorporated_interval_ids
        )
        if interval_ids:
            credits.append((int(manifest.chunk_id), interval_ids))
    return credits


def rebuild_sqlite_cache_from_manifests(
    db_manager: DatabaseManagerProtocol,
    *,
    output_dir: str | Path,
    run_digest: str,
    residual_parameter_digest: str | None = None,
) -> RunStateSnapshot:
    scan_started = time.perf_counter()
    snapshot = scan_run_state(output_dir, run_digest)
    write_performance_metrics(
        output_dir=output_dir,
        run_digest=run_digest,
        recovery_scan_seconds=time.perf_counter() - scan_started,
    )
    if not db_manager.cache_enabled:
        return snapshot
    # Last-write-wins overlay reproducing the sequential loops' final
    # state, applied in ONE transaction (the sequential form was up to
    # ~18k single-row fsync commits).
    final_state: dict[tuple[int, int], bool] = {
        (int(interval_id), int(chunk_id)): False
        for interval_id, chunk_id in db_manager.get_interval_chunks()
    }
    for stage in (snapshot.scattering, snapshot.residual_field):
        for payload in stage.selected_payloads:
            for interval_id in payload.interval_ids:
                final_state[(int(interval_id), int(payload.chunk_id))] = True
    if residual_parameter_digest:
        for chunk_id, interval_ids in _committed_streaming_residual_credits(
            output_dir, str(residual_parameter_digest)
        ):
            for interval_id in interval_ids:
                final_state[(int(interval_id), int(chunk_id))] = True
    db_manager.update_interval_chunk_status_batch(
        [
            (interval_id, chunk_id, int(saved))
            for (interval_id, chunk_id), saved in sorted(final_state.items())
        ]
    )
    return snapshot


__all__ = [
    "ChunkState",
    "RunStateSnapshot",
    "StageState",
    "pending_residual_interval_chunks",
    "pending_scattering_interval_chunks",
    "rebuild_sqlite_cache_from_manifests",
    "scan_run_state",
]
