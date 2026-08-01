"""Residual-field manifest I/O primitives.

Low-level serialisation helpers, atomic-write utilities, and manifest
(de)serialisation functions.  Nothing in this module imports from
``core.residual_field.artifacts`` — it depends only on external libraries,
``core.contracts``, and ``core.residual_field.contracts``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from core.storage.atomic import atomic_write_json

from core.residual_field.contracts import (
    RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA,
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
    make_residual_field_artifact_key,
    make_residual_field_reducer_key,
    validate_residual_field_reducer_progress_manifest,
    validate_residual_field_shard_manifest,
)
from core.contracts import (
    ArtifactRef,
    CompletionStatus,
    RetryDisposition,
    RetryIdempotencySemantics,
)

__all__ = [
    "_GENERATION_FILENAME_RE",
    "_artifact_ref_from_payload",
    "_build_residual_field_reducer_progress_manifest",
    "_load_array_payload",
    "_normalize_residual_shard_cleanup_policy",
    "_residual_field_reducer_progress_manifest_to_payload",
    "_residual_field_shard_manifest_to_payload",
    "_write_json_atomic",
    "_write_residual_field_shard_manifest_json",
    "build_residual_field_reducer_progress_artifact",
    "load_residual_field_generation_metadata",
    "load_residual_field_reducer_progress_manifest",
    "load_residual_field_shard_manifest",
    "parse_residual_field_generation_ref",
    "write_residual_field_reducer_progress_manifest",
]

_GENERATION_FILENAME_RE = re.compile(
    r"^generation_partition_(?P<partition_token>[^_]+)_seq_(?P<generation_seq>\d+)_params_(?P<parameter_digest>.+)$"
)


# ---------------------------------------------------------------------------
# Atomic write primitives
# ---------------------------------------------------------------------------

def _write_json_atomic(target_path: Path, payload: dict[str, object]) -> None:
    # indent=2 keeps the residual-field manifest byte format stable:
    # existing manifests are re-read and re-written mid-run.
    atomic_write_json(target_path, payload, indent=2)


def _load_array_payload(path: str | Path) -> dict[str, np.ndarray]:
    payload_path = Path(path)
    if payload_path.suffix in {".h5", ".hdf5"}:
        with h5py.File(payload_path, "r") as h5file:
            return {name: np.asarray(h5file[name]) for name in h5file.keys()}
    with np.load(payload_path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


# ---------------------------------------------------------------------------
# Payload serialisation helpers
# ---------------------------------------------------------------------------

def _artifact_ref_from_payload(payload: dict[str, object]) -> ArtifactRef:
    return ArtifactRef(
        stage=str(payload["stage"]),
        kind=str(payload["kind"]),
        key=str(payload["key"]),
        path=str(payload["path"]) if payload.get("path") is not None else None,
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )


def _residual_field_shard_manifest_to_payload(
    manifest: ResidualFieldShardManifest,
) -> dict[str, object]:
    return {
        "artifact_key": manifest.artifact_key,
        "completion_status": manifest.completion_status.value,
        "retry": {
            "failure_unit": manifest.retry.failure_unit,
            "retry_unit": manifest.retry.retry_unit,
            "idempotency_key": manifest.retry.idempotency_key,
            "replay_disposition": manifest.retry.replay_disposition.value,
            "crash_recovery_rule": manifest.retry.crash_recovery_rule,
        },
        "interval_id": manifest.interval_id,
        "contributing_interval_ids": list(manifest.contributing_interval_ids),
        "chunk_id": manifest.chunk_id,
        "parameter_digest": manifest.parameter_digest,
        "point_count": manifest.point_count,
        "contribution_reciprocal_point_count": manifest.contribution_reciprocal_point_count,
        "total_reciprocal_point_count": manifest.total_reciprocal_point_count,
        "scratch_root": manifest.scratch_root,
        "producer_stage": manifest.producer_stage,
        "consumer_stage": manifest.consumer_stage,
        "artifact_schema_name": manifest.artifact_schema_name,
        "schema_version": manifest.schema_version,
        "artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.artifacts
        ],
        "upstream_artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.upstream_artifacts
        ],
    }


def _residual_field_reducer_progress_manifest_to_payload(
    manifest: ResidualFieldReducerProgressManifest,
) -> dict[str, object]:
    return {
        "artifact": {
            "stage": manifest.artifact.stage,
            "kind": manifest.artifact.kind,
            "key": manifest.artifact.key,
            "path": manifest.artifact.path,
            "schema_version": manifest.artifact.schema_version,
        },
        "reducer_key": manifest.reducer_key,
        "chunk_id": manifest.chunk_id,
        "parameter_digest": manifest.parameter_digest,
        "completion_status": manifest.completion_status.value,
        "durable_truth_unit": manifest.durable_truth_unit,
        "incorporated_shard_keys": list(manifest.incorporated_shard_keys),
        "incorporated_interval_ids": list(manifest.incorporated_interval_ids),
        "pending_shard_keys": list(manifest.pending_shard_keys),
        "pending_interval_ids": list(manifest.pending_interval_ids),
        "reclaimable_shard_keys": list(manifest.reclaimable_shard_keys),
        "cleanup_policy": manifest.cleanup_policy,
        "final_artifacts": [
            {
                "stage": artifact.stage,
                "kind": artifact.kind,
                "key": artifact.key,
                "path": artifact.path,
                "schema_version": artifact.schema_version,
            }
            for artifact in manifest.final_artifacts
        ],
        "schema_version": manifest.schema_version,
    }


# ---------------------------------------------------------------------------
# Generation-ref parsing
# ---------------------------------------------------------------------------

def parse_residual_field_generation_ref(
    manifest: ResidualFieldShardManifest,
) -> tuple[int | None, int] | None:
    manifest_ref = next(
        (artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-manifest"),
        None,
    )
    if manifest_ref is None or manifest_ref.path is None:
        return None
    match = _GENERATION_FILENAME_RE.match(Path(manifest_ref.path).stem.removesuffix(".manifest"))
    if match is None:
        return None
    partition_token = match.group("partition_token")
    partition_id = None if partition_token == "owner" else int(partition_token)
    return partition_id, int(match.group("generation_seq"))


# ---------------------------------------------------------------------------
# Manifest load / write
# ---------------------------------------------------------------------------

def load_residual_field_shard_manifest(manifest_path: str | Path) -> ResidualFieldShardManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    retry_payload = payload["retry"]
    manifest = ResidualFieldShardManifest(
        artifact_key=str(payload["artifact_key"]),
        artifacts=tuple(_artifact_ref_from_payload(item) for item in payload["artifacts"]),
        completion_status=CompletionStatus(str(payload["completion_status"])),
        retry=RetryIdempotencySemantics(
            failure_unit=str(retry_payload["failure_unit"]),
            retry_unit=str(retry_payload["retry_unit"]),
            idempotency_key=str(retry_payload["idempotency_key"]),
            replay_disposition=RetryDisposition(str(retry_payload["replay_disposition"])),
            crash_recovery_rule=str(retry_payload["crash_recovery_rule"]),
        ),
        interval_id=int(payload["interval_id"]),
        contributing_interval_ids=tuple(
            int(interval_id)
            for interval_id in payload.get(
                "contributing_interval_ids",
                [payload["interval_id"]],
            )
        ),
        chunk_id=int(payload["chunk_id"]),
        parameter_digest=str(payload["parameter_digest"]),
        point_count=int(payload["point_count"]),
        contribution_reciprocal_point_count=int(payload["contribution_reciprocal_point_count"]),
        total_reciprocal_point_count=int(payload["total_reciprocal_point_count"]),
        scratch_root=(
            str(payload["scratch_root"])
            if payload.get("scratch_root") is not None
            else None
        ),
        producer_stage=str(payload.get("producer_stage", "residual_field")),
        consumer_stage=payload.get("consumer_stage"),
        upstream_artifacts=tuple(
            _artifact_ref_from_payload(item) for item in payload.get("upstream_artifacts", [])
        ),
        artifact_schema_name=str(payload.get("artifact_schema_name", RESIDUAL_FIELD_SHARD_ARTIFACT_SCHEMA.name)),
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )
    validate_residual_field_shard_manifest(manifest)
    return manifest


def _write_residual_field_shard_manifest_json(
    manifest: ResidualFieldShardManifest,
    *,
    extra_payload: dict[str, object] | None = None,
) -> None:
    manifest_ref = next(
        artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-manifest"
    )
    if manifest_ref.path is None:
        raise ValueError("Residual-field shard manifest path is required.")
    payload = _residual_field_shard_manifest_to_payload(manifest)
    if extra_payload:
        payload.update(extra_payload)
    _write_json_atomic(
        Path(manifest_ref.path),
        payload,
    )


def load_residual_field_generation_metadata(
    manifest: ResidualFieldShardManifest,
) -> dict[str, object]:
    manifest_ref = next(
        artifact for artifact in manifest.artifacts if artifact.kind == "residual-shard-manifest"
    )
    if manifest_ref.path is None or not Path(manifest_ref.path).exists():
        return {}
    payload = json.loads(Path(manifest_ref.path).read_text(encoding="utf-8"))
    generation_ref = parse_residual_field_generation_ref(manifest)
    partition_id = None
    generation_seq = None
    if generation_ref is not None:
        partition_id, generation_seq = generation_ref
    return {
        "partition_id": (
            int(payload["partition_id"])
            if payload.get("partition_id") is not None
            else partition_id
        ),
        "generation_seq": (
            int(payload["generation_seq"])
            if payload.get("generation_seq") is not None
            else generation_seq
        ),
        "checkpoint_bytes_written": int(payload.get("checkpoint_bytes_written", 0)),
        "checkpoint_wall_seconds": float(payload.get("checkpoint_wall_seconds", 0.0)),
        "compression": str(payload.get("compression", "")),
    }


def load_residual_field_reducer_progress_manifest(
    manifest_path: str | Path,
) -> ResidualFieldReducerProgressManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = ResidualFieldReducerProgressManifest(
        artifact=_artifact_ref_from_payload(payload["artifact"]),
        reducer_key=str(
            payload.get(
                "reducer_key",
                make_residual_field_reducer_key(
                    chunk_id=int(payload["chunk_id"]),
                    parameter_digest=str(payload["parameter_digest"]),
                ),
            )
        ),
        chunk_id=int(payload["chunk_id"]),
        parameter_digest=str(payload["parameter_digest"]),
        completion_status=CompletionStatus(str(payload["completion_status"])),
        durable_truth_unit=str(
            payload.get("durable_truth_unit", "committed_shard_checkpoint")
        ),
        incorporated_shard_keys=tuple(str(key) for key in payload["incorporated_shard_keys"]),
        incorporated_interval_ids=tuple(int(interval_id) for interval_id in payload["incorporated_interval_ids"]),
        pending_shard_keys=tuple(str(key) for key in payload.get("pending_shard_keys", [])),
        pending_interval_ids=tuple(int(interval_id) for interval_id in payload.get("pending_interval_ids", [])),
        reclaimable_shard_keys=tuple(str(key) for key in payload.get("reclaimable_shard_keys", [])),
        cleanup_policy=str(payload.get("cleanup_policy", "off")),
        final_artifacts=tuple(
            _artifact_ref_from_payload(item) for item in payload.get("final_artifacts", [])
        ),
        schema_version=int(payload.get("schema_version", RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION)),
    )
    validate_residual_field_reducer_progress_manifest(manifest)
    return manifest


def write_residual_field_reducer_progress_manifest(
    manifest: ResidualFieldReducerProgressManifest,
) -> ResidualFieldReducerProgressManifest:
    validate_residual_field_reducer_progress_manifest(manifest)
    if manifest.artifact.path is None:
        raise ValueError("Reducer progress manifest path is required.")
    _write_json_atomic(
        Path(manifest.artifact.path),
        _residual_field_reducer_progress_manifest_to_payload(manifest),
    )
    return manifest


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------

def build_residual_field_reducer_progress_artifact(
    output_dir: str,
    *,
    chunk_id: int,
    parameter_digest: str,
) -> ArtifactRef:
    shard_dir = Path(output_dir) / "residual_checkpoints" / f"chunk_{chunk_id}"
    return ArtifactRef(
        stage="residual_field",
        kind="residual-reducer-progress-manifest",
        key=make_residual_field_artifact_key(
            "residual-reducer-progress-manifest",
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        path=str(shard_dir / f"reducer_progress_params_{parameter_digest}.manifest.json"),
        schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    )


# ---------------------------------------------------------------------------
# Cleanup policy normalisation
# ---------------------------------------------------------------------------

def _normalize_residual_shard_cleanup_policy(policy: str | bool | None) -> str:
    if isinstance(policy, bool):
        return "delete_reclaimable" if policy else "off"
    normalized = str(policy or "off").strip().lower()
    if normalized in {"off", "false", "0", "keep"}:
        return "off"
    if normalized in {"delete_reclaimable", "cleanup", "on", "true", "1"}:
        return "delete_reclaimable"
    raise ValueError(
        "Residual-field cleanup policy must be 'off' or 'delete_reclaimable'."
    )


# ---------------------------------------------------------------------------
# Progress-manifest builder
# ---------------------------------------------------------------------------

def _build_residual_field_reducer_progress_manifest(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    completion_status: CompletionStatus,
    durable_truth_unit: str = "committed_shard_checkpoint",
    incorporated_shard_keys: tuple[str, ...],
    incorporated_interval_ids: tuple[int, ...],
    reclaimable_shard_keys: tuple[str, ...],
    final_artifacts: tuple[ArtifactRef, ...],
    pending_shard_keys: tuple[str, ...] = (),
    pending_interval_ids: tuple[int, ...] = (),
    cleanup_policy: str = "off",
) -> ResidualFieldReducerProgressManifest:
    return ResidualFieldReducerProgressManifest(
        artifact=build_residual_field_reducer_progress_artifact(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        reducer_key=make_residual_field_reducer_key(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
        ),
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        completion_status=completion_status,
        durable_truth_unit=str(durable_truth_unit),
        incorporated_shard_keys=tuple(sorted(set(str(key) for key in incorporated_shard_keys))),
        incorporated_interval_ids=tuple(
            sorted(set(int(interval_id) for interval_id in incorporated_interval_ids))
        ),
        reclaimable_shard_keys=tuple(sorted(set(str(key) for key in reclaimable_shard_keys))),
        final_artifacts=final_artifacts,
        pending_shard_keys=tuple(sorted(set(str(key) for key in pending_shard_keys))),
        pending_interval_ids=tuple(
            sorted(set(int(interval_id) for interval_id in pending_interval_ids))
        ),
        cleanup_policy=_normalize_residual_shard_cleanup_policy(cleanup_policy),
    )
