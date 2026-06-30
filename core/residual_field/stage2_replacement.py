"""Stage-2 replacement expected-metadata cluster.

Functions for normalising, digesting, building, writing, and loading
the stage-2 replacement expected-coverage manifest.  This module has no
dependency on ``core.residual_field.artifacts`` — it imports only from
``core.residual_field.manifest_io``, ``core.residual_field.contracts``, and
``core.contracts``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from core.residual_field.contracts import (
    RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
)
from core.residual_field.manifest_io import _write_json_atomic
from core.contracts import ArtifactRef

__all__ = [
    "build_stage2_replacement_expected_artifact",
    "load_stage2_replacement_expected_manifest",
    "load_stage2_replacement_expected_metadata",
    "normalize_stage2_replacement_expected_by_chunk",
    "normalize_stage2_replacement_expected_metadata",
    "stage2_replacement_expected_digest",
    "write_stage2_replacement_expected_manifest",
]


def normalize_stage2_replacement_expected_by_chunk(
    raw: Mapping[object, Iterable[object]] | None,
) -> dict[int, tuple[int, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("Stage-2 replacement expected coverage must be a mapping.")
    expected: dict[int, tuple[int, ...]] = {}
    for chunk_id, interval_ids in raw.items():
        try:
            normalized_ids = tuple(
                sorted({int(interval_id) for interval_id in interval_ids})
            )
        except TypeError as exc:
            raise ValueError(
                "Stage-2 replacement expected interval ids must be iterable."
            ) from exc
        expected[int(chunk_id)] = normalized_ids
    return dict(sorted(expected.items()))


def stage2_replacement_expected_digest(
    expected_by_chunk: Mapping[object, Iterable[object]] | None,
) -> str:
    normalized = normalize_stage2_replacement_expected_by_chunk(expected_by_chunk)
    payload = {
        str(chunk_id): list(interval_ids)
        for chunk_id, interval_ids in normalized.items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_stage2_replacement_expected_metadata(
    raw: Mapping[str, object],
) -> dict[str, object]:
    expected = normalize_stage2_replacement_expected_by_chunk(
        raw.get("expected_by_chunk", {})
    )
    expected_digest = raw.get("expected_digest")
    if (
        expected_digest is not None
        and str(expected_digest) != stage2_replacement_expected_digest(expected)
    ):
        raise ValueError("Stage-2 replacement expected manifest digest mismatch.")
    return {
        "expected_by_chunk": expected,
        "run_digest": None if raw.get("run_digest") is None else str(raw["run_digest"]),
        "source_scattering_commit_digest": (
            None
            if raw.get("source_scattering_commit_digest") is None
            else str(raw["source_scattering_commit_digest"])
        ),
    }


def build_stage2_replacement_expected_artifact(
    output_dir: str,
    *,
    parameter_digest: str,
) -> ArtifactRef:
    path = (
        Path(output_dir)
        / "residual_checkpoints"
        / f"stage2_replacement_expected_params_{parameter_digest}.manifest.json"
    )
    return ArtifactRef(
        stage="residual_field",
        kind="stage2-replacement-expected-manifest",
        key=f"stage2-replacement-expected:params-{parameter_digest}",
        path=str(path),
        schema_version=RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
    )


def write_stage2_replacement_expected_manifest(
    *,
    output_dir: str,
    parameter_digest: str,
    expected_by_chunk: Mapping[object, Iterable[object]] | None,
    run_digest: str | None = None,
    source_scattering_commit_digest: str | None = None,
) -> ArtifactRef:
    artifact = build_stage2_replacement_expected_artifact(
        output_dir,
        parameter_digest=parameter_digest,
    )
    if artifact.path is None:
        raise ValueError("Stage-2 replacement expected manifest path is required.")
    normalized = normalize_stage2_replacement_expected_by_chunk(expected_by_chunk)
    payload = {
        "schema_version": RESIDUAL_FIELD_CONTRACT_SCHEMA_VERSION,
        "stage": artifact.stage,
        "kind": artifact.kind,
        "artifact": {
            "stage": artifact.stage,
            "kind": artifact.kind,
            "key": artifact.key,
            "path": artifact.path,
            "schema_version": artifact.schema_version,
        },
        "parameter_digest": str(parameter_digest),
        "run_digest": None if run_digest is None else str(run_digest),
        "source_scattering_commit_digest": (
            None
            if source_scattering_commit_digest is None
            else str(source_scattering_commit_digest)
        ),
        "expected_digest": stage2_replacement_expected_digest(normalized),
        "expected_by_chunk": {
            str(chunk_id): list(interval_ids)
            for chunk_id, interval_ids in normalized.items()
        },
    }
    _write_json_atomic(Path(artifact.path), payload)
    return artifact


def load_stage2_replacement_expected_metadata(
    *,
    output_dir: str,
    parameter_digest: str,
) -> dict[str, object] | None:
    artifact = build_stage2_replacement_expected_artifact(
        output_dir,
        parameter_digest=parameter_digest,
    )
    if artifact.path is None:
        raise ValueError("Stage-2 replacement expected manifest path is required.")
    path = Path(artifact.path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("kind")) != artifact.kind:
        raise ValueError("Stage-2 replacement expected manifest has invalid kind.")
    if str(payload.get("parameter_digest")) != str(parameter_digest):
        raise ValueError(
            "Stage-2 replacement expected manifest parameter digest mismatch."
        )
    return normalize_stage2_replacement_expected_metadata(payload)


def load_stage2_replacement_expected_manifest(
    *,
    output_dir: str,
    parameter_digest: str,
) -> dict[int, tuple[int, ...]] | None:
    metadata = load_stage2_replacement_expected_metadata(
        output_dir=output_dir,
        parameter_digest=parameter_digest,
    )
    if metadata is None:
        return None
    return dict(metadata["expected_by_chunk"])
