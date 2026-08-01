"""Residual-field shard discovery helpers.

Functions for scanning the filesystem for shard manifests, merging manifest
collections, and querying the reducer-progress manifest.

This module imports from ``core.residual_field.manifest_io`` and
``core.residual_field.contracts`` only — it has no dependency on
``core.residual_field.artifacts``.

Note: ``is_residual_field_shard_reclaimable`` and
``delete_reclaimable_residual_field_shards`` are intentionally left in
``artifacts.py`` because they call ``build_residual_field_chunk_manifest``
and ``assess_residual_field_manifest``, which in turn depend on
``ResidualFieldArtifactStore`` and ``_ResidualFieldChunkStatusUpdater`` —
both of which must remain in ``artifacts.py``.  Moving those two functions
here would create a circular import.
"""

from __future__ import annotations

from pathlib import Path

from core.residual_field.contracts import (
    ResidualFieldReducerProgressManifest,
    ResidualFieldShardManifest,
)
from core.residual_field.manifest_io import (
    build_residual_field_reducer_progress_artifact,
    load_residual_field_reducer_progress_manifest,
    load_residual_field_shard_manifest,
)

__all__ = [
    "_merge_residual_field_shard_manifests",
    "_shard_manifests_by_key",
    "discover_residual_field_reducer_progress_manifest",
    "discover_residual_field_shard_manifests",
    "list_reclaimable_residual_field_shards",
]


def discover_residual_field_shard_manifests(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    shard_storage_root: str | None = None,
) -> list[ResidualFieldShardManifest]:
    shard_dir = Path(shard_storage_root or output_dir) / "residual_checkpoints" / f"chunk_{chunk_id}"
    if not shard_dir.exists():
        return []
    manifests: list[ResidualFieldShardManifest] = []
    for path in sorted(shard_dir.glob(f"batch_*_params_{parameter_digest}.manifest.json")):
        manifests.append(load_residual_field_shard_manifest(path))
    return manifests


def _merge_residual_field_shard_manifests(
    *manifest_groups: list[ResidualFieldShardManifest] | tuple[ResidualFieldShardManifest, ...] | None,
) -> list[ResidualFieldShardManifest]:
    merged: dict[str, ResidualFieldShardManifest] = {}
    for manifest_group in manifest_groups:
        if not manifest_group:
            continue
        for manifest in manifest_group:
            merged[manifest.artifact_key] = manifest
    return [merged[key] for key in sorted(merged)]


def _shard_manifests_by_key(
    manifests: list[ResidualFieldShardManifest],
) -> dict[str, ResidualFieldShardManifest]:
    return {manifest.artifact_key: manifest for manifest in manifests}


def discover_residual_field_reducer_progress_manifest(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
) -> ResidualFieldReducerProgressManifest | None:
    artifact = build_residual_field_reducer_progress_artifact(
        output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if artifact.path is None or not Path(artifact.path).exists():
        return None
    return load_residual_field_reducer_progress_manifest(artifact.path)


def list_reclaimable_residual_field_shards(
    *,
    output_dir: str,
    chunk_id: int,
    parameter_digest: str,
    shard_storage_root: str | None = None,
) -> list[ResidualFieldShardManifest]:
    progress = discover_residual_field_reducer_progress_manifest(
        output_dir=output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
    )
    if progress is None:
        return []
    reclaimable = set(progress.reclaimable_shard_keys)
    if not reclaimable:
        return []
    return [
        manifest
        for manifest in discover_residual_field_shard_manifests(
            output_dir=output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            shard_storage_root=shard_storage_root,
        )
        if manifest.artifact_key in reclaimable
    ]
