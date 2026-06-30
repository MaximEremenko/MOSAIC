"""
core/residual_field/reducer_helpers.py

Pure, stateless helper functions for the residual-field reducer.

These have no shared mutable state and no pickle coupling, and are
separated from backend.py purely for size/navigability.  The public API
of core.residual_field.backend is unchanged — every name here is
re-exported from there.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from core.residual_field.artifacts import (
    _missing_artifact_kinds,
    _missing_artifact_paths,
    _ResidualFieldChunkStatusUpdater,
)
from core.residual_field.contracts import ResidualFieldArtifactManifest
from core.runtime import is_sync_client


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _allocate_finalize_output(
    *,
    shape: tuple[int, ...],
    dtype,
    scratch_dir: Path | None,
    name: str,
) -> np.ndarray:
    """Allocate a C-contiguous output for finalize concatenation.

    When ``scratch_dir`` is provided, a disk-backed memmap is used so peak
    worker RAM stays at roughly one input block. Otherwise an in-memory
    ``np.empty`` is returned. Both paths produce bytes that are, once
    populated, bit-identical to ``np.concatenate`` / ``np.vstack`` of the
    same blocks in the same order.
    """
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(
            str(scratch_dir / f"{name}.npy"),
            mode="w+",
            dtype=np.dtype(dtype),
            shape=shape,
        )
    return np.empty(shape, dtype=np.dtype(dtype))


def _finalize_scratch_dir(
    scratch_root: str | None,
    *,
    chunk_id: int,
    parameter_digest: str,
) -> Path | None:
    if scratch_root is None:
        return None
    return Path(scratch_root) / "_residual_finalize" / (
        f"chunk_{int(chunk_id)}_params_{str(parameter_digest)}"
    )


def _manifest_final_artifacts_present(manifest: ResidualFieldArtifactManifest) -> bool:
    return (
        not _missing_artifact_kinds(manifest)
        and not _missing_artifact_paths(manifest.artifacts)
    )


def _mark_residual_intervals_saved(
    *,
    db_path: str,
    chunk_id: int,
    interval_ids: tuple[int, ...] | list[int] | set[int],
) -> None:
    status_updater = _ResidualFieldChunkStatusUpdater(db_path)
    for interval_id in sorted(int(value) for value in interval_ids):
        status_updater.mark_saved(int(interval_id), int(chunk_id))


def is_same_node_local_client(client) -> bool:
    if client is None or is_sync_client(client):
        return True
    backend = str(os.getenv("DASK_BACKEND", "")).strip().lower()
    if backend in {"local", "cuda-local", "sync", "synchronous", "single-threaded"}:
        return True
    cluster = getattr(client, "cluster", None)
    cluster_name = type(cluster).__name__.lower() if cluster is not None else ""
    return "localcluster" in cluster_name


def _normalize_reducer_backend_kind(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"local", "local_restartable", "local_restartable_reducer"}:
        return "local_restartable"
    if normalized in {
        "durable",
        "durable_shared",
        "durable_shared_restartable",
        "durable_restartable",
    }:
        return "durable_shared_restartable"
    raise ValueError(
        "Residual-field reducer backend must be 'local_restartable' or "
        "'durable_shared_restartable'."
    )


def checkpoint_cadence(
    total_expected_partials: int,
    *,
    uses_shared_durable_generations: bool,
) -> int:
    """Durable-snapshot cadence (every N accepted partials). Pure POLICY, not state.

    For shared-durable generations an env override is honored; otherwise a quarter of
    the expected partials, floored at 1. Lifted verbatim from the reducer backend so
    the cadence policy lives beside the other stateless reducer helpers.
    """
    if uses_shared_durable_generations:
        override = os.getenv("MOSAIC_DISTRIBUTED_CHECKPOINT_CADENCE")
        if override is not None and str(override).strip():
            return max(int(override), 1)
    return max(int(total_expected_partials) // 4, 1)


def live_trim_cadence(checkpoint_cadence_value: int) -> int:
    """Live in-RAM trim cadence (env-overridable), floored at 8. Pure POLICY."""
    override = os.getenv("MOSAIC_LOCAL_ACCUMULATOR_TRIM_CADENCE_BATCHES")
    if override is not None and str(override).strip():
        return max(int(override), 1)
    return max(int(checkpoint_cadence_value) // 2, 8)


__all__ = [
    "_allocate_finalize_output",
    "checkpoint_cadence",
    "live_trim_cadence",
    "_finalize_scratch_dir",
    "_manifest_final_artifacts_present",
    "_mark_residual_intervals_saved",
    "_normalize_reducer_backend_kind",
    "DEFAULT_LOCAL_ACCUMULATOR_MAX_RAM_BYTES",
    "is_same_node_local_client",
]
