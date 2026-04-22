from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from core.residual_field.contracts import (
    build_legacy_residual_field_output_artifacts,
    build_residual_field_output_artifacts,
)


def load_scalar_from_store(rifft_saver, filename, key_candidates):
    try:
        data = rifft_saver.load_data(filename)
    except FileNotFoundError:
        return None
    for key in key_candidates:
        if key in data:
            value = np.array(data[key])
            try:
                return int(np.ravel(value)[0])
            except Exception:
                pass
    return None


def resolve_residual_chunk_artifact_filename(output_dir: str, chunk_id: int, kind: str) -> str:
    primary_refs = {
        artifact.kind: artifact
        for artifact in build_residual_field_output_artifacts(output_dir, chunk_id)
    }
    primary = Path(primary_refs[kind].path).name
    if (Path(output_dir) / primary).exists():
        return primary

    legacy_refs = {
        artifact.kind: artifact
        for artifact in build_legacy_residual_field_output_artifacts(output_dir, chunk_id)
    }
    legacy = Path(legacy_refs[kind].path).name
    if (Path(output_dir) / legacy).exists():
        return legacy
    return primary


def _resolve_store_output_dir(rifft_saver, chunk_id: int) -> str:
    out_by_saver = getattr(rifft_saver, "output_dir", None)
    if out_by_saver is not None:
        return str(out_by_saver)
    return os.path.dirname(
        os.path.abspath(rifft_saver.generate_filename(chunk_id, suffix="_amplitudes"))
    )


def normalize_residual_values_ntotal(values, *, rifft_saver, chunk_id, logger=None):
    output_dir = _resolve_store_output_dir(rifft_saver, chunk_id)
    filename = resolve_residual_chunk_artifact_filename(
        output_dir,
        chunk_id,
        "chunk-total-reciprocal-point-count",
    )
    ntot = load_scalar_from_store(
        rifft_saver,
        filename,
        ["ntotal_reciprocal_space_points", "ntotal_reciprocal_points"],
    )
    if not ntot or ntot <= 0:
        if logger:
            logger.warning(
                "[normalize_residual_values_ntotal] Missing ntotal; leaving values unscaled."
            )
        return values

    scale = 1.0 / float(ntot)
    try:
        if values.ndim == 2 and values.shape[1] >= 2:
            values[:, 1] *= scale
        else:
            values[...] *= scale
    except Exception as exc:
        if logger:
            logger.warning("[normalize_residual_values_ntotal] scaling failed: %s", exc)
    return values


__all__ = [
    "load_scalar_from_store",
    "normalize_residual_values_ntotal",
    "resolve_residual_chunk_artifact_filename",
]
