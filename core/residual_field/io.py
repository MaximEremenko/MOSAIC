from __future__ import annotations

import numpy as np


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


def normalize_residual_values_ntotal(values, *, rifft_saver, chunk_id, logger=None):
    filename = rifft_saver.generate_filename(
        chunk_id,
        suffix="_amplitudes_ntotal_reciprocal_space_points",
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


__all__ = ["load_scalar_from_store", "normalize_residual_values_ntotal"]
