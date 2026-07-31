from __future__ import annotations

import logging
import os

import numpy as np

from core.residual_field.io import (
    normalize_residual_values_ntotal,
    resolve_residual_chunk_artifact_filename,
)


def resolve_output_dir(rifft_saver, chunk_id, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    out_by_saver = getattr(rifft_saver, "output_dir", None)
    output_dir = out_by_saver or os.path.dirname(
        os.path.abspath(rifft_saver.generate_filename(chunk_id, suffix="_amplitudes"))
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_rifft_space_grid(processor, chunk_id, point_data_list):
    """Regenerate the rifft-space grid (zero I/O)."""
    grids = []
    grids_shapeNd = []
    id_values = []
    counts = []
    for point_data in point_data_list:
        grid_points, grid_shapeNd = processor.point_data_processor.generate_grid(
            chunk_id=chunk_id,
            dimensionality=len(point_data["coordinates"]),
            step_in_frac=point_data["step_in_frac"],
            central_point=point_data["coordinates"],
            dist=point_data["dist_from_atom_center"],
            central_point_id=point_data["central_point_id"],
        )
        grids.append(grid_points)
        grids_shapeNd.append(grid_shapeNd)
        id_values.append(point_data["central_point_id"])
        counts.append(len(grid_points))
    if not grids:
        return np.array([]), grids_shapeNd
    # np.repeat over per-site counts, not a 211.7M-entry Python list.
    central_point_ids = np.repeat(
        np.asarray(id_values, dtype=np.int64),
        np.asarray(counts, dtype=np.int64),
    )
    rifft_space_grid = np.hstack((np.vstack(grids), central_point_ids[:, None]))
    return rifft_space_grid, grids_shapeNd


def load_residual_field_and_generate_grid(processor, chunk_id, point_data_list, rifft_saver):
    output_dir = resolve_output_dir(rifft_saver, chunk_id)
    filename = resolve_residual_chunk_artifact_filename(
        output_dir,
        chunk_id,
        "chunk-residual-values",
    )
    try:
        data = rifft_saver.load_data(filename)
        amplitudes = data.get("amplitudes", None)
        if amplitudes is None:
            logging.getLogger(__name__).warning("Amplitudes not found in %s", filename)
            return np.array([]), None, None

        amplitudes = normalize_residual_values_ntotal(
            amplitudes,
            rifft_saver=rifft_saver,
            chunk_id=chunk_id,
            logger=logging.getLogger(__name__),
        )

        rifft_space_grid, grids_shapeNd = _build_rifft_space_grid(
            processor, chunk_id, point_data_list
        )
        return rifft_space_grid, amplitudes, grids_shapeNd
    except FileNotFoundError:
        logging.getLogger(__name__).warning("File not found: %s", filename)
        return np.array([]), None, None


def load_chunk_residual_field_and_grid(processor, *, chunk_id, point_data_list, rifft_saver, logger):
    output_dir = resolve_output_dir(rifft_saver, chunk_id)
    fn_amp = resolve_residual_chunk_artifact_filename(
        output_dir,
        chunk_id,
        "chunk-residual-values",
    )
    try:
        data = rifft_saver.load_data(fn_amp)
        amplitudes = data.get("amplitudes", None)
        rifft_space_grid = data.get("rifft_space_grid", None)
    except FileNotFoundError:
        data = {}
        amplitudes = None
        rifft_space_grid = None

    if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
        if amplitudes is not None:
            # Grid missing (the normal case: no producer ever writes
            # 'rifft_space_grid'): regenerate it WITHOUT re-reading the
            # 6.78 GB amplitudes file — the old fallback did a second full
            # load whose amplitudes were then discarded. Normalize exactly
            # once, as before.
            rifft_space_grid, _grids_shapeNd = _build_rifft_space_grid(
                processor, chunk_id, point_data_list
            )
            if len(rifft_space_grid) == 0:
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")
            amplitudes = normalize_residual_values_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=logger,
            )
        else:
            rifft_space_grid2, amplitudes2, _ = load_residual_field_and_generate_grid(
                processor,
                chunk_id,
                point_data_list,
                rifft_saver,
            )
            # amplitudes2 is already normalized inside — do NOT normalize
            # again (the old code's unreachable double-normalization).
            amplitudes = amplitudes2
            if rifft_space_grid is None or len(rifft_space_grid) == 0:
                rifft_space_grid = rifft_space_grid2
            if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")

    return data, amplitudes, np.asarray(rifft_space_grid)



__all__ = [
    "load_chunk_residual_field_and_grid",
    "load_residual_field_and_generate_grid",
    "resolve_output_dir",
]
