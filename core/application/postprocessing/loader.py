from __future__ import annotations

import logging
import os

import numpy as np

from core.application.postprocessing.io import normalize_amplitudes_ntotal


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


def load_amplitudes_and_generate_grid(processor, chunk_id, point_data_list, rifft_saver):
    filename = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
    try:
        data = rifft_saver.load_data(filename)
        amplitudes = data.get("amplitudes", None)
        if amplitudes is None:
            logging.getLogger(__name__).warning("Amplitudes not found in %s", filename)
            return np.array([]), None, None

        amplitudes = normalize_amplitudes_ntotal(
            amplitudes,
            rifft_saver=rifft_saver,
            chunk_id=chunk_id,
            logger=logging.getLogger(__name__),
        )

        grids = []
        grids_shapeNd = []
        central_point_ids = []
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
            central_point_ids.extend([point_data["central_point_id"]] * len(grid_points))

        rifft_space_grid = (
            np.hstack((np.vstack(grids), np.array(central_point_ids)[:, None]))
            if grids
            else np.array([])
        )
        return rifft_space_grid, amplitudes, grids_shapeNd
    except FileNotFoundError:
        logging.getLogger(__name__).warning("File not found: %s", filename)
        return np.array([]), None, None


def load_chunk_amplitudes_and_grid(processor, *, chunk_id, point_data_list, rifft_saver, logger):
    fn_amp = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
    try:
        data = rifft_saver.load_data(fn_amp)
        amplitudes = data.get("amplitudes", None)
        rifft_space_grid = data.get("rifft_space_grid", None)
    except FileNotFoundError:
        data = {}
        amplitudes = None
        rifft_space_grid = None

    if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
        rifft_space_grid2, amplitudes2, _ = load_amplitudes_and_generate_grid(
            processor,
            chunk_id,
            point_data_list,
            rifft_saver,
        )
        if amplitudes is None:
            amplitudes = amplitudes2
        if rifft_space_grid is None:
            rifft_space_grid = rifft_space_grid2
        if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
            raise RuntimeError(f"Nothing to process for chunk {chunk_id}")
        amplitudes = normalize_amplitudes_ntotal(
            amplitudes,
            rifft_saver=rifft_saver,
            chunk_id=chunk_id,
            logger=logger,
        )

    return data, amplitudes, np.asarray(rifft_space_grid)
