from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from core.scattering.grid import (
    IntervalTask,
    _point_list_to_recarray,
    _process_chunk as _build_rifft_grid_for_chunk,
    _to_interval_dict,
    generate_q_space_grid,
    generate_q_space_grid_sync,
    reciprocal_space_points_counter,
)
from core.adapters.cunufft_wrapper import execute_cunufft


def to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    return _to_interval_dict(iv)


def build_rifft_grid_for_chunk(chunk_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return _build_rifft_grid_for_chunk(chunk_data)


def point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    return _point_list_to_recarray(point_data_list)


def compute_interval_element_contribution(
    interval: dict,
    q_grid: np.ndarray,
    element: str,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
) -> Tuple | None:
    ff = ff_factory.calculate(q_grid, element, charge=charge)
    mask = elements_arr == element
    if not np.any(mask):
        return None
    q_amp = ff * execute_cunufft(
        original_coords[mask],
        np.ones(mask.sum()),
        q_grid,
        eps=1e-12,
    )
    q_av = execute_cunufft(
        cells_origin,
        np.ones(original_coords.shape[0]),
        q_grid,
        eps=1e-12,
    )
    q_delta = execute_cunufft(
        original_coords[mask] - cells_origin[mask],
        np.ones(mask.sum()),
        q_grid,
        eps=1e-12,
    )
    q_av_final = ff * q_av * q_delta / original_coords.shape[0]
    return (interval["id"], element, q_grid, q_amp, q_av_final)


def compute_interval_coeff_contribution(
    interval: dict,
    q_grid: np.ndarray,
    coeff: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
) -> Tuple:
    n_points = original_coords.shape[0]
    coeff_arr = coeff * (np.ones(n_points) + 1j * np.zeros(n_points))
    q_amplitudes = execute_cunufft(original_coords, coeff_arr, q_grid, eps=1e-12)
    q_amplitudes_av = execute_cunufft(
        cells_origin,
        coeff_arr * 0.0 + 1.0,
        q_grid,
        eps=1e-12,
    )
    q_amplitudes_delta = execute_cunufft(
        original_coords - cells_origin,
        coeff_arr,
        q_grid,
        eps=1e-12,
    )
    q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / n_points
    return (interval["id"], "All", q_grid, q_amplitudes, q_amplitudes_av_final)


def aggregate_interval_contributions(
    contributions: list[tuple],
    *,
    use_coeff: bool,
) -> IntervalTask:
    if use_coeff:
        interval_id, element, q_grid, q_amp, q_amp_av = contributions[0]
        return IntervalTask(interval_id, element, q_grid, q_amp, q_amp_av)

    interval_id = contributions[0][0]
    q_grid = contributions[0][2]
    q_amp = np.sum([contribution[3] for contribution in contributions], axis=0)
    q_amp_av = np.sum([contribution[4] for contribution in contributions], axis=0)
    return IntervalTask(interval_id, "All", q_grid, q_amp, q_amp_av)


__all__ = [
    "IntervalTask",
    "aggregate_interval_contributions",
    "build_rifft_grid_for_chunk",
    "compute_interval_coeff_contribution",
    "compute_interval_element_contribution",
    "generate_q_space_grid",
    "generate_q_space_grid_sync",
    "point_list_to_recarray",
    "reciprocal_space_points_counter",
    "to_interval_dict",
]
