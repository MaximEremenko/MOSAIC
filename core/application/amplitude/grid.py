from __future__ import annotations

import inspect
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

from core.application.amplitude.runtime import _timed


class IntervalTask(NamedTuple):
    irecip_id: int
    element: str
    q_grid: np.ndarray
    q_amp: np.ndarray
    q_amp_av: np.ndarray


def _to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for axis in ("h", "k", "l"):
        rng = iv.get(f"{axis}_range")
        if rng is not None:
            out[f"{axis}_start"], out[f"{axis}_end"] = rng
    return out


def reciprocal_space_points_counter(interval: Dict[str, float], supercell: np.ndarray) -> int:
    supercell = np.asarray(supercell, dtype=float)
    step = 1.0 / supercell
    dim = len(supercell)

    def npts(start: float, end: float, st: float) -> int:
        return int(np.floor((end - start) / st + 0.5)) + 1

    h_n = npts(interval["h_start"], interval["h_end"], step[0])
    k_n = (
        npts(interval.get("k_start", 0.0), interval.get("k_end", 0.0), step[1])
        if dim > 1
        else 1
    )
    l_n = (
        npts(interval.get("l_start", 0.0), interval.get("l_end", 0.0), step[2])
        if dim > 2
        else 1
    )

    total = h_n * k_n * l_n
    if dim > 2 and not (interval["l_start"] == 0 and interval["l_end"] == 0):
        total *= 2
    return total


def _call_generate_mask(mask_strategy, hkl: np.ndarray, mask_params: Dict[str, Any]):
    sig = inspect.signature(mask_strategy.generate_mask)
    if len(sig.parameters) == 1:
        return mask_strategy.generate_mask(hkl)
    return mask_strategy.generate_mask(hkl, mask_params)


def generate_q_space_grid(
    interval: Dict[str, float],
    B_: np.ndarray,
    mask_parameters: Dict[str, Any],
    mask_strategy,
    supercell: np.ndarray,
) -> np.ndarray:
    supercell = np.asarray(supercell, dtype=float)
    int_supercell = np.round(supercell).astype(int)

    def get_axis_vals(axis: str, index: int):
        start = interval.get(f"{axis}_start", 0.0)
        end = interval.get(f"{axis}_end", 0.0)
        size = int_supercell[index]
        idx0 = int(np.ceil(start * size))
        idx1 = int(np.floor(end * size))
        return np.arange(idx0, idx1 + 1) / size

    h_vals = get_axis_vals("h", 0)
    k_vals = get_axis_vals("k", 1) if supercell.size > 1 else np.array([0.0])
    l_vals = get_axis_vals("l", 2) if supercell.size > 2 else np.array([0.0])

    mesh = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
    hkl = np.stack([m.ravel() for m in mesh], axis=1)

    mask = (
        _call_generate_mask(mask_strategy, hkl, mask_parameters)
        if mask_strategy is not None
        else np.ones(len(hkl), dtype=bool)
    )
    hkl_masked = hkl[mask]
    return 2 * np.pi * (hkl_masked[:, : supercell.size] @ B_)


def generate_q_space_grid_sync(*args, **kwargs):
    return generate_q_space_grid(*args, **kwargs)


def _generate_grid(
    dimensionality: int,
    step_sizes: np.ndarray,
    central_point: np.ndarray,
    dist_from_atom_center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    axes = []
    for index in range(dimensionality):
        dist = dist_from_atom_center[index]
        step = step_sizes[index]
        if step <= 0 or dist <= step:
            axis = np.array([0.0])
        else:
            axis = np.arange(-dist, dist + step - eps, step)
            if axis.size == 0:
                axis = np.array([0.0])
        axes.append(axis)

    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.vstack([m.ravel() for m in mesh]).T + central_point
    shape_nd = np.array(mesh[0].shape)
    return pts, shape_nd


def _process_chunk(chunk_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.array([point_data["coordinates"] for point_data in chunk_data])
    dist_vec = np.array([point_data["dist_from_atom_center"] for point_data in chunk_data])
    step_vec = np.array([point_data["step_in_frac"] for point_data in chunk_data])

    grids, shapes = [], []
    for cp_, dv, sv in zip(coords, dist_vec, step_vec):
        grid, shape = _generate_grid(coords.shape[1], sv, cp_, dv)
        grids.append(grid)
        shapes.append(shape)

    return np.vstack(grids), np.vstack(shapes)


def generate_rifft_grid(chunk_data: List[dict]):
    return _process_chunk(chunk_data)


def _build_rifft_grid_locally(chunk_data: List[dict]):
    with _timed("RIFFT grid build"):
        return _process_chunk(chunk_data)


def _point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    if not point_data_list:
        raise ValueError("point_data_list is empty")

    dim = len(point_data_list[0]["coordinates"])
    for point_data in point_data_list:
        if len(point_data["coordinates"]) != dim:
            raise ValueError("Mixed dimensionalities in point_data_list")

    vect = (dim,)
    dtype = np.dtype(
        [
            ("chunk_id", "<i4"),
            ("coordinates", "<f8", vect),
            ("dist_from_atom_center", "<f8", vect),
            ("step_in_frac", "<f8", vect),
        ]
    )

    out = np.empty(len(point_data_list), dtype=dtype)
    for index, point_data in enumerate(point_data_list):
        out[index] = (
            point_data["chunk_id"],
            point_data["coordinates"],
            point_data["dist_from_atom_center"],
            point_data["step_in_frac"],
        )
    return out.view(np.recarray)
