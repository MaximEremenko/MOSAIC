from __future__ import annotations

import copy
import inspect
import logging
import os
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np

from core.qspace.masking.mask_strategies import EqBasedStrategy, get_last_eq_mask_telemetry
from core.runtime.progress import timed


logger = logging.getLogger(__name__)
_LAST_QSPACE_GRID_TELEMETRY = None

_QSPACE_BLOCK_POINTS_DEFAULT = 500_000


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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.debug("Ignoring invalid integer %s=%r", name, raw)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean %s=%r", name, raw)
    return default


def _qspace_telemetry_enabled() -> bool:
    return _env_bool("MOSAIC_QSPACE_CAPTURE_TELEMETRY", False)


def _finish_qspace_telemetry(telemetry) -> None:
    global _LAST_QSPACE_GRID_TELEMETRY
    if telemetry is None:
        return
    _LAST_QSPACE_GRID_TELEMETRY = copy.deepcopy(telemetry)


def get_last_qspace_grid_telemetry():
    return copy.deepcopy(_LAST_QSPACE_GRID_TELEMETRY)


def _get_axis_values(
    interval: Dict[str, float],
    *,
    supercell: np.ndarray,
    int_supercell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return h_vals, k_vals, l_vals


def _build_mask_context(
    mask_parameters: Dict[str, Any],
    *,
    axis_values: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    context = dict(mask_parameters or {})
    coord_min = np.array(
        [axis[0] if len(axis) else 0.0 for axis in axis_values],
        dtype=np.float64,
    )
    coord_max = np.array(
        [axis[-1] if len(axis) else 0.0 for axis in axis_values],
        dtype=np.float64,
    )
    context.setdefault("interval_coord_min", coord_min)
    context.setdefault("interval_coord_max", coord_max)
    return context


def _strategy_supports_blockwise_mask(mask_strategy) -> bool:
    if mask_strategy is None:
        return True
    return bool(getattr(mask_strategy, "blockwise_safe", False))


def _mask_strategy_name(mask_strategy) -> str:
    return "NoMaskStrategy" if mask_strategy is None else type(mask_strategy).__name__


def _mask_backend_summary(mask_strategy, mask_telemetry) -> tuple[str, str]:
    if mask_strategy is None:
        return "none", "no-mask"
    if mask_telemetry is not None:
        return (
            str(mask_telemetry.get("backend_used", "cpu")),
            str(mask_telemetry.get("final_reason", mask_telemetry.get("decision_reason", "strategy-mask"))),
        )
    return "cpu", "strategy-mask"


def _qspace_mode_decision(*, mask_strategy, total_points: int, block_points: int) -> tuple[bool, str]:
    if int(total_points) <= int(block_points):
        return False, "below-block-threshold"
    if _strategy_supports_blockwise_mask(mask_strategy):
        if mask_strategy is None:
            return True, "no-mask-large-interval"
        return True, "blockwise-safe-strategy"
    return False, "strategy-requires-full-interval"


def _build_hkl_block(
    *,
    axis_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    start: int,
    stop: int,
) -> np.ndarray:
    shape = tuple(len(axis) for axis in axis_values)
    flat_indices = np.arange(start, stop, dtype=np.int64)
    unravelled = np.unravel_index(flat_indices, shape, order="C")
    return np.column_stack(
        [axis_values[index][unravelled[index]] for index in range(len(axis_values))]
    )


def _generate_q_space_grid_full(
    *,
    axis_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    B_: np.ndarray,
    mask_parameters: Dict[str, Any],
    mask_strategy,
    dim: int,
) -> tuple[np.ndarray, dict]:
    mesh_start = time.perf_counter()
    mesh = np.meshgrid(*axis_values, indexing="ij")
    hkl = np.stack([m.ravel() for m in mesh], axis=1)
    mesh_build_seconds = time.perf_counter() - mesh_start
    mask_start = time.perf_counter()
    mask = (
        _call_generate_mask(mask_strategy, hkl, mask_parameters)
        if mask_strategy is not None
        else np.ones(len(hkl), dtype=bool)
    )
    mask_seconds = time.perf_counter() - mask_start
    mask_telemetry = (
        get_last_eq_mask_telemetry()
        if isinstance(mask_strategy, EqBasedStrategy) and _qspace_telemetry_enabled()
        else None
    )
    mask_backend_used, mask_reason = _mask_backend_summary(mask_strategy, mask_telemetry)
    convert_start = time.perf_counter()
    hkl_masked = hkl[mask]
    q_grid = 2 * np.pi * (hkl_masked[:, :dim] @ B_)
    q_conversion_seconds = time.perf_counter() - convert_start
    return q_grid, {
        "mesh_build_seconds": float(mesh_build_seconds),
        "mask_seconds": float(mask_seconds),
        "q_conversion_seconds": float(q_conversion_seconds),
        "accepted_points": int(hkl_masked.shape[0]),
        "block_count": 1,
        "blocks": [
            {
                "block_index": 0,
                "input_points": int(hkl.shape[0]),
                "accepted_points": int(hkl_masked.shape[0]),
                "accepted_fraction": float(hkl_masked.shape[0] / max(len(hkl), 1)),
                "hkl_build_seconds": float(mesh_build_seconds),
                "mask_seconds": float(mask_seconds),
                "q_conversion_seconds": float(q_conversion_seconds),
                "mask_strategy": _mask_strategy_name(mask_strategy),
                "mask_backend_used": mask_backend_used,
                "mask_reason": mask_reason,
            }
        ],
        "mask_backend_counts": {mask_backend_used: 1},
    }


def _generate_q_space_grid_blockwise(
    *,
    axis_values: tuple[np.ndarray, np.ndarray, np.ndarray],
    B_: np.ndarray,
    mask_parameters: Dict[str, Any],
    mask_strategy,
    dim: int,
    block_points: int,
) -> tuple[np.ndarray, dict]:
    total_points = int(np.prod([len(axis) for axis in axis_values], dtype=np.int64))
    q_blocks: list[np.ndarray] = []
    accepted_points = 0
    block_count = 0
    mesh_build_seconds_total = 0.0
    mask_seconds_total = 0.0
    q_conversion_seconds_total = 0.0
    block_metrics: list[dict] = []
    mask_backend_counts: dict[str, int] = {}

    for start in range(0, total_points, int(block_points)):
        stop = min(total_points, start + int(block_points))
        build_start = time.perf_counter()
        hkl_block = _build_hkl_block(
            axis_values=axis_values,
            start=start,
            stop=stop,
        )
        build_seconds = time.perf_counter() - build_start
        mask_start = time.perf_counter()
        mask = (
            _call_generate_mask(mask_strategy, hkl_block, mask_parameters)
            if mask_strategy is not None
            else np.ones(len(hkl_block), dtype=bool)
        )
        mask_seconds = time.perf_counter() - mask_start
        mask_telemetry = (
            get_last_eq_mask_telemetry()
            if isinstance(mask_strategy, EqBasedStrategy) and _qspace_telemetry_enabled()
            else None
        )
        mask_backend_used, mask_reason = _mask_backend_summary(mask_strategy, mask_telemetry)
        convert_start = time.perf_counter()
        hkl_masked = hkl_block[mask]
        if hkl_masked.size:
            q_blocks.append(2 * np.pi * (hkl_masked[:, :dim] @ B_))
            accepted_points += int(hkl_masked.shape[0])
        q_conversion_seconds = time.perf_counter() - convert_start
        mesh_build_seconds_total += build_seconds
        mask_seconds_total += mask_seconds
        q_conversion_seconds_total += q_conversion_seconds
        mask_backend_counts[mask_backend_used] = mask_backend_counts.get(mask_backend_used, 0) + 1
        block_metrics.append(
            {
                "block_index": int(block_count),
                "input_points": int(hkl_block.shape[0]),
                "accepted_points": int(hkl_masked.shape[0]),
                "accepted_fraction": float(hkl_masked.shape[0] / max(len(hkl_block), 1)),
                "hkl_build_seconds": float(build_seconds),
                "mask_seconds": float(mask_seconds),
                "q_conversion_seconds": float(q_conversion_seconds),
                "mask_strategy": _mask_strategy_name(mask_strategy),
                "mask_backend_used": mask_backend_used,
                "mask_reason": mask_reason,
            }
        )
        block_count += 1

    if not q_blocks:
        return np.empty((0, dim), dtype=np.float64), {
            "mesh_build_seconds": float(mesh_build_seconds_total),
            "mask_seconds": float(mask_seconds_total),
            "q_conversion_seconds": float(q_conversion_seconds_total),
            "accepted_points": 0,
            "block_count": int(block_count),
            "blocks": block_metrics,
            "mask_backend_counts": mask_backend_counts,
        }

    q_grid = np.vstack(q_blocks)
    logger.debug(
        "generate_q_space_grid blockwise | total_points=%d block_points=%d blocks=%d accepted_points=%d",
        total_points,
        block_points,
        block_count,
        accepted_points,
    )
    return q_grid, {
        "mesh_build_seconds": float(mesh_build_seconds_total),
        "mask_seconds": float(mask_seconds_total),
        "q_conversion_seconds": float(q_conversion_seconds_total),
        "accepted_points": int(accepted_points),
        "block_count": int(block_count),
        "blocks": block_metrics,
        "mask_backend_counts": mask_backend_counts,
    }


def generate_q_space_grid(
    interval: Dict[str, float],
    B_: np.ndarray,
    mask_parameters: Dict[str, Any],
    mask_strategy,
    supercell: np.ndarray,
) -> np.ndarray:
    supercell = np.asarray(supercell, dtype=float)
    int_supercell = np.round(supercell).astype(int)
    axis_values = _get_axis_values(
        interval,
        supercell=supercell,
        int_supercell=int_supercell,
    )
    mask_context = _build_mask_context(
        mask_parameters,
        axis_values=axis_values,
    )
    total_points = int(np.prod([len(axis) for axis in axis_values], dtype=np.int64))
    block_points = max(1, _env_int("MOSAIC_QSPACE_BLOCK_POINTS", _QSPACE_BLOCK_POINTS_DEFAULT))
    start_time = time.perf_counter()
    use_blockwise, decision_reason = _qspace_mode_decision(
        mask_strategy=mask_strategy,
        total_points=total_points,
        block_points=block_points,
    )
    telemetry = {
        "mode": None,
        "decision_reason": decision_reason,
        "mask_strategy": _mask_strategy_name(mask_strategy),
        "blockwise_safe": bool(_strategy_supports_blockwise_mask(mask_strategy)),
        "total_points": int(total_points),
        "block_points": int(block_points),
        "dim": int(supercell.size),
        "accepted_points": 0,
        "accepted_fraction": 0.0,
        "block_count": 0,
        "mesh_build_seconds": 0.0,
        "mask_seconds": 0.0,
        "q_conversion_seconds": 0.0,
        "total_seconds": 0.0,
        "mask_backend_counts": {},
        "blocks": [] if _qspace_telemetry_enabled() else None,
    }

    if use_blockwise:
        q_grid, details = _generate_q_space_grid_blockwise(
            axis_values=axis_values,
            B_=B_,
            mask_parameters=mask_context,
            mask_strategy=mask_strategy,
            dim=supercell.size,
            block_points=block_points,
        )
        telemetry["mode"] = "blockwise"
        telemetry["accepted_points"] = int(details["accepted_points"])
        telemetry["accepted_fraction"] = float(
            details["accepted_points"] / max(total_points, 1)
        )
        telemetry["block_count"] = int(details["block_count"])
        telemetry["mesh_build_seconds"] = float(details["mesh_build_seconds"])
        telemetry["mask_seconds"] = float(details["mask_seconds"])
        telemetry["q_conversion_seconds"] = float(details["q_conversion_seconds"])
        telemetry["mask_backend_counts"] = dict(details["mask_backend_counts"])
        if telemetry["blocks"] is not None:
            telemetry["blocks"] = list(details["blocks"])
        telemetry["total_seconds"] = float(time.perf_counter() - start_time)
        logger.debug(
            "generate_q_space_grid mode=blockwise total_points=%d block_points=%d blocks=%d accepted_fraction=%.6f build=%.6fs mask=%.6fs q=%.6fs total=%.6fs",
            total_points,
            block_points,
            telemetry["block_count"],
            telemetry["accepted_fraction"],
            telemetry["mesh_build_seconds"],
            telemetry["mask_seconds"],
            telemetry["q_conversion_seconds"],
            telemetry["total_seconds"],
        )
        _finish_qspace_telemetry(telemetry if _qspace_telemetry_enabled() else None)
        return q_grid

    q_grid, details = _generate_q_space_grid_full(
        axis_values=axis_values,
        B_=B_,
        mask_parameters=mask_context,
        mask_strategy=mask_strategy,
        dim=supercell.size,
    )
    telemetry["mode"] = "full"
    telemetry["accepted_points"] = int(details["accepted_points"])
    telemetry["accepted_fraction"] = float(details["accepted_points"] / max(total_points, 1))
    telemetry["block_count"] = int(details["block_count"])
    telemetry["mesh_build_seconds"] = float(details["mesh_build_seconds"])
    telemetry["mask_seconds"] = float(details["mask_seconds"])
    telemetry["q_conversion_seconds"] = float(details["q_conversion_seconds"])
    telemetry["mask_backend_counts"] = dict(details["mask_backend_counts"])
    if telemetry["blocks"] is not None:
        telemetry["blocks"] = list(details["blocks"])
    telemetry["total_seconds"] = float(time.perf_counter() - start_time)
    logger.debug(
        "generate_q_space_grid mode=full total_points=%d accepted_fraction=%.6f build=%.6fs mask=%.6fs q=%.6fs total=%.6fs reason=%s",
        total_points,
        telemetry["accepted_fraction"],
        telemetry["mesh_build_seconds"],
        telemetry["mask_seconds"],
        telemetry["q_conversion_seconds"],
        telemetry["total_seconds"],
        decision_reason,
    )
    _finish_qspace_telemetry(telemetry if _qspace_telemetry_enabled() else None)
    return q_grid


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
    with timed("RIFFT grid build"):
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
