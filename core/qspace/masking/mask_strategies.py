# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:50:03 2024

@author: Maksim Eremenko
"""

# strategies/mask_strategies.py

import copy
import logging
import os
import time

import numpy as np
import sympy as sp
from core.qspace.masking import (
    allowed_locals,
    parse_logic,
    preprocess,
    symbol_map,
)
import pandas as pd


logger = logging.getLogger(__name__)
_LAST_EQ_MASK_TELEMETRY = None

_MASK_GPU_MIN_POINTS_DEFAULT = 250_000
_MASK_GPU_ESTIMATED_BYTES_PER_POINT = 128
_MASK_GPU_ESTIMATE_SAFETY_FACTOR = 2.0
_MASK_GPU_MIN_RESERVE_BYTES = 256 << 20
_MASK_GPU_MAX_RESERVE_BYTES = 2 << 30


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower()


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


def _mask_telemetry_enabled() -> bool:
    return _env_bool("MOSAIC_MASK_CAPTURE_TELEMETRY", False)


def _finish_mask_telemetry(telemetry) -> None:
    global _LAST_EQ_MASK_TELEMETRY
    if telemetry is None:
        return
    _LAST_EQ_MASK_TELEMETRY = copy.deepcopy(telemetry)


def get_last_eq_mask_telemetry():
    return copy.deepcopy(_LAST_EQ_MASK_TELEMETRY)


def _mask_gpu_estimated_bytes_per_point() -> int:
    return max(
        32,
        _env_int(
            "MOSAIC_MASK_EQUATION_GPU_ESTIMATED_BYTES_PER_POINT",
            _MASK_GPU_ESTIMATED_BYTES_PER_POINT,
        ),
    )


def _mask_gpu_reserve_override_bytes() -> int | None:
    raw = os.getenv("MOSAIC_MASK_EQUATION_GPU_RESERVE_BYTES")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logger.debug("Ignoring invalid integer MOSAIC_MASK_EQUATION_GPU_RESERVE_BYTES=%r", raw)
        return None


def _min_variadic(lib, *args):
    result = args[0]
    for arg in args[1:]:
        result = lib.minimum(result, arg)
    return result


def _max_variadic(lib, *args):
    result = args[0]
    for arg in args[1:]:
        result = lib.maximum(result, arg)
    return result


def _numpy_func_map() -> dict:
    return {
        "Mod": np.mod,
        "Heaviside": np.heaviside,
        "Abs": np.abs,
        "Min": lambda *args: _min_variadic(np, *args),
        "Max": lambda *args: _max_variadic(np, *args),
    }


def _cupy_func_map(cp_mod) -> dict:
    return {
        "Mod": cp_mod.mod,
        "Heaviside": lambda x, h0: cp_mod.heaviside(x, h0),
        "Abs": cp_mod.abs,
        "Min": lambda *args: _min_variadic(cp_mod, *args),
        "Max": lambda *args: _max_variadic(cp_mod, *args),
    }


def _load_cupy():
    if os.getenv("MOSAIC_NUFFT_CPU_ONLY", "0") == "1":
        return None
    try:
        import cupy as cp_mod  # type: ignore

        try:
            if cp_mod.cuda.runtime.getDeviceCount() <= 0:
                return None
        except cp_mod.cuda.runtime.CUDARuntimeError:
            return None
        return cp_mod
    except ImportError:
        return None


def _mask_gpu_memory_info(cp_mod) -> tuple[int, int]:
    with cp_mod.cuda.Device(0):
        free_bytes, total_bytes = cp_mod.cuda.runtime.memGetInfo()
    return int(free_bytes), int(total_bytes)


def _estimate_mask_gpu_bytes(point_count: int, dim: int) -> int:
    bytes_per_point = _mask_gpu_estimated_bytes_per_point()
    estimated = int(point_count) * max(
        bytes_per_point,
        int(dim) * 8 + 96,
    )
    return int(estimated * _MASK_GPU_ESTIMATE_SAFETY_FACTOR)


def _mask_gpu_reserve_bytes(*, free_bytes: int, total_bytes: int) -> int:
    reserve_override = _mask_gpu_reserve_override_bytes()
    if reserve_override is not None:
        return int(min(max(reserve_override, 0), max(free_bytes - (64 << 20), 0)))
    reserve = max(
        _MASK_GPU_MIN_RESERVE_BYTES,
        min(_MASK_GPU_MAX_RESERVE_BYTES, max(total_bytes // 8, free_bytes // 4)),
    )
    return int(min(reserve, max(free_bytes - (64 << 20), 0)))


def _resolve_mask_backend(
    *,
    point_count: int,
    dim: int,
    gpu_available: bool,
    free_bytes: int | None,
    total_bytes: int | None,
    min_points: int,
    backend_override: str,
) -> dict:
    decision = {
        "backend": "cpu",
        "reason": "cpu-default",
        "point_count": int(point_count),
        "dim": int(dim),
        "free_bytes": None if free_bytes is None else int(free_bytes),
        "total_bytes": None if total_bytes is None else int(total_bytes),
        "reserve_bytes": 0,
        "estimated_bytes": 0,
        "min_points": int(min_points),
        "backend_override": backend_override,
    }
    if backend_override == "cpu":
        decision["reason"] = "env-force-cpu"
        return decision
    if not gpu_available:
        decision["reason"] = "gpu-unavailable"
        return decision
    if free_bytes is None or total_bytes is None:
        decision["reason"] = "gpu-memory-unknown"
        return decision

    reserve_bytes = _mask_gpu_reserve_bytes(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
    )
    estimated_bytes = _estimate_mask_gpu_bytes(point_count, dim)
    decision["reserve_bytes"] = int(reserve_bytes)
    decision["estimated_bytes"] = int(estimated_bytes)

    if backend_override != "gpu" and int(point_count) < int(min_points):
        decision["reason"] = "below-min-points"
        return decision

    if max(0, int(free_bytes) - int(reserve_bytes)) < int(estimated_bytes):
        decision["reason"] = "insufficient-free-vram"
        return decision

    decision["backend"] = "gpu"
    decision["reason"] = "gpu-eligible"
    return decision


def _release_mask_gpu_memory(cp_mod) -> None:
    try:
        cp_mod.get_default_memory_pool().free_all_blocks()
        cp_mod.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


class DefaultMaskStrategy:
    blockwise_safe = True

    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Returns a mask of all True values (no masking).

        Args:
            hkl_mesh (np.ndarray): An array of hkl points.

        Returns:
            np.ndarray: A boolean array of all True values.
        """
        return np.ones(hkl_mesh.shape[0], dtype=bool)
    

class CustomReciprocalSpacePointsStrategy:
    blockwise_safe = True

    def __init__(self, file_path: str, ih: np.ndarray, ik: np.ndarray, il: np.ndarray):
        """
        Initializes the strategy with the file containing custom reciprocal space points.

        Args:
            file_path (str): Path to the file with custom reciprocal space points.
            ih (np.ndarray): h indices array.
            ik (np.ndarray): k indices array.
            il (np.ndarray): l indices array.
        """
        self.file_path = file_path
        self.ih = ih
        self.ik = ik
        self.il = il
        self._reflections_in_lim = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_reflections_in_lim"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reflections_in_lim = None

    def _load_reflections_in_lim(self) -> np.ndarray:
        if self._reflections_in_lim is None:
            df_hkl_reflections = pd.read_table(
                self.file_path, skiprows=0, sep='\\s+', engine='python'
            )
            hkl_reflections = df_hkl_reflections.values

            h_min, h_max = self.ih.min() - 0.5, self.ih.max() + 0.5
            k_min, k_max = self.ik.min() - 0.5, self.ik.max() + 0.5
            l_min, l_max = self.il.min() - 0.5, self.il.max() + 0.5

            mask = (
                (hkl_reflections[:, 0] >= h_min) & (hkl_reflections[:, 0] <= h_max) &
                (hkl_reflections[:, 1] >= k_min) & (hkl_reflections[:, 1] <= k_max) &
                (hkl_reflections[:, 2] >= l_min) & (hkl_reflections[:, 2] <= l_max)
            )
            self._reflections_in_lim = np.ascontiguousarray(hkl_reflections[mask])
        return self._reflections_in_lim

    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Generates a mask based on custom reciprocal space points from a file.

        Args:
            hkl_mesh (np.ndarray): An array of hkl points.

        Returns:
            np.ndarray: A boolean array representing the mask.
        """
        hkl_reflections_in_lim = self._load_reflections_in_lim()

        # Create mask for hkl_mesh
        mask = self._create_mask_from_reflections(hkl_mesh, hkl_reflections_in_lim)
        return mask

    def _create_mask_from_reflections(self, hkl_mesh: np.ndarray, reflections: np.ndarray) -> np.ndarray:
        # Use structured arrays for comparison
        hkl_dtype = np.dtype([('h', hkl_mesh.dtype), ('k', hkl_mesh.dtype), ('l', hkl_mesh.dtype)])
        hkl_mesh_structured = hkl_mesh.view(hkl_dtype).reshape(-1)
        reflections_structured = reflections.view(hkl_dtype).reshape(-1)
        mask = np.isin(hkl_mesh_structured, reflections_structured)
        return mask
    
# from mantid.geometry import SpaceGroupFactory, PointGroupFactory
# class SpaceGroupSymmetryStrategy(IMaskStrategy):
#     def __init__(self, space_group_symmetry: str, ih: np.ndarray, ik: np.ndarray, il: np.ndarray):
#         """
#         Initializes the strategy with space group symmetry information.

#         Args:
#             space_group_symmetry (str): Space group symbol.
#             ih (np.ndarray): h indices array.
#             ik (np.ndarray): k indices array.
#             il (np.ndarray): l indices array.
#         """
#         self.space_group_symmetry = space_group_symmetry
#         self.ih = ih
#         self.ik = ik
#         self.il = il

#     def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
#         """
#         Generates a mask based on space group symmetry.

#         Args:
#             hkl_mesh (np.ndarray): An array of hkl points.

#         Returns:
#             np.ndarray: A boolean array representing the mask.
#         """
#         # Generate central points (assuming origin)
#         central_point = np.array([0, 0, 0])
#         acpx = self._find_central_points(central_point[0], self.ih)
#         acpy = self._find_central_points(central_point[1], self.ik)
#         acpz = self._find_central_points(central_point[2], self.il)

#         # Generate all combinations of central points
#         hkl_reflections = self._generate_hkl_reflections(acpx, acpy, acpz)

#         # Filter reflections using space group symmetry
#         allowed_reflections = self._filter_allowed_reflections(hkl_reflections)

#         # Create mask for hkl_mesh
#         mask = self._create_mask_from_reflections(hkl_mesh, allowed_reflections)
#         return mask

#     def _find_central_points(self, coordinate: float, indices: np.ndarray) -> np.ndarray:
#         min_index, max_index = indices.min() - 0.5, indices.max() + 0.5
#         # Implement logic to find central points within the indices range
#         return np.array([coordinate])  # Placeholder

#     def _generate_hkl_reflections(self, acpx: np.ndarray, acpy: np.ndarray, acpz: np.ndarray) -> np.ndarray:
#         hkl_reflections = np.array(np.meshgrid(acpx, acpy, acpz)).T.reshape(-1, 3)
#         return hkl_reflections

#     def _filter_allowed_reflections(self, hkl_reflections: np.ndarray) -> np.ndarray:
#         sg_object = SpaceGroupFactory.createSpaceGroup(self.space_group_symmetry)
#         is_allowed = [sg_object.isAllowedReflection(hkl) for hkl in hkl_reflections]
#         allowed_reflections = hkl_reflections[is_allowed]
#         return allowed_reflections

#     def _create_mask_from_reflections(self, hkl_mesh: np.ndarray, reflections: np.ndarray) -> np.ndarray:
#         hkl_dtype = np.dtype([('h', hkl_mesh.dtype), ('k', hkl_mesh.dtype), ('l', hkl_mesh.dtype)])
#         hkl_mesh_structured = hkl_mesh.view(hkl_dtype).reshape(-1)
#         reflections_structured = reflections.view(hkl_dtype).reshape(-1)
#         mask = np.in1d(hkl_mesh_structured, reflections_structured)
#         return mask



class EqBasedStrategy:
    blockwise_safe = True

    """
    Strategy that masks hkl points according to an arbitrary logical equation.
    
    Example:
        condition = (
            "(cos(pi*h) + cos(pi*k) + cos(pi*l) > -0.5 and "
            "cos(pi*h) + cos(pi*k) + cos(pi*l) < 0.5) and "
            "sqrt((acos(-cos(2*pi*h))/(2*pi))**2 + … ) >= 0.2"
        )
        strat = EqBasedStrategy(condition)
        mask = strat.generate_mask(hkl_mesh)
    """
    def __init__(self, condition: str):
        """
        Args:
            condition: A Sympy‐friendly logical string in terms of h, k, l.
        """
        self._condition = condition
        self._normalized_condition = preprocess(condition)
        self._expr = parse_logic(self._normalized_condition, symbol_map, allowed_locals)
        self._f_cpu = None
        self._f_gpu = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_f_cpu"] = None
        state["_f_gpu"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._f_cpu = None
        self._f_gpu = None

    def _ensure_cpu_callable(self):
        if self._f_cpu is None:
            h, k, l = symbol_map["h"], symbol_map["k"], symbol_map["l"]
            self._f_cpu = sp.lambdify(
                (h, k, l),
                self._expr,
                modules=[_numpy_func_map(), "numpy"],
            )
        return self._f_cpu

    def _ensure_gpu_callable(self, cp_mod):
        if self._f_gpu is None:
            h, k, l = symbol_map["h"], symbol_map["k"], symbol_map["l"]
            self._f_gpu = sp.lambdify(
                (h, k, l),
                self._expr,
                modules=[_cupy_func_map(cp_mod), "cupy"],
            )
        return self._f_gpu

    def _split_components(self, hkl_mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hkl_mesh = np.asarray(hkl_mesh, dtype=np.float64)
        if hkl_mesh.ndim != 2:
            raise ValueError(f"hkl_mesh must be 2D, got shape={hkl_mesh.shape}")
        if hkl_mesh.shape[1] >= 3:
            return hkl_mesh[:, 0], hkl_mesh[:, 1], hkl_mesh[:, 2]
        if hkl_mesh.shape[1] == 2:
            h_vals = hkl_mesh[:, 0]
            k_vals = hkl_mesh[:, 1]
            return h_vals, k_vals, np.zeros_like(h_vals)
        if hkl_mesh.shape[1] == 1:
            h_vals = hkl_mesh[:, 0]
            zeros = np.zeros_like(h_vals)
            return h_vals, zeros, zeros
        raise ValueError(f"hkl_mesh must have at least 1 column, got shape={hkl_mesh.shape}")

    def _cpu_mask(self, h_vals: np.ndarray, k_vals: np.ndarray, l_vals: np.ndarray) -> np.ndarray:
        func = self._ensure_cpu_callable()
        return np.asarray(func(h_vals, k_vals, l_vals), dtype=bool).reshape(-1)

    def _backend_decision(self, point_count: int, dim: int) -> tuple[dict, object | None]:
        backend_override = _env_str("MOSAIC_MASK_EQUATION_BACKEND", "auto")
        min_points = _env_int(
            "MOSAIC_MASK_EQUATION_GPU_MIN_POINTS",
            _MASK_GPU_MIN_POINTS_DEFAULT,
        )
        cp_mod = _load_cupy()
        if cp_mod is None:
            decision = _resolve_mask_backend(
                point_count=point_count,
                dim=dim,
                gpu_available=False,
                free_bytes=None,
                total_bytes=None,
                min_points=min_points,
                backend_override=backend_override,
            )
            return decision, None

        try:
            free_bytes, total_bytes = _mask_gpu_memory_info(cp_mod)
        except Exception:
            decision = _resolve_mask_backend(
                point_count=point_count,
                dim=dim,
                gpu_available=False,
                free_bytes=None,
                total_bytes=None,
                min_points=min_points,
                backend_override=backend_override,
            )
            return decision, None

        decision = _resolve_mask_backend(
            point_count=point_count,
            dim=dim,
            gpu_available=True,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            min_points=min_points,
            backend_override=backend_override,
        )
        return decision, cp_mod

    def _validate_gpu_result(
        self,
        *,
        gpu_mask: np.ndarray,
        h_vals: np.ndarray,
        k_vals: np.ndarray,
        l_vals: np.ndarray,
    ) -> None:
        if not _env_bool("MOSAIC_MASK_GPU_VALIDATE", False):
            return
        cpu_mask = self._cpu_mask(h_vals, k_vals, l_vals)
        if not np.array_equal(cpu_mask, gpu_mask):
            logger.warning(
                "EqBasedStrategy GPU validation mismatch | condition=%s points=%d",
                self._condition,
                len(gpu_mask),
            )

    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Evaluate the logical condition at each point in hkl_mesh.

        Args:
            hkl_mesh: np.ndarray of shape (N, 3), columns = h, k, l.

        Returns:
            mask: np.ndarray of shape (N,), dtype=bool.
        """
        h_vals, k_vals, l_vals = self._split_components(hkl_mesh)
        point_count = int(h_vals.shape[0])
        dim = max(1, min(3, int(np.asarray(hkl_mesh).shape[1])))
        decision, cp_mod = self._backend_decision(point_count, dim)
        start_time = time.perf_counter()
        decision_reason = decision.get("reason")
        telemetry = None
        if _mask_telemetry_enabled():
            telemetry = {
                "point_count": point_count,
                "dim": dim,
                "decision_backend": decision.get("backend"),
                "decision_reason": decision_reason,
                "backend_used": None,
                "backend_override": decision.get("backend_override"),
                "free_bytes": decision.get("free_bytes"),
                "total_bytes": decision.get("total_bytes"),
                "reserve_bytes": decision.get("reserve_bytes"),
                "estimated_bytes": decision.get("estimated_bytes"),
                "estimated_bytes_per_point": int(_mask_gpu_estimated_bytes_per_point()),
                "min_points": decision.get("min_points"),
                "fallback_reason": None,
                "validated_gpu_output": bool(_env_bool("MOSAIC_MASK_GPU_VALIDATE", False)),
            }

        if decision["backend"] == "gpu" and cp_mod is not None:
            try:
                func = self._ensure_gpu_callable(cp_mod)
                d_h = cp_mod.asarray(h_vals)
                d_k = cp_mod.asarray(k_vals)
                d_l = cp_mod.asarray(l_vals)
                d_mask = func(d_h, d_k, d_l)
                mask = np.asarray(cp_mod.asnumpy(d_mask), dtype=bool).reshape(-1)
                self._validate_gpu_result(
                    gpu_mask=mask,
                    h_vals=h_vals,
                    k_vals=k_vals,
                    l_vals=l_vals,
                )
                logger.debug(
                    "EqBasedStrategy backend=gpu points=%d dim=%d free_vram=%s reserve_bytes=%s estimated_bytes=%s duration=%.6fs",
                    point_count,
                    dim,
                    decision.get("free_bytes"),
                    decision.get("reserve_bytes"),
                    decision.get("estimated_bytes"),
                    time.perf_counter() - start_time,
                )
                if telemetry is not None:
                    telemetry["backend_used"] = "gpu"
                    telemetry["final_reason"] = decision_reason
                    telemetry["duration_seconds"] = float(time.perf_counter() - start_time)
                    _finish_mask_telemetry(telemetry)
                return mask
            except Exception as exc:
                decision["reason"] = f"gpu-fallback:{type(exc).__name__}"
                logger.debug(
                    "EqBasedStrategy GPU fallback to CPU | points=%d dim=%d reason=%s",
                    point_count,
                    dim,
                    exc,
                )
                if telemetry is not None:
                    telemetry["fallback_reason"] = type(exc).__name__
            finally:
                _release_mask_gpu_memory(cp_mod)

        mask = self._cpu_mask(h_vals, k_vals, l_vals)
        logger.debug(
            "EqBasedStrategy backend=cpu points=%d dim=%d reason=%s duration=%.6fs",
            point_count,
            dim,
            decision.get("reason"),
            time.perf_counter() - start_time,
        )
        if telemetry is not None:
            telemetry["backend_used"] = "cpu"
            telemetry["final_reason"] = decision.get("reason")
            telemetry["duration_seconds"] = float(time.perf_counter() - start_time)
            _finish_mask_telemetry(telemetry)
        return mask



# class CoordinateBasedStrategy(IMaskStrategy):
#     def __init__(
#         self,
#         coordinate: np.ndarray,
#         ih: np.ndarray,
#         ik: np.ndarray,
#         il: np.ndarray,
#         shape_strategy: ShapeStrategy,
#         symmetry_applier=None  # Optional symmetry applier
#     ):
#         """
#         Initializes the strategy with coordinates and shape strategy.

#         Args:
#             coordinate (np.ndarray): Central coordinate.
#             ih (np.ndarray): h indices array.
#             ik (np.ndarray): k indices array.
#             il (np.ndarray): l indices array.
#             shape_strategy (ShapeStrategy): Shape strategy for masking.
#             symmetry_applier (optional): Symmetry applier for point group symmetry.
#         """
#         self.coordinate = coordinate
#         self.ih = ih
#         self.ik = ik
#         self.il = il
#         self.shape_strategy = shape_strategy
#         self.symmetry_applier = symmetry_applier

#     def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
#         """
#         Generates a mask based on coordinates and shape.

#         Args:
#             hkl_mesh (np.ndarray): An array of hkl points.

#         Returns:
#             np.ndarray: A boolean array representing the mask.
#         """
#         # Generate central points
#         acpx = self._find_central_points(self.coordinate[0], self.ih)
#         acpy = self._find_central_points(self.coordinate[1], self.ik)
#         acpz = self._find_central_points(self.coordinate[2], self.il)

#         central_points = self._generate_central_points(acpx, acpy, acpz)

#         # Apply shape mask
#         mask = self.shape_strategy.apply(hkl_mesh, central_points)

#         # Apply symmetry if specified
#         if self.symmetry_applier:
#             symmetry_mask = self.symmetry_applier.apply(hkl_mesh)
#             mask &= symmetry_mask

#         return mask

#     def _find_central_points(self, coordinate: float, indices: np.ndarray) -> np.ndarray:
#         min_index, max_index = indices.min() - 0.5, indices.max() + 0.5
#         # Implement logic to find central points within the indices range
#         return np.array([coordinate])  # Placeholder

#     def _generate_central_points(self, acpx: np.ndarray, acpy: np.ndarray, acpz: np.ndarray) -> np.ndarray:
#         central_points = np.array(np.meshgrid(acpx, acpy, acpz)).T.reshape(-1, 3)
#         return central_points
    
