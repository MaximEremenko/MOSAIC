from __future__ import annotations

from typing import Any

import numpy as np


HALF_SPACE_ROLE_FULL = "full"
HALF_SPACE_ROLE_ZERO_PLANE = "zero_plane"
HALF_SPACE_ROLE_POSITIVE_HALF = "positive_half"
HALF_SPACE_ROLE_LEGACY = "legacy"

_VALID_HALF_SPACE_ROLES = {
    HALF_SPACE_ROLE_FULL,
    HALF_SPACE_ROLE_ZERO_PLANE,
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_LEGACY,
}
_L_INDEX_ATOL = 1e-7


def normalize_half_space_role(role: object | None) -> str:
    if role is None:
        return HALF_SPACE_ROLE_LEGACY
    if isinstance(role, np.ndarray):
        if role.shape == ():
            role = role.item()
        else:
            role = role.ravel()[0].item()
    if isinstance(role, bytes):
        role = role.decode("utf-8")
    normalized = str(role)
    if normalized not in _VALID_HALF_SPACE_ROLES:
        raise ValueError(f"Unknown reciprocal half-space role: {normalized!r}")
    return normalized


def half_space_role_multiplicity(role: object | None) -> int | None:
    normalized = normalize_half_space_role(role)
    if normalized == HALF_SPACE_ROLE_LEGACY:
        return None
    if normalized == HALF_SPACE_ROLE_POSITIVE_HALF:
        return 2
    return 1


def _axis_grid_index_bounds(start: float, end: float, axis_size: int) -> tuple[int, int]:
    start_scaled = float(start) * int(axis_size)
    end_scaled = float(end) * int(axis_size)
    return (
        int(np.ceil(start_scaled - _L_INDEX_ATOL)),
        int(np.floor(end_scaled + _L_INDEX_ATOL)),
    )


def classify_interval_half_space_role(
    interval: dict[str, Any],
    supercell: np.ndarray,
) -> str:
    supercell_arr = np.asarray(supercell, dtype=float)
    if supercell_arr.size <= 2:
        return HALF_SPACE_ROLE_FULL

    l_start: float
    l_end: float
    if "l_range" in interval:
        l_start, l_end = (float(value) for value in interval["l_range"])
    else:
        l_start = float(interval.get("l_start", 0.0))
        l_end = float(interval.get("l_end", 0.0))

    l_axis_size = int(np.rint(supercell_arr[2]))
    if l_axis_size <= 0:
        raise ValueError(f"Invalid L supercell size: {supercell_arr[2]!r}")

    start_index, end_index = _axis_grid_index_bounds(l_start, l_end, l_axis_size)
    if end_index < start_index:
        raise ValueError(
            "Cannot classify reciprocal half-space role for an interval with no L grid points."
        )
    if start_index == 0 and end_index == 0:
        return HALF_SPACE_ROLE_ZERO_PLANE
    if start_index > 0:
        return HALF_SPACE_ROLE_POSITIVE_HALF
    if end_index < 0:
        raise ValueError(
            "Negative-L intervals are not valid for the optimized half-space pipeline."
        )
    raise ValueError(
        "Cannot classify reciprocal half-space role for an interval that mixes "
        "L=0 and nonzero-L points."
    )


def half_space_conjugate_reconstruction_required(
    q_grid: np.ndarray,
    half_space_role: object | None = None,
) -> bool:
    role = normalize_half_space_role(half_space_role)
    if role == HALF_SPACE_ROLE_POSITIVE_HALF:
        return True
    if role in {HALF_SPACE_ROLE_FULL, HALF_SPACE_ROLE_ZERO_PLANE}:
        return False

    q_arr = np.asarray(q_grid)
    if q_arr.ndim != 2 or q_arr.shape[1] <= 2 or q_arr.shape[0] == 0:
        return False
    l_values = q_arr[:, 2]
    nonzero_l = np.abs(l_values) > _L_INDEX_ATOL
    if not bool(np.any(nonzero_l)):
        return False
    if not bool(np.all(nonzero_l)):
        raise ValueError(
            "Cannot apply half-space conjugate reconstruction to a legacy interval "
            "that mixes L=0 and nonzero-L points."
        )
    if bool(np.any(l_values > _L_INDEX_ATOL)) and bool(np.any(l_values < -_L_INDEX_ATOL)):
        raise ValueError(
            "Cannot apply half-space conjugate reconstruction to a legacy interval "
            "that contains both positive- and negative-L points."
        )
    return True


def apply_half_space_conjugate_reconstruction(
    amplitude_values: np.ndarray,
    q_grid: np.ndarray,
    half_space_role: object | None = None,
) -> np.ndarray:
    values = np.asarray(amplitude_values)
    if half_space_conjugate_reconstruction_required(q_grid, half_space_role):
        return values + np.conj(values)
    return values


__all__ = [
    "HALF_SPACE_ROLE_FULL",
    "HALF_SPACE_ROLE_LEGACY",
    "HALF_SPACE_ROLE_POSITIVE_HALF",
    "HALF_SPACE_ROLE_ZERO_PLANE",
    "apply_half_space_conjugate_reconstruction",
    "classify_interval_half_space_role",
    "half_space_conjugate_reconstruction_required",
    "half_space_role_multiplicity",
    "normalize_half_space_role",
]
