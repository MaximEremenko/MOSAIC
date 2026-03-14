from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.config.values import first_present
from core.scattering.form_factors.neutron_scattering_lengths import rmc_neutron_scl_


def _invert_2x2(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 matrix, got shape {matrix.shape}.")

    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])
    det = (a * d) - (b * c)
    if abs(det) <= np.finfo(float).eps:
        raise ValueError("vectors must be invertible for 2D coefficient mapping.")

    return np.array([[d, -b], [-c, a]], dtype=float) / det


def resolve_structure_coefficients(
    *,
    struct: dict[str, Any],
    cfg_proc,
    working_path: str,
    elements,
    supercell,
    vectors,
    cells_origin,
    logger: logging.Logger,
):
    coeff = elements.apply(lambda element: rmc_neutron_scl_(element)[0])
    coeff_from_cfg = cfg_proc.get_coeff() if hasattr(cfg_proc, "get_coeff") else None
    coeff_source = first_present(struct, ("coeff_source", "coeffSource")) or "auto"
    coeff_source = str(coeff_source).strip().lower()
    if coeff_source not in {"auto", "config", "file"}:
        raise ValueError(
            f"Unsupported coeff_source={coeff_source!r} (use 'auto'|'config'|'file')."
        )
    if coeff_source in {"auto", "config"} and coeff_from_cfg is not None:
        coeff = coeff_from_cfg

    coeff_file = first_present(
        struct,
        (
            "coeff_file",
            "coeff_filename",
            "coeff_path",
            "coefficients_file",
            "intensity_coeff_file",
            "intensity_coeff_filename",
        ),
    )
    if coeff_file is not None and coeff_source in {"auto", "file"} and (
        coeff_from_cfg is None or coeff_source == "file"
    ):
        coeff_path = Path(coeff_file)
        if not coeff_path.is_absolute():
            coeff_path = Path(working_path) / coeff_path
        coeff = load_coefficients_from_file(
            coeff_path=str(coeff_path),
            dim=int(struct["dimension"]),
            n_atoms=int(len(elements)),
            supercell=np.asarray(supercell, int),
            vectors=np.asarray(vectors, float),
            cells_origin=np.asarray(cells_origin.to_numpy(), float),
            logger=logger,
        )

    return coeff


def load_coefficients_from_file(
    *,
    coeff_path: str,
    dim: int,
    n_atoms: int,
    supercell: np.ndarray,
    vectors: np.ndarray,
    cells_origin: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    arr = np.asarray(np.loadtxt(coeff_path, dtype=float), dtype=float)

    if arr.ndim == 0:
        return np.full((n_atoms,), float(arr))
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim == 1:
        if arr.size != n_atoms:
            raise ValueError(
                f"Coefficient file '{coeff_path}' has {arr.size} values, "
                f"but configuration has {n_atoms} atoms."
            )
        return arr
    if arr.ndim != 2:
        raise ValueError(
            f"Coefficient file '{coeff_path}' must be 1D or 2D, got shape {arr.shape}."
        )
    if dim != 2:
        raise ValueError(
            f"2D coefficient-matrix mapping is only supported for dim=2; got dim={dim}."
        )

    nx, ny = int(supercell[0]), int(supercell[1])
    if arr.shape not in {(ny, nx), (nx, ny)}:
        raise ValueError(
            f"Coefficient matrix shape {arr.shape} does not match supercell "
            f"(ny,nx)=({ny},{nx}) or (nx,ny)=({nx},{ny})."
        )

    vectors_arr = np.asarray(vectors, float)
    cells_origin_arr = np.asarray(cells_origin, float)
    if cells_origin_arr.ndim != 2 or cells_origin_arr.shape[1] != 2:
        raise ValueError(
            "cells_origin must be a 2D array with shape (n_atoms, 2) for 2D mapping."
        )

    inv_vectors = _invert_2x2(vectors_arr)
    x = np.asarray(cells_origin_arr[:, 0], float)
    y = np.asarray(cells_origin_arr[:, 1], float)
    fx = np.asarray(x * inv_vectors[0, 0] + y * inv_vectors[1, 0], float)
    fy = np.asarray(x * inv_vectors[0, 1] + y * inv_vectors[1, 1], float)

    ix = np.clip(np.round(fx * nx).astype(int), 0, nx - 1)
    iy = np.clip(np.round(fy * ny).astype(int), 0, ny - 1)

    if arr.shape == (ny, nx):
        coeff = arr[iy, ix]
    else:
        coeff = arr[ix, iy]

    logger.info(
        "Loaded coefficient matrix '%s' (shape %s) mapped onto %d atoms.",
        coeff_path,
        tuple(arr.shape),
        n_atoms,
    )
    return np.asarray(coeff, float)
