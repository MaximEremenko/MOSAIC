from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from core.config.values import first_present
from core.scattering.form_factors.neutron_scattering_lengths import rmc_neutron_scl_


_ATOMIC_SYMBOLS = (
    "h", "he",
    "li", "be", "b", "c", "n", "o", "f", "ne",
    "na", "mg", "al", "si", "p", "s", "cl", "ar",
    "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
    "ga", "ge", "as", "se", "br", "kr",
    "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd",
    "in", "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy",
    "ho", "er", "tm", "yb", "lu",
    "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi",
    "po", "at", "rn",
    "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf",
    "es", "fm", "md", "no", "lr",
    "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn", "nh", "fl", "mc",
    "lv", "ts", "og",
)
_ATOMIC_NUMBER_BY_SYMBOL = {symbol: index + 1 for index, symbol in enumerate(_ATOMIC_SYMBOLS)}
_ATOMIC_NUMBER_BY_SYMBOL.update(
    {
        "d": 1,
        "7l": 3,
        "va": 0,
    }
)


def _normalize_element_symbol(element: Any) -> str:
    raw = str(element).strip().lower()
    if raw in _ATOMIC_NUMBER_BY_SYMBOL:
        return raw
    cleaned = re.sub(r"[^a-z0-9]", "", raw)
    if cleaned in _ATOMIC_NUMBER_BY_SYMBOL:
        return cleaned
    return raw


def _build_atomic_number_coefficients(elements) -> np.ndarray:
    values = []
    for element in np.asarray(elements).reshape(-1):
        symbol = _normalize_element_symbol(element)
        if symbol not in _ATOMIC_NUMBER_BY_SYMBOL:
            raise ValueError(
                f"Unknown element symbol {element!r} for coefficient scheme 'atomic_number'."
            )
        values.append(float(_ATOMIC_NUMBER_BY_SYMBOL[symbol]))
    return np.asarray(values, float)


def _build_neutron_coefficients(elements) -> np.ndarray:
    values = []
    for element in np.asarray(elements).reshape(-1):
        symbol = _normalize_element_symbol(element)
        scattering_length, missing = rmc_neutron_scl_(symbol)
        if missing:
            raise ValueError(
                f"Unknown element symbol {element!r} for coefficient scheme 'neutron'."
            )
        values.append(float(scattering_length))
    return np.asarray(values, float)


def _coerce_config_coefficients(coeff_values, *, n_atoms: int) -> np.ndarray:
    arr = np.asarray(coeff_values, dtype=float)
    if arr.ndim == 0:
        return np.full((n_atoms,), float(arr))
    arr = arr.reshape(-1)
    if arr.size != n_atoms:
        raise ValueError(
            f"Configuration coefficients provide {arr.size} values, but "
            f"configuration has {n_atoms} atoms."
        )
    return arr


def _resolve_coefficient_scheme(
    *,
    struct: dict[str, Any],
    coeff_from_cfg,
    logger: logging.Logger,
) -> str:
    scheme = first_present(
        struct,
        ("coeff_scheme", "coeffScheme", "coefficient_scheme", "coeff_source", "coeffSource"),
    )
    if scheme is None:
        return "ones"
    scheme = str(scheme).strip().lower()
    if scheme in {"auto", "atomicnumber", "z", "neutron_scattering_length", "neutron_scattering_lengths"}:
        raise ValueError(
            "Legacy coefficient schemes are no longer supported. "
            "Use one of: ones, atomic_number, neutron, config, file."
        )
    if scheme not in {"ones", "atomic_number", "neutron", "config", "file"}:
        raise ValueError(
            "Unsupported coefficient scheme "
            f"{scheme!r} (use 'ones'|'atomic_number'|'neutron'|'config'|'file')."
        )
    return scheme


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
    coeff_from_cfg = cfg_proc.get_coeff() if hasattr(cfg_proc, "get_coeff") else None
    coeff_scheme = _resolve_coefficient_scheme(
        struct=struct,
        coeff_from_cfg=coeff_from_cfg,
        logger=logger,
    )
    n_atoms = int(len(elements))
    if coeff_scheme == "ones":
        return np.ones((n_atoms,), dtype=float)
    if coeff_scheme == "atomic_number":
        return _build_atomic_number_coefficients(elements)
    if coeff_scheme == "neutron":
        return _build_neutron_coefficients(elements)
    if coeff_scheme == "config":
        if coeff_from_cfg is None:
            raise ValueError(
                "Coefficient scheme 'config' requires coefficient values in the structure/config file."
            )
        return _coerce_config_coefficients(coeff_from_cfg, n_atoms=n_atoms)

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
    if coeff_scheme == "file":
        if coeff_file is None:
            raise ValueError(
                "Coefficient scheme 'file' requires struct_info.coeff_file or structure.coefficients.file."
            )
        coeff_path = Path(coeff_file)
        if not coeff_path.is_absolute():
            coeff_path = Path(working_path) / coeff_path
        return load_coefficients_from_file(
            coeff_path=str(coeff_path),
            dim=int(struct["dimension"]),
            n_atoms=n_atoms,
            supercell=np.asarray(supercell, int),
            vectors=np.asarray(vectors, float),
            cells_origin=np.asarray(cells_origin.to_numpy(), float),
            logger=logger,
        )

    raise AssertionError(f"Unhandled coefficient scheme: {coeff_scheme}")


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
