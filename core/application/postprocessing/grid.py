from __future__ import annotations

import numpy as np
from scipy.signal import convolve

from core.application.postprocessing.features import qspace_psf_in_r


def axes_from_coords(coords):
    coords = np.asarray(coords, float)
    dim = coords.shape[1]
    axes_vals = [np.sort(np.unique(coords[:, axis])) for axis in range(dim)]
    shape = tuple(int(values.size) for values in axes_vals)
    return shape, axes_vals


def indices_from_axes(coords, axes_vals):
    coords = np.asarray(coords, float)
    dim = coords.shape[1]
    idxs = []
    for axis in range(dim):
        axis_values = axes_vals[axis]
        if len(axis_values) == 1:
            idxs.append(np.zeros(coords.shape[0], dtype=np.int64))
            continue
        index = np.searchsorted(axis_values, coords[:, axis])
        index = np.clip(index, 1, len(axis_values) - 1)
        left = np.abs(coords[:, axis] - axis_values[index - 1])
        right = np.abs(coords[:, axis] - axis_values[index])
        index = np.where(right < left, index, index - 1).astype(np.int64)
        idxs.append(index)
    return tuple(idxs)


def regrid_patch_to_c(coords, vals):
    shape, axes_vals = axes_from_coords(coords)
    flat = np.ravel_multi_index(indices_from_axes(coords, axes_vals), shape, order="C")
    y_flat = np.empty(flat.size, float)
    y_flat[flat] = np.asarray(vals, float)
    return y_flat.reshape(shape, order="C"), shape, axes_vals, y_flat


def fractional_center_index(center_val, axis_vals):
    axis_values = np.asarray(axis_vals, float)
    if axis_values.size == 1:
        return 0.0
    j = int(np.clip(np.searchsorted(axis_values, float(center_val)), 1, axis_values.size - 1))
    j0 = j - 1
    j1 = j
    dv = axis_values[j1] - axis_values[j0]
    t = 0.0 if abs(dv) < 1e-30 else (float(center_val) - axis_values[j0]) / dv
    return j0 + t


def fourier_shift_nd(arr, shift):
    F = np.fft.fftn(arr)
    for axis, sh in enumerate(shift):
        n = arr.shape[axis]
        phase = np.exp(-2j * np.pi * sh * np.fft.fftfreq(n))
        shape = [1] * arr.ndim
        shape[axis] = n
        F *= phase.reshape(shape)
    return np.fft.ifftn(F).real


def center_patch_subvoxel(y_grid, axes_vals, center_abs):
    mid = tuple((n - 1) / 2.0 for n in y_grid.shape)
    i_star = [
        fractional_center_index(center_abs[axis], axes_vals[axis])
        for axis in range(y_grid.ndim)
    ]
    delta = tuple(mid[axis] - i_star[axis] for axis in range(y_grid.ndim))
    y_c = fourier_shift_nd(y_grid, delta)
    slices = [slice(None) if n % 2 else slice(0, n - 1) for n in y_c.shape]
    return y_c[tuple(slices)], None


def infer_shape_from_coords(coords):
    coords = np.asarray(coords, float)
    dim = coords.shape[1]
    return tuple(int(np.unique(coords[:, axis]).size) for axis in range(dim))


def apply_rq_pipeline_local(
    Rvals: np.ndarray,
    coords: np.ndarray,
    *,
    q_window_kind: str,
    q_window_at_db: float,
    size_aver: np.ndarray,
    hkl_max_xyz: tuple[float, float, float],
    guard_frac: float,
) -> np.ndarray:
    shape = infer_shape_from_coords(coords)
    dim = len(shape)
    Rvals = Rvals.real
    size_aver = np.asarray(size_aver, dtype=int).ravel()
    if size_aver.size < dim:
        raise ValueError(
            f"size_aver has {size_aver.size} dims, but local coords imply dim={dim}"
        )
    q_psf = qspace_psf_in_r(
        size_aver=size_aver[:dim],
        hkl_max_xyz=hkl_max_xyz,
        guard_frac=guard_frac,
        window_kind=q_window_kind,
        window_at_db=q_window_at_db,
    )
    if dim == 1:
        return convolve(Rvals, q_psf, mode="same")
    return convolve(Rvals.reshape(shape), q_psf, mode="same").ravel()


def axis_max_from_item(item, axis):
    rng_key = f"{axis}_range"
    s_key = f"{axis}_start"
    e_key = f"{axis}_end"
    vals = []
    if rng_key in item and item[rng_key] is not None:
        try:
            start, end = item[rng_key]
            vals.extend([start, end])
        except Exception:
            pass
    if s_key in item:
        vals.append(item[s_key])
    if e_key in item:
        vals.append(item[e_key])
    if not vals:
        return 0.0
    vals = [float(v) for v in vals]
    return float(np.max(np.abs(vals)))


def compute_hkl_max_from_intervals(intervals):
    if not intervals:
        return (0.0, 0.0, 0.0)
    h_max = k_max = l_max = 0.0
    for item in intervals:
        h_max = max(h_max, axis_max_from_item(item, "h"))
        k_max = max(k_max, axis_max_from_item(item, "k"))
        l_max = max(l_max, axis_max_from_item(item, "l"))
    return (h_max, k_max, l_max)
