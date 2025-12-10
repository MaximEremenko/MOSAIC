# -*- coding: utf-8 -*-
"""
processors/point_data_postprocessing_processor.py

Single global linear decoder M that maps local RIFFT residual patches
to site-resolved displacements:

    u_m = M r_m ,

where r_m is a feature vector built linearly from the masked RIFFT
residual R(r) around site m.

M is determined once, inside this module, from the known ground-truth
displacements (original_coords - average_coords) and the corresponding
RIFFT patches for the current Q-subset. After that, the same M is used
for all chunks and all sites within this run.

All operations from A(Q) → R(r) → r_m → u_m are linear in the masked
amplitude, so additivity over disjoint masks holds:
    u_total = u_set1 + u_set2 (up to numerical noise).

Author: Maksim Eremenko (M-decoder variant)
"""

import os
import csv
import logging
from collections import defaultdict

import numpy as np
from scipy.signal.windows import chebwin
from scipy.signal import convolve
from scipy.fft import fft, fftshift
import json
import hashlib
from numba import set_num_threads
set_num_threads(32)

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_scalar_from_store(rifft_saver, filename, key_candidates):
    """
    Load a scalar integer (ntotal, etc.) from a HDF5 file if present.
    """
    try:
        d = rifft_saver.load_data(filename)
    except FileNotFoundError:
        return None
    for k in key_candidates:
        if k in d:
            v = np.array(d[k])
            try:
                return int(np.ravel(v)[0])
            except Exception:
                pass
    return None


def _normalize_amplitudes_ntotal(amplitudes, *, rifft_saver, chunk_id,
                                 logger=None):
    """
    Scale amplitudes by 1/ntotal only (total number of reciprocal-space
    sampling points on the *unmasked* grid). This keeps a single global
    scale factor across different masks and runs.
    """
    fn_tot = rifft_saver.generate_filename(
        chunk_id,
        suffix="_amplitudes_ntotal_reciprocal_space_points",
    )
    ntot = _load_scalar_from_store(
        rifft_saver,
        fn_tot,
        ["ntotal_reciprocal_space_points", "ntotal_reciprocal_points"],
    )
    if not ntot or ntot <= 0:
        if logger:
            logger.warning(
                "[normalize_amplitudes_ntotal] Missing ntotal; "
                "leaving amplitudes unscaled."
            )
        return amplitudes

    scale = 1.0 / float(ntot)
    try:
        if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
            amplitudes[:, 1] *= scale
        else:
            amplitudes[...] *= scale
    except Exception as e:
        if logger:
            logger.warning(
                "[normalize_amplitudes_ntotal] scaling failed: %s", e
            )
    return amplitudes


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate/grid helpers (Cartesian)
# ──────────────────────────────────────────────────────────────────────────────

def _axes_from_coords(coords):
    coords = np.asarray(coords, float)
    D = coords.shape[1]
    axes_vals = [np.sort(np.unique(coords[:, a])) for a in range(D)]
    shape = tuple(int(v.size) for v in axes_vals)
    return shape, axes_vals


def _indices_from_axes(coords, axes_vals):
    coords = np.asarray(coords, float)
    D = coords.shape[1]
    idxs = []
    for a in range(D):
        av = axes_vals[a]
        i = np.searchsorted(av, coords[:, a])
        i = np.clip(i, 1, len(av) - 1)
        left = np.abs(coords[:, a] - av[i - 1])
        right = np.abs(coords[:, a] - av[i])
        i = np.where(right < left, i, i - 1).astype(np.int64)
        idxs.append(i)
    return tuple(idxs)


def _regrid_patch_to_c(coords, vals):
    """
    Put scattered coordinates+values onto a dense C-order grid defined by
    the unique sorted coords along each axis.
    """
    shape, axes_vals = _axes_from_coords(coords)
    flat = np.ravel_multi_index(
        _indices_from_axes(coords, axes_vals), shape, order="C"
    )
    y_flat = np.empty(flat.size, float)
    y_flat[flat] = np.asarray(vals, float)
    return y_flat.reshape(shape, order="C"), shape, axes_vals, y_flat


def _fractional_center_index(center_val, axis_vals):
    av = np.asarray(axis_vals, float)
    j = int(np.clip(np.searchsorted(av, float(center_val)), 1, av.size - 1))
    j0 = j - 1
    j1 = j
    dv = av[j1] - av[j0]
    t = 0.0 if abs(dv) < 1e-30 else (float(center_val) - av[j0]) / dv
    return j0 + t


def _fourier_shift_nd(arr, shift):
    """
    Sub-voxel shift via Fourier phase multiplication (linear in arr).
    """
    F = np.fft.fftn(arr)
    for ax, sh in enumerate(shift):
        n = arr.shape[ax]
        phase = np.exp(-2j * np.pi * sh * np.fft.fftfreq(n))
        shape = [1] * arr.ndim
        shape[ax] = n
        F *= phase.reshape(shape)
    return np.fft.ifftn(F).real


def _center_patch_subvoxel(y_grid, axes_vals, center_abs):
    """
    Center the patch y_grid so that center_abs (in Å) sits exactly in the
    middle voxel (sub-voxel accurate). Returns a grid with *odd* sizes.
    """
    mid = tuple((n - 1) / 2.0 for n in y_grid.shape)
    i_star = [
        _fractional_center_index(center_abs[a], axes_vals[a])
        for a in range(y_grid.ndim)
    ]
    delta = tuple(mid[a] - i_star[a] for a in range(y_grid.ndim))
    y_c = _fourier_shift_nd(y_grid, delta)

    # Ensure odd shape by dropping one cell at positive edge if needed
    sl = [slice(None) if n % 2 else slice(0, n - 1) for n in y_c.shape]
    return y_c[tuple(sl)], None


def _radial_gaussian_weights_cart(shape, steps_cart, gamma=0.35):
    """
    Gaussian radial weight in Cartesian coordinates (Å). Used both as
    LS weight and feature preconditioner.
    """
    coords_1d = [
        (np.arange(n) - (n - 1) / 2.0) * float(steps_cart[a])
        for a, n in enumerate(shape)
    ]
    if len(shape) == 1:
        r2 = coords_1d[0] ** 2
    elif len(shape) == 2:
        x, y = np.meshgrid(coords_1d[0], coords_1d[1], indexing="ij")
        r2 = x * x + y * y
    else:
        x, y, z = np.meshgrid(
            coords_1d[0], coords_1d[1], coords_1d[2], indexing="ij"
        )
        r2 = x * x + y * y + z * z

    edge = np.array(
        [max(abs(s[0]), abs(s[-1])) for s in coords_1d], float
    )
    r_edge = float(np.linalg.norm(edge))
    sigma = max(gamma * r_edge, 1e-12)
    return np.exp(-0.5 * (r2 / (sigma * sigma)))


# ──────────────────────────────────────────────────────────────────────────────
# Q-space PSF (kept, even if currently pass-through in R)
# ──────────────────────────────────────────────────────────────────────────────

def _kband_vector_unshifted(nS, S, hmax, guard_frac):
    nS = int(nS)
    S = int(S)
    if nS <= 0 or S <= 0:
        raise ValueError("nS and S must be positive")

    h = np.fft.fftfreq(nS, d=1.0 / S)  # DC at index 0
    hnyq = S / 2.0

    if hmax is None or hmax <= 0.0:
        return np.ones(nS, dtype=float)

    hmax_eff = min(float(hmax), hnyq - 1e-9)
    if hmax_eff <= 0.0:
        v = np.zeros(nS, dtype=float)
        v[0] = 1.0
        return v

    hedge = (1.0 - float(guard_frac)) * hmax_eff
    hedge = max(0.0, min(hedge, hmax_eff))

    v = np.zeros(nS, dtype=float)
    ah = np.abs(h)
    v[ah <= hedge] = 1.0
    mask = (ah > hedge) & (ah < hmax_eff)
    if np.any(mask):
        t = (ah[mask] - hedge) / (hmax_eff - hedge + 1e-12)
        v[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
    return v


def _window_spectrum_vectors_unshifted(size_aver, kind="cheb", at_db=100.0):
    size_aver = np.asarray(size_aver, dtype=int)
    wd = []
    for ax in range(len(size_aver)):
        S = int(size_aver[ax])
        if kind == "cheb":
            w = chebwin(S, at_db, sym=False).astype(float)
        else:
            raise ValueError("kind must be 'cheb'")
        Fw = fft(w, S) / (len(w) / 2.0)
        mag_unshift = np.fft.ifftshift(
            np.abs(fftshift(Fw / (np.abs(Fw).max() or 1.0)))
        )
        wd.append(mag_unshift)
    return wd


def _qspace_psf_in_r(size_aver, hkl_max_xyz, *,
                     guard_frac, window_kind="cheb", window_at_db=100.0):
    """
    PSF in R corresponding to |FFT(window)| × k-band mask. Returned
    kernel is real, fftshifted, sum-normalised.
    """
    size_aver = np.asarray(size_aver, dtype=int)
    dim = len(size_aver)

    wd = _window_spectrum_vectors_unshifted(
        size_aver, kind=window_kind, at_db=window_at_db
    )

    h_max = float(hkl_max_xyz[0]) if dim >= 1 else 0.0
    k_max = float(hkl_max_xyz[1]) if dim >= 2 else 0.0
    l_max = float(hkl_max_xyz[2]) if dim >= 3 else 0.0

    bx = _kband_vector_unshifted(size_aver[0], size_aver[0], h_max, guard_frac)
    if dim >= 2:
        by = _kband_vector_unshifted(size_aver[1], size_aver[1], k_max, guard_frac)
    if dim == 3:
        bz = _kband_vector_unshifted(size_aver[2], size_aver[2], l_max, guard_frac)

    if dim == 1:
        B = wd[0] * bx
    elif dim == 2:
        B = (wd[0] * bx)[:, None] * (wd[1] * by)[None, :]
    else:
        B = (
            (wd[0] * bx)[:, None, None]
            * (wd[1] * by)[None, :, None]
            * (wd[2] * bz)[None, None, :]
        )

    dc = float(B.flat[0]) if B.size else 1.0
    if dc != 0.0:
        B = B / dc

    psf = np.fft.ifftn(B).real
    psf = np.fft.fftshift(psf)

    s = psf.sum()
    if s != 0.0:
        psf = psf / s
    else:
        psf *= 0.0
        center = tuple(n // 2 for n in psf.shape)
        psf[center] = 1.0
    return psf


def _infer_shape_from_coords(coords):
    coords = np.asarray(coords, float)
    D = coords.shape[1]
    return tuple(int(np.unique(coords[:, a]).size) for a in range(D))


# def _apply_rq_pipeline_local(
#     Rvals, coords, *,
#     q_window_kind, q_window_at_db,
#     size_aver, hkl_max_xyz, guard_frac,
# ):
#     """
#     Per-site *Q-only* processing for the linear path.

#     Currently returns the raw residual Rvals (already RIFFT of the chosen
#     ΔA(Q) mask). The PSF call is left in-place for future experimentation,
#     but commented out in the return.
#     """
#     shape = _infer_shape_from_coords(coords)
#     dim = len(shape)
#     Rvals = np.asarray(Rvals, float).real

#     # PSF in R (not used yet, but linear if enabled)
#     _ = _qspace_psf_in_r(
#         size_aver=np.asarray(size_aver, dtype=int),
#         hkl_max_xyz=hkl_max_xyz,
#         guard_frac=guard_frac,
#         window_kind=q_window_kind,
#         window_at_db=q_window_at_db,
#     )
#     # Example if you ever want to enable this:
#     if dim == 1:
#         r_after = convolve(Rvals, q_psf, mode="same")
#     else:
#         r_after = convolve(Rvals.reshape(shape), q_psf, mode="same").ravel()
#     return r_after

#    return Rvals  # current behaviour: no extra Q-stage smoothing
def _apply_rq_pipeline_local(
    Rvals: np.ndarray,
    coords: np.ndarray,
    *,
    q_window_kind: str,
    q_window_at_db: float,
    size_aver: np.ndarray,
    hkl_max_xyz: tuple[float, float, float],
    guard_frac: float,
) -> np.ndarray:
    """Per-site *Q-only* processing for the linear path (r_stage is NONE)."""
    shape = _infer_shape_from_coords(coords)
    dim   = len(shape)
    Rvals  = Rvals.real
    # Q-stage: PSF in R (sum-normalized, DC-preserving) — linear
    q_psf = _qspace_psf_in_r(
        size_aver=np.asarray(size_aver, dtype=int),
        hkl_max_xyz=hkl_max_xyz,
        guard_frac=guard_frac,
        window_kind=q_window_kind,
        window_at_db=q_window_at_db,
    )
    if dim == 1:
        r_after_Q = convolve(Rvals, q_psf, mode="same")
    else:
        r_after_Q = convolve(Rvals.reshape(shape), q_psf, mode="same").ravel()
    #coeff = np.sum(np.abs(r_after_Q.real))/np.sum(np.abs(Rvals.real))
    return r_after_Q

# ──────────────────────────────────────────────────────────────────────────────
# Misc: hkl_max extraction & CSV writer
# ──────────────────────────────────────────────────────────────────────────────

def _axis_max_from_item(item, axis):
    rng_key = f"{axis}_range"
    s_key = f"{axis}_start"
    e_key = f"{axis}_end"
    vals = []
    if rng_key in item and item[rng_key] is not None:
        try:
            a, b = item[rng_key]
            vals.extend([a, b])
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
    for it in intervals:
        h_max = max(h_max, _axis_max_from_item(it, "h"))
        k_max = max(k_max, _axis_max_from_item(it, "k"))
        l_max = max(l_max, _axis_max_from_item(it, "l"))
    return (h_max, k_max, l_max)


def _write_displacements_csv(csv_path, ids, U):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["central_point_id", "ux", "uy", "uz", "units=cartesian"])
        for i, (ux, uy, uz) in zip(ids.tolist(), U.tolist()):
            w.writerow([i, f"{ux:.6E}", f"{uy:.6E}", f"{uz:.6E}", "cartesian"])


# ──────────────────────────────────────────────────────────────────────────────
# Feature builder for the M-decoder
# ──────────────────────────────────────────────────────────────────────────────

def _build_feature_vector_from_patch(
    y_grid,
    axes_vals,
    center_abs,
    *,
    D,
    weight_gamma=0.35,
    remove_odd_tilt=True,
):
    """
    Convert a local RIFFT residual patch into a feature vector r.

    Steps (all linear in y_grid):
      1. sub-voxel centering,
      2. odd (dipole) component,
      3. optional odd-tilt removal on outer ring,
      4. radial sqrt(W) weighting,
      5. flatten (C-order).
    """
    # 1) center
    y_c, _ = _center_patch_subvoxel(y_grid, axes_vals, center_abs[:D])
    flip_axes = tuple(range(D))

    # 2) odd component
    O = 0.5 * (y_c - np.flip(y_c, axis=flip_axes))

    # geometry
    steps = tuple(
        float(np.mean(np.diff(axes_vals[a]))) if len(axes_vals[a]) > 1 else 1.0
        for a in range(D)
    )
    coords_1d = [
        (np.arange(n) - (n - 1) / 2.0) * float(steps[a])
        for a, n in enumerate(y_c.shape)
    ]

    if D == 1:
        s0 = coords_1d[0]
        r2 = s0 ** 2
    elif D == 2:
        s0, s1 = np.meshgrid(coords_1d[0], coords_1d[1], indexing="ij")
        r2 = s0 * s0 + s1 * s1
    else:
        s0, s1, s2 = np.meshgrid(
            coords_1d[0], coords_1d[1], coords_1d[2], indexing="ij"
        )
        r2 = s0 * s0 + s1 * s1 + s2 * s2

    # 3) optional odd "tilt" removal on outer ring (still linear in y_grid)
    if remove_odd_tilt:
        r = np.sqrt(r2)
        r_max = float(np.max(r)) if r.size else 1.0
        mask = (r > 0.55 * r_max).ravel("C")

        N = O.size
        if D == 1:
            Sodd = np.vstack([s0.ravel("C"), np.zeros(N), np.zeros(N)])
        elif D == 2:
            Sodd = np.vstack([s0.ravel("C"), s1.ravel("C"), np.zeros(N)])
        else:
            Sodd = np.vstack(
                [s0.ravel("C"), s1.ravel("C"), s2.ravel("C")]
            )

        Ao = Sodd[:, mask]
        rhs = O.ravel("C")[mask]
        Ho = Ao @ Ao.T + 1e-12 * np.eye(3)
        fo = Ao @ rhs
        b_hat = np.linalg.solve(Ho, fo)

        O = O - (
            b_hat[0] * s0
            + (b_hat[1] * s1 if D >= 2 else 0.0)
            + (b_hat[2] * s2 if D == 3 else 0.0)
        )

    # 4) radial weights (sqrt to keep scale moderate)
    W = _radial_gaussian_weights_cart(y_c.shape, steps, gamma=weight_gamma)
    feat = (np.sqrt(W) * O).ravel("C")
    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Main processor: build/learn/apply M
# ──────────────────────────────────────────────────────────────────────────────

class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        self.parameters = dict(parameters or {})

        # fixed linear path
        self.parameters.setdefault("normalize_amplitudes_by", "ntotal")
        self.parameters.setdefault("coords_are_fractional", False)
        self.parameters.setdefault("ls_weight_gamma", 0.35)
        self.parameters.setdefault("dog_lambda_reg", 1e-3)
        self.parameters.setdefault("linear_max_training_samples", None)

        # Q-stage knobs
        self.parameters.setdefault("q_window_kind", "cheb")
        self.parameters.setdefault("q_window_at_db", 100.0)
        self.parameters.setdefault("edge_guard_frac", 0.10)

        # geometry for ground-truth displacements
        if "original_coords" not in self.parameters:
            raise KeyError(
                "parameters['original_coords'] is required for M-decoder."
            )
        if "average_coords" not in self.parameters:
            raise KeyError(
                "parameters['average_coords'] is required for M-decoder. "
                "Add avg_coords.to_numpy() to params in main.py."
            )

        self.original_coords = np.asarray(
            self.parameters["original_coords"], float
        )
        self.average_coords = np.asarray(
            self.parameters["average_coords"], float
        )
 # NEW: optional precomputed displacements from config (Cartesian Å)
        self.u_true_all = None
        if "displacements_from_config" in self.parameters:
            self.u_true_all = np.asarray(
                self.parameters["displacements_from_config"], float
            )
        # decoder cache
        self._decoder_M = None  # shape (3, P)
        self._feature_dim = None

    # ------------------------------------------------------------------ public API

    def process_chunk(self, chunk_id, rifft_saver, client, output_dir):
        point_data_list = self.db_manager.get_point_data_for_chunk(chunk_id)
        if not point_data_list:
            print(f"No point data found for chunk_id: {chunk_id}")
            return None
        return self.compute_and_save_displacements(
            chunk_id=chunk_id,
            rifft_saver=rifft_saver,
            point_data_list=point_data_list,
            output_dir=output_dir,
            broadcast_into_rows=self.parameters.get(
                "broadcast_displacement_into_rows", False
            ),
        )
    def _get_decoder_cache_path(self, output_dir: str) -> str:
            """
            Build a cache filename for the decoder M that is tied to the
            current Q-subset / supercell / linear knobs. This avoids mixing
            decoders between incompatible runs.
            """
            def _to_plain(x):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, np.generic):
                    return x.item()
                if isinstance(x, (list, tuple)):
                    return [_to_plain(v) for v in x]
                if isinstance(x, dict):
                    return {k: _to_plain(v) for k, v in x.items()}
                return x
    
            intervals = self.parameters.get("reciprocal_space_intervals_all", [])
            key_obj = {
                "supercell": _to_plain(np.asarray(self.parameters["supercell"], int)),
                "intervals": _to_plain(intervals),
                "q_window_kind": self.parameters.get("q_window_kind", "cheb"),
                "q_window_at_db": float(self.parameters.get("q_window_at_db", 100.0)),
                "edge_guard_frac": float(self.parameters.get("edge_guard_frac", 0.10)),
                "ls_weight_gamma": float(self.parameters.get("ls_weight_gamma", 0.35)),
                "dog_lambda_reg": float(self.parameters.get("dog_lambda_reg", 1e-3)),
            }
    
            key_json = json.dumps(key_obj, sort_keys=True)
            h = hashlib.sha1(key_json.encode("utf-8")).hexdigest()[:16]
            fname = f"decoder_M_{h}.npz"
            return os.path.join(output_dir, fname)
    # ------------------------------------------------------------------ amplitude+grid loader

    def load_amplitudes_and_generate_grid(
        self, chunk_id, point_data_list, rifft_saver
    ):
        """
        Fallback when rifft_space_grid is not stored: regenerate the local
        grids using PointDataProcessor._generate_grid (as in the original
        implementation) and normalise amplitudes by ntotal.
        """
        filename = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            data = rifft_saver.load_data(filename)
            amplitudes = data.get("amplitudes", None)
            if amplitudes is None:
                print(f"Amplitudes not found in {filename}")
                return np.array([]), None, None

            amplitudes = _normalize_amplitudes_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=logging.getLogger(__name__),
            )

            grids = []
            grids_shapeNd = []
            central_point_ids = []
            for pd in point_data_list:
                grid_points, grid_shapeNd = self.point_data_processor._generate_grid(
                    chunk_id=chunk_id,
                    dimensionality=len(pd["coordinates"]),
                    step_in_frac=pd["step_in_frac"],
                    central_point=pd["coordinates"],
                    dist=pd["dist_from_atom_center"],
                    central_point_id=pd["central_point_id"],
                )
                grids.append(grid_points)
                grids_shapeNd.append(grid_shapeNd)
                central_point_ids.extend(
                    [pd["central_point_id"]] * len(grid_points)
                )

            rifft_space_grid = (
                np.hstack(
                    (np.vstack(grids), np.array(central_point_ids)[:, None])
                )
                if grids
                else np.array([])
            )
            return rifft_space_grid, amplitudes, grids_shapeNd
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return np.array([]), None, None

    # ------------------------------------------------------------------ main compute
    def compute_and_save_displacements(
        self,
        *,
        chunk_id,
        rifft_saver,
        point_data_list,
        output_dir=None,
        broadcast_into_rows=False,
    ):
        log = logging.getLogger(__name__)

        if output_dir is None:
            out_by_saver = getattr(rifft_saver, "output_dir", None)
            output_dir = out_by_saver or os.path.dirname(
                os.path.abspath(
                    rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
                )
            )
        os.makedirs(output_dir, exist_ok=True)

        # ---- load amplitudes + rifft grid ------------------------------------
        fn_amp = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            d = rifft_saver.load_data(fn_amp)
            amplitudes = d.get("amplitudes", None)
            rifft_space_grid = d.get("rifft_space_grid", None)
        except FileNotFoundError:
            d = {}
            amplitudes = None
            rifft_space_grid = None

        if (
            amplitudes is None
            or rifft_space_grid is None
            or len(rifft_space_grid) == 0
        ):
            rifft_space_grid2, amplitudes2, _ = self.load_amplitudes_and_generate_grid(
                chunk_id, point_data_list, rifft_saver
            )
            if amplitudes is None:
                amplitudes = amplitudes2
            if rifft_space_grid is None:
                rifft_space_grid = rifft_space_grid2
            if (
                amplitudes is None
                or rifft_space_grid is None
                or len(rifft_space_grid) == 0
            ):
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")

            # Enforce ntotal-only normalisation once more for safety
            amplitudes = _normalize_amplitudes_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=log,
            )

        # residual ΔR column (already RIFFT of masked ΔA(Q))
        if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
            Rvals_all = amplitudes[:, 1]
        else:
            Rvals_all = np.ravel(amplitudes)

        rifft_space_grid = np.asarray(rifft_space_grid)
        D_all = rifft_space_grid.shape[1] - 1
        coords_all = rifft_space_grid[:, :D_all]
        ids_all = rifft_space_grid[:, -1].astype(int)

        # map central_point_id → center coords
        id2center = {}
        for pd in point_data_list:
            cid = int(pd["central_point_id"])
            id2center[cid] = np.asarray(pd["coordinates"], float)[:D_all]

        # group rows by central_point_id
        groups = defaultdict(list)
        for i, cid in enumerate(ids_all):
            groups[int(cid)].append(i)

        # shared knobs
        weight_g = float(self.parameters.get("ls_weight_gamma", 0.35))
        lam_reg = float(self.parameters.get("dog_lambda_reg", 1e-3))
        max_train = self.parameters.get("linear_max_training_samples", None)

        intervals = self.parameters["reciprocal_space_intervals_all"]
        hkl_max_xyz = compute_hkl_max_from_intervals(intervals)
        guard_frac = float(self.parameters.get("edge_guard_frac", 0.10))
        q_window_kind = str(
            self.parameters.get("q_window_kind", "cheb")
        ).lower()
        q_window_at_db = float(
            self.parameters.get("q_window_at_db", 100.0)
        )
        size_aver = np.asarray(self.parameters["supercell"], dtype=int)

        original_coords = self.original_coords
        average_coords = self.average_coords

        # ------------------------------------------------------------------ NEW: try to load decoder M from cache
        decoder_cache_path = self._get_decoder_cache_path(output_dir)
        if self._decoder_M is None and os.path.isfile(decoder_cache_path):
            try:
                cache = np.load(decoder_cache_path, allow_pickle=False)
                self._decoder_M = np.asarray(cache["M"], float)
                fd = cache.get("feature_dim", None)
                if fd is not None:
                    self._feature_dim = int(np.ravel(fd)[0])
                else:
                    self._feature_dim = self._decoder_M.shape[1]
                log.info(
                    "Loaded decoder M from '%s' (shape %s, feature_dim=%d).",
                    decoder_cache_path,
                    self._decoder_M.shape,
                    self._feature_dim,
                )
            except Exception as e:
                log.warning(
                    "Failed to load decoder M from '%s': %s. Will retrain.",
                    decoder_cache_path,
                    e,
                )
                self._decoder_M = None
                self._feature_dim = None
        # ------------------------------------------------------------------ end NEW

        # containers
        features_all = []
        cids_all = []
        features_train = []
        u_train = []

        # ---- per-site loop: build features, training data --------------------
        for pd in point_data_list:
            cid = int(pd["central_point_id"])
            if cid not in groups:
                continue
            center = id2center.get(cid, None)
            if center is None:
                continue

            idxs = np.asarray(groups[cid], int)
            coords = coords_all[idxs, :]
            Rvals = Rvals_all[idxs]

            # Q-stage
            Rvals_proc = _apply_rq_pipeline_local(
                Rvals,
                coords,
                q_window_kind=q_window_kind,
                q_window_at_db=q_window_at_db,
                size_aver=size_aver,
                hkl_max_xyz=hkl_max_xyz,
                guard_frac=guard_frac,
            )

            # local patch in real space
            y_grid, shape, axes_vals, _ = _regrid_patch_to_c(
                coords, Rvals_proc
            )
            D = len(shape)

            feat = _build_feature_vector_from_patch(
                y_grid,
                axes_vals,
                center_abs=center,
                D=D,
                weight_gamma=weight_g,
                remove_odd_tilt=True,
            )

            features_all.append(feat)
            cids_all.append(cid)

            # training pairs (only needed while decoder not yet known)
            if self._decoder_M is None:
                if (max_train is not None) and (
                    len(features_train) >= max_train
                ):
                    continue

                if cid < 0 or cid >= original_coords.shape[0]:
                    raise IndexError(
                        f"central_point_id {cid} out of bounds "
                        f"for original_coords shape {original_coords.shape}"
                    )

                if self.u_true_all is not None:
                    u_true = self.u_true_all[cid, :3]
                else:
                    u_true = (
                        original_coords[cid, :3] @ np.linalg.inv(self.parameters["vectors"])
                        - average_coords[cid, :3] @ np.linalg.inv(self.parameters["vectors"])
                    )
                    u_true = (u_true - np.rint(u_true)) @ self.parameters["vectors"]

                features_train.append(feat)
                u_train.append(np.asarray(u_true, float))

        if not features_all:
            raise RuntimeError("No site features constructed; nothing to do.")

        # ---- train decoder M (once) -----------------------------------------
        if self._decoder_M is None:
            if not features_train:
                raise RuntimeError(
                    "Decoder M not present and no training samples collected. "
                    "Check that original_coords / average_coords have matching "
                    "indices with central_point_id."
                )

            R_data = np.stack(features_train, axis=1)  # (P, N)
            U_data = np.stack(u_train, axis=1)         # (3, N)

            P, N = R_data.shape
            log.info(
                "Training linear decoder M on chunk %s with %d samples (P=%d).",
                chunk_id,
                N,
                P,
            )

            RR = R_data @ R_data.T                     # (P, P)
            UR = U_data @ R_data.T                     # (3, P)
            H = RR + float(lam_reg) * np.eye(P)

            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                log.warning(
                    "Decoder normal matrix H is singular; using pseudo-inverse."
                )
                H_inv = np.linalg.pinv(H, rcond=1e-12)

            M = UR @ H_inv                             # (3, P)

            self._decoder_M = M.astype(np.float64, copy=False)
            self._feature_dim = P
            log.info("Decoder M trained (shape %s).", self._decoder_M.shape)

            # ----------------------- NEW: save decoder to cache --------------
            try:
                np.savez(
                    decoder_cache_path,
                    M=self._decoder_M,
                    feature_dim=np.array(self._feature_dim, dtype=np.int64),
                )
                log.info("Decoder M saved to '%s'.", decoder_cache_path)
            except Exception as e:
                log.warning(
                    "Failed to save decoder M to '%s': %s",
                    decoder_cache_path,
                    e,
                )
            # ------------------------------------------------------------------
        else:
            M = self._decoder_M
            P = M.shape[1]

        # sanity check
        if self._feature_dim is None:
            self._feature_dim = self._decoder_M.shape[1]
        if any(f.size != self._feature_dim for f in features_all):
            raise RuntimeError(
                "Feature dimension mismatch: decoder expects "
                f"{self._feature_dim}, but some features differ."
            )

        # ---- apply decoder to all sites in this chunk -----------------------
        R_all = np.stack(features_all, axis=1)          # (P, N_sites_chunk)
        U_all = (self._decoder_M @ R_all).T             # (N_sites_chunk, 3)

        ids = np.array(cids_all, dtype=np.int64)
        U = U_all.astype(np.float64, copy=False)

        # prepare output table
        out_table = {
            "central_point_id": ids,
            "u": U,
            "columns": np.array(["ux", "uy", "uz"], dtype=object),
            "coordinate_system": np.array(["cartesian"], dtype=object),
            "units": np.array(
                ["angstrom", "angstrom", "angstrom"], dtype=object
            ),
        }

        h5_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.h5"
        )
        csv_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.csv"
        )
        rifft_saver.save_data(out_table, h5_path)
        _write_displacements_csv(csv_path, ids, U)

        # Optional: broadcast u back into amplitude rows
        if broadcast_into_rows:
            cid2u = {int(i): U[k, :] for k, i in enumerate(ids)}
            urows = np.stack([cid2u[int(c)] for c in ids_all], axis=0)
            if amplitudes.ndim == 1:
                aug = np.column_stack([amplitudes, urows])
            else:
                aug = np.concatenate([amplitudes, urows], axis=1)

            d_aug = dict(d)
            d_aug["amplitudes_with_displacement"] = aug
            d_aug["amplitudes_with_displacement_columns"] = np.array(
                ["<orig...>", "ux", "uy", "uz"], dtype=object
            )
            h5_aug = os.path.join(
                output_dir,
                f"output_chunk_{chunk_id}_amplitudes_with_displacements.h5",
            )
            rifft_saver.save_data(d_aug, h5_aug)

        return out_table

    # ------------------------------------------------------------------ legacy helper (unused in M path)

    def filter_from_window(self, window0, dimensionality,
                           window1=None, window2=None, size_aver=None):
        """
        Legacy window→kernel helper kept for compatibility.
        """
        if dimensionality == 1:
            fd0 = fft(
                window0, np.array(self.parameters["supercell"])[0]
            ) / (len(window0) / 2.0)
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            return k0 / k0.sum()
        if dimensionality == 2:
            fd0 = fft(
                window0, self.parameters["supercell"][0]
            ) / (len(window0) / 2.0)
            fd1 = fft(
                window1, self.parameters["supercell"][1]
            ) / (len(window1) / 2.0)
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
            kern = k0[:, None] * k1[None, :]
            return kern / kern.sum()
        if dimensionality == 3:
            sc = np.array(self.parameters["supercell"])
            fd0 = fft(window0, sc[0]) / (len(window0) / 2.0)
            fd1 = fft(window1, sc[1]) / (len(window1) / 2.0)
            fd2 = fft(window2, sc[2]) / (len(window2) / 2.0)
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
            k2 = np.abs(fftshift(fd2 / np.abs(fd2).max()))
            kern = (
                k0[:, None, None] * k1[None, :, None] * k2[None, None, :]
            )
            return kern / kern.sum()
        raise ValueError("Unsupported dimensionality")
