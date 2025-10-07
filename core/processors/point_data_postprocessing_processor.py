# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:19:13 2025

@author: Maksim Eremenko
"""
# processors/point_data_postprocessing_processor.py
import numpy as np
import os
import time
from dask import delayed, compute
from scipy.signal import convolve
from numba import jit
from scipy.fft import fft, fftshift
from scipy.signal.windows import chebwin, lanczos
from math import sqrt, log
import logging
import sys, platform
from dask.distributed import get_client, Client, LocalCluster

from numba import set_num_threads
set_num_threads(32)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers to aggregate (h_max, k_max, l_max)
# ──────────────────────────────────────────────────────────────────────────────

def _axis_max_from_item(item: dict, axis: str) -> float:
    rng_key = f"{axis}_range"; s_key = f"{axis}_start"; e_key = f"{axis}_end"
    vals = []
    if rng_key in item and item[rng_key] is not None:
        try:
            a, b = item[rng_key]; vals.extend([a, b])
        except Exception:
            pass
    if s_key in item: vals.append(item[s_key])
    if e_key in item: vals.append(item[e_key])
    if not vals:
        return 0.0
    vals = [float(v) for v in vals]
    return float(np.max(np.abs(vals)))

def compute_hkl_max_from_intervals(intervals: list) -> tuple[float, float, float]:
    if intervals is None or len(intervals) == 0:
        return (0.0, 0.0, 0.0)
    h_max = k_max = l_max = 0.0
    for it in intervals:
        h_max = max(h_max, _axis_max_from_item(it, "h"))
        k_max = max(k_max, _axis_max_from_item(it, "k"))
        l_max = max(l_max, _axis_max_from_item(it, "l"))
    return (h_max, k_max, l_max)

# ──────────────────────────────────────────────────────────────────────────────
# Your kernel builders (kept as-is)
# ──────────────────────────────────────────────────────────────────────────────

def _build_kernel(dim, size_aver, window0, window1=None, window2=None):
    """Returns an ND convolution kernel normalised to sum-to-1."""
    if dim == 1:
        return window0 / window0.sum()
    if dim == 2:
        win = np.outer(window0, window1)
        return win / win.sum()
    if dim == 3:
        win = (window0[:, None, None] *
               window1[None, :, None] *
               window2[None, None, :])
        return win / win.sum()
    raise ValueError("Unsupported dimensionality")

def filter_kernel(window0,
                  dimensionality: int,
                  window1=None,
                  window2=None,
                  size_aver=None):
    """
    Stand-alone, picklable replacement for
    PointDataPostprocessingProcessor.filter_from_window.
    """
    import numpy as np
    from scipy.fft import fft, fftshift

    if size_aver is None:
        raise ValueError("size_aver must be provided")
    size_aver = np.asarray(size_aver, dtype=int)

    if dimensionality == 1:
        fd0  = fft(window0, size_aver[0]) / (len(window0) / 2.0)
        k0   = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        return k0 / k0.sum()

    if dimensionality == 2:
        fd0 = fft(window0, size_aver[0]) / (len(window0) / 2.0)
        fd1 = fft(window1, size_aver[1]) / (len(window1) / 2.0)
        k0  = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        k1  = np.abs(fftshift(fd1 / np.abs(fd1).max()))
        kern = k0[:, None] * k1[None, :]
        return kern / kern.sum()

    if dimensionality == 3:
        fd0 = fft(window0, size_aver[0]) / (len(window0) / 2.0)
        fd1 = fft(window1, size_aver[1]) / (len(window1) / 2.0)
        fd2 = fft(window2, size_aver[2]) / (len(window2) / 2.0)
        k0  = np.abs(fftshift(fd0 / np.abs(fd0).max()))
        k1  = np.abs(fftshift(fd1 / np.abs(fd1).max()))
        k2  = np.abs(fftshift(fd2 / np.abs(fd2).max()))
        kern = k0[:, None, None] * k1[None, :, None] * k2[None, None, :]
        return kern / kern.sum()

    raise ValueError(f"Unsupported dimensionality {dimensionality}")

# ──────────────────────────────────────────────────────────────────────────────
# Correct-domain filters:
#   • R-apodizer (multiply in R)
#   • Q-space PSF (IFFT of your window-spectrum × soft (h,k,l)_max mask; convolve in R)
# ──────────────────────────────────────────────────────────────────────────────

def _r_apodizer_window(shape_nd, at_db: float = 100.0) -> np.ndarray:
    """
    Separable Chebyshev window directly in R (correct domain), sized to shape_nd.
    To preserve DC, we normalise by mean value.
    """
    shape_nd = tuple(int(s) for s in shape_nd)
    if len(shape_nd) == 1:
        w0 = chebwin(shape_nd[0], at_db, sym=True).astype(float)
        W  = w0
    elif len(shape_nd) == 2:
        w0 = chebwin(shape_nd[0], at_db, sym=True).astype(float)
        w1 = chebwin(shape_nd[1], at_db, sym=True).astype(float)
        W  = w0[:, None] * w1[None, :]
    elif len(shape_nd) == 3:
        w0 = chebwin(shape_nd[0], at_db, sym=True).astype(float)
        w1 = chebwin(shape_nd[1], at_db, sym=True).astype(float)
        w2 = chebwin(shape_nd[2], at_db, sym=True).astype(float)
        W  = w0[:, None, None] * w1[None, :, None] * w2[None, None, :]
    else:
        raise ValueError("Unsupported dimensionality for R-apodizer")
    m = float(W.mean()) if W.size else 1.0
    if m != 0.0:
        W = W / m
    return W

def _kband_vector_unshifted(nS: int, S: int, hmax: float, guard_frac: float) -> np.ndarray:
    """
    Build a 1-D *unshifted* soft band mask on the k-grid of length nS (== S here),
    with physical bins h = fftfreq(nS, d=1/S) (rlu). Raised-cosine guard to hmax.
    """
    nS = int(nS); S = int(S)
    if nS <= 0 or S <= 0:
        raise ValueError("nS and S must be positive")

    h = np.fft.fftfreq(nS, d=1.0 / S)  # DC at index 0
    hnyq = S / 2.0

    if hmax is None or hmax <= 0.0:
        return np.ones(nS, dtype=float)

    hmax_eff = min(float(hmax), hnyq - 1e-9)
    if hmax_eff <= 0.0:
        v = np.zeros(nS, dtype=float); v[0] = 1.0
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

def _window_spectrum_vectors_unshifted(size_aver: np.ndarray,
                                       kind: str = "cheb",
                                       at_db: float = 100.0):
    """
    Recreate your per-axis |FFT(window)| vectors (like filter_from_window),
    but return them **unshifted** to be used directly in IFFT.
    """
    size_aver = np.asarray(size_aver, dtype=int)
    dim = len(size_aver)
    wd = []

    for ax in range(dim):
        S = int(size_aver[ax])
        if kind == "cheb":
            w = chebwin(S, at_db, sym=False).astype(float)
        elif kind == "lanczos":
            w = lanczos(S).astype(float)
        else:
            raise ValueError("kind must be 'cheb' or 'lanczos'")

        Fw = fft(w, S) / (len(w) / 2.0)
        mag = np.abs(Fw / (np.abs(Fw).max() if np.abs(Fw).max() != 0 else 1.0))
        # your code used fftshift; to build unshifted arrays for IFFT we invert that step
        mag_unshift = np.fft.ifftshift(np.abs(fftshift(Fw / (np.abs(Fw).max() if np.abs(Fw).max() != 0 else 1.0))))
        wd.append(mag_unshift)
    return wd

def _qspace_psf_in_r(size_aver: np.ndarray,
                     hkl_max_xyz: tuple[float, float, float],
                     *,
                     guard_frac: float,
                     window_kind: str = "cheb",
                     window_at_db: float = 100.0) -> np.ndarray:
    """
    Build B(k) = (|FFT(window)| per axis product) × (soft band mask by hkl_max),
    then PSF b(r) = IFFT{B(k)}. Return **fftshifted** real kernel, sum-normalised.
    • size_aver: supercell (Na,Nb,Nc) → discrete k grid
    • hkl_max_xyz: (h_max,k_max,l_max) → physical passband
    """
    size_aver = np.asarray(size_aver, dtype=int)
    dim = len(size_aver)

    # window spectra (unshifted) à la your code
    wd = _window_spectrum_vectors_unshifted(size_aver, kind=window_kind, at_db=window_at_db)

    # band masks per axis (unshifted), using physical binning from supercell
    h_max = float(hkl_max_xyz[0]) if dim >= 1 else 0.0
    k_max = float(hkl_max_xyz[1]) if dim >= 2 else 0.0
    l_max = float(hkl_max_xyz[2]) if dim >= 3 else 0.0

    bx = _kband_vector_unshifted(size_aver[0], size_aver[0], h_max, guard_frac)
    if dim >= 2:
        by = _kband_vector_unshifted(size_aver[1], size_aver[1], k_max, guard_frac)
    if dim == 3:
        bz = _kband_vector_unshifted(size_aver[2], size_aver[2], l_max, guard_frac)

    # combine: B(k) = window-spectrum × band-mask, unshifted indexing
    if dim == 1:
        B = wd[0] * bx
    elif dim == 2:
        B = (wd[0] * bx)[:, None] * (wd[1] * by)[None, :]
    else:
        B = (wd[0] * bx)[:, None, None] * (wd[1] * by)[None, :, None] * (wd[2] * bz)[None, None, :]

    # DC normalise (preserve mean after convolution)
    dc = float(B.flat[0]) if B.size else 1.0
    if dc != 0.0:
        B = B / dc

    # PSF in R, center it for spatial convolution
    psf = np.fft.ifftn(B).real
    psf = np.fft.fftshift(psf)

    # sum normalise for safety
    s = psf.sum()
    if s != 0.0:
        psf = psf / s
    else:
        psf *= 0.0
        center = tuple(n // 2 for n in psf.shape)
        psf[center] = 1.0

    return psf

# ──────────────────────────────────────────────────────────────────────────────
# Per-ID worker
# ──────────────────────────────────────────────────────────────────────────────

def _process_one_id(
    cid          : int,
    amp_full     : np.ndarray,
    grid_full    : np.ndarray,
    indices      : np.ndarray,
    shape_nd     : np.ndarray,
    dim          : int,
    size_aver    : np.ndarray,
    filtr_fun,                          # your original filter_from_window (kept)
    comp_max_fun,
    comp_min_fun,
    hkl_max_xyz  : tuple[float, float, float],
    guard_frac   : float,
    r_stage      : str,                 # "multiply" (preferred) or "convolve"
    r_at_db      : float,               # Chebyshev param for R window
    q_window_kind: str,                 # "cheb" or "lanczos"
    q_window_at_db: float,              # Cheb sidelobe for Q-window spectrum
) -> str:
    """
    Steps:
      R-stage:
        • multiply: r <- r * W_R  (apodize in R; correct domain), or
        • convolve: r <- r (*) K_R (legacy path using your filter_from_window)
      Q-stage (to mimic tapering in NUFFT domain before iNUFFT):
        • b(r) = IFFT{ |FFT(window)|_prod × soft band mask(hkl_max) }
        • r <- r (*) b(r)
    Both are zero-phase & DC-preserving.
    """
    if indices.size == 0:
        return ""
    print(cid)
    r_val_delta = np.real(amp_full[indices][:, 1])
    grid_pts    = grid_full[indices, :-1]
    shape       = tuple(int(s) for s in np.atleast_1d(shape_nd))

    # ----- R-stage -----------------------------------------------------------
    mode = (r_stage or "multiply").lower()
    if mode == "multiply":
        Wr = _r_apodizer_window(shape, at_db=r_at_db)
        r_after_R = (r_val_delta.reshape(shape) * Wr).ravel()
    elif mode == "convolve":
        # your legacy path: FFT(window) magnitude kernel, then convolve in R
        if dim == 1:
            w0   = chebwin(size_aver[0] // 2, 100)
            rker = filtr_fun(w0, 1, size_aver=size_aver)
            r_after_R = convolve(r_val_delta, rker, mode="same")
        elif dim == 2:
            w0, w1 = (chebwin(size_aver[i] // 2, 100) for i in (0, 1))
            rker   = filtr_fun(w0, 2, w1, size_aver=size_aver)
            r_after_R = convolve(r_val_delta.reshape(shape),
                                 rker, mode="same").ravel()
        elif dim == 3:
            w0, w1, w2 = (chebwin(size_aver[i]//2, 100) for i in (0, 1, 2))
            rker = filtr_fun(w0, 3, w1, w2, size_aver=size_aver)
            # FIX: use rker directly (do NOT FFT it)
            r_after_R = convolve(r_val_delta.reshape(shape),
                                 rker, mode="same").ravel()
        else:
            raise ValueError(f"Unsupported dimensionality {dim}")
    else:
        raise ValueError("r_stage must be 'multiply' or 'convolve'")

    # ----- Q-stage (k taper → PSF in R) -------------------------------------
    # q_psf = _qspace_psf_in_r(size_aver=size_aver/np.array(2),
    #                          hkl_max_xyz=hkl_max_xyz,
    #                          guard_frac=guard_frac,
    #                          window_kind=q_window_kind,
    #                          window_at_db=q_window_at_db)
    # if dim == 1:
    #     r_after_Q = convolve(r_after_R, q_psf, mode="same")
    # elif dim == 2:
    #     r_after_Q = convolve(r_after_R.reshape(shape), q_psf, mode="same").ravel()
    # elif dim == 3:
    #     r_after_Q = convolve(r_after_R.reshape(shape), q_psf, mode="same").ravel()
    # else:
    #     r_after_Q = r_after_R
        
    w0, w1, w2 = (chebwin(int(hkl_max_xyz[i]//2), 100) for i in (0, 1, 2))
    q_psf = filter_kernel(w0, 3, w1, w2, size_aver=np.int32(hkl_max_xyz)/np.array(1))
    r_after_Q = convolve(r_val_delta.reshape(shape), q_psf, mode="same").ravel()
    
    
    # ----- extrema -----------------------------------------------------------
    pmax_coord, pmax_idx, pmax_flat, amp_max_w = comp_max_fun(r_after_Q, grid_pts)
    pmin_coord, pmin_idx, pmin_flat, amp_min_w = comp_min_fun(r_after_Q, grid_pts)

    if pmax_coord is None:
        pmax_coord = [float("nan")] * dim; pmax_idx = pmax_flat = -1; amp_max_w = float("nan")
    if pmin_coord is None:
        pmin_coord = [float("nan")] * dim; pmin_idx = pmin_flat = -1; amp_min_w = float("nan")

    displ_vec = (
        [float("nan")] * dim
        if -1 in (pmax_idx, pmin_idx)
        else grid_pts[pmax_idx] - grid_pts[pmin_idx]
    )

    return (
        f"{cid} {pmin_idx} {pmax_idx} "
        f"{' '.join(map(str, pmax_coord))} "
        f"{' '.join(map(str, pmin_coord))} "
        f"{' '.join(map(str, displ_vec))} "
        f"{amp_min_w} {amp_max_w} "
        f"{pmin_flat} {pmax_flat}\n"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main processor
# ──────────────────────────────────────────────────────────────────────────────

class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        self.parameters = parameters

    def process_chunk(self, chunk_id, rifft_saver, client, output_dir):
        point_data_list = self.db_manager.get_point_data_for_chunk(chunk_id)
        if not point_data_list:
            print(f"No point data found for chunk_id: {chunk_id}")
            return None

        rifft_space_grid, amplitudes, grids_shapeNd = self.load_amplitudes_and_generate_grid(
            chunk_id, point_data_list, rifft_saver)
        if rifft_space_grid.size == 0 or amplitudes is None:
            print(f"No valid data for chunk_id: {chunk_id}")
            return None

        self.calculate_and_save_positions(chunk_id, rifft_space_grid, amplitudes, grids_shapeNd, client, output_dir)

    def load_amplitudes_and_generate_grid(self, chunk_id, point_data_list, rifft_saver):
        filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
        try:
            data = rifft_saver.load_data(filename)
            amplitudes = data.get('amplitudes', None)
            if amplitudes is None:
                print(f"Amplitudes not found in {filename}")
                return np.array([]), None

            grids = []
            grids_shapeNd = []
            central_point_ids = []
            for pd in point_data_list:
                grid_points, grid_shapeNd = self.point_data_processor._generate_grid(
                    chunk_id=chunk_id,
                    dimensionality=len(pd['coordinates']),
                    step_in_frac=pd['step_in_frac'],
                    central_point=pd['coordinates'],
                    dist=pd['dist_from_atom_center'],
                    central_point_id=pd['central_point_id']
                )
                grids.append(grid_points)
                grids_shapeNd.append(grid_shapeNd)
                central_point_ids.extend([pd['central_point_id']] * len(grid_points))

            rifft_space_grid = np.hstack((np.vstack(grids), np.array(central_point_ids)[:, None])) if grids else np.array([])
            return rifft_space_grid, amplitudes, grids_shapeNd
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return np.array([]), None

    # Your original function (unchanged, still available)
    def filter_from_window(self, window0, dimensionality, window1=None, window2=None, size_aver=None):
        if dimensionality == 1:
            wd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (len(window0) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            win = wd0_s
            win = win/np.sum(win)
        elif dimensionality == 2:
            wd0 = fft(window0, self.parameters["supercell"][0]) / (len(window0) / 2.0)
            wd1 = fft(window1, self.parameters["supercell"][1]) / (len(window1) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            wd1_s = np.abs(fftshift(wd1 / np.abs(wd1).max()))
            wd0_q =  chebwin(self.parameters["supercell"][0], 100) #lanczos(80)  #
            wd1_q  = chebwin(self.parameters["supercell"][1], 100) #lanczos(80) #
            wd0_s = wd0_q * wd0_s
            wd1_s = wd1_q * wd1_s
            win = np.ones((self.parameters["supercell"][0], self.parameters["supercell"][1]))
            for i in range(self.parameters["supercell"][0]):
                for j in range(self.parameters["supercell"][1]):
                        win[i, j] = wd0_s[i] * wd1_s[j]
            win =win/np.sum(win)
        elif dimensionality == 3:
            wd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (len(window0) / 2.0)
            wd1 = fft(window1, np.array(self.parameters["supercell"])[1]) / (len(window1) / 2.0)
            wd2 = fft(window2, np.array(self.parameters["supercell"])[2]) / (len(window2) / 2.0)
            wd0_s = np.abs(fftshift(wd0 / np.abs(wd0).max()))
            wd1_s = np.abs(fftshift(wd1 / np.abs(wd1).max()))
            wd2_s = np.abs(fftshift(wd2 / np.abs(wd2).max()))
            win = np.ones((np.array(self.parameters["supercell"])[0], np.array(self.parameters["supercell"])[1], np.array(self.parameters["supercell"])[2]))
            for i in range(np.array(self.parameters["supercell"])[0]):
                for j in range(np.array(self.parameters["supercell"])[1]):
                    for k in range(np.array(self.parameters["supercell"])[2]):
                        win[i, j, k] = wd0_s[i] * wd1_s[j] * wd2_s[k]
            win = win/np.sum(win)
        else:
            raise ValueError("Unsupported dimensionality: {}".format(dimensionality))
        return win

    def calculate_and_save_positions(
            self,
            chunk_id        : int,
            rifft_space_grid: np.ndarray,
            amplitudes      : np.ndarray,
            grids_shapeNd   : np.ndarray,
            client,
            output_dir      : str) -> None:
        """
        Write one *.dat* file per chunk, applying:
          r ← ( r * W_R ) (*) b(r)
        where W_R is an R-window (multiply) and b(r) is PSF from Q taper (convolve).
        """
        import os, logging, dask
        from dask.distributed import wait
        import numpy as np
    
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, f"output_chunk_{chunk_id}.dat")
    
        # ------------------------ dimensions & sizes ------------------------
        if rifft_space_grid.ndim != 2 or rifft_space_grid.shape[1] < 2:
            raise ValueError(f"rifft_space_grid must be (N, D+1) with last col=id; got {rifft_space_grid.shape}")
        dim       = rifft_space_grid.shape[1] - 1
        size_aver = np.asarray(self.parameters["supercell"], dtype=int)
    
        # ------------------------ stable ids & mapping ----------------------
        ids = rifft_space_grid[:, -1].astype(int)
    
        # stable (first-occurrence) unique ids; avoids np.unique sorting
        unique_ids = np.fromiter(dict.fromkeys(ids), dtype=int)
    
        # row indices per id
        id2indices = {int(cid): np.flatnonzero(ids == int(cid)) for cid in unique_ids}
    
        # robust shape map for each cid
        def _make_shape_map(grids_shapeNd, unique_ids_arr):
            if isinstance(grids_shapeNd, dict):
                return {int(cid): tuple(map(int, grids_shapeNd[int(cid)])) for cid in unique_ids_arr}
    
            G = np.asarray(grids_shapeNd)
            if G.ndim == 1:
                G = G.reshape(-1, 1)
    
            shape_map = {}
            # Case B: G aligned with stable unique id order (n_ids x dim)
            if G.shape[0] == len(unique_ids_arr):
                for i, cid in enumerate(unique_ids_arr):
                    shape_map[int(cid)] = tuple(map(int, G[i]))
                return shape_map
    
            # Case C: G indexable by numeric cid (rows correspond to cid)
            max_cid = int(unique_ids_arr.max()) if unique_ids_arr.size else -1
            if G.shape[0] > max_cid:
                for cid in unique_ids_arr:
                    shape_map[int(cid)] = tuple(map(int, G[int(cid)]))
                return shape_map
    
            raise ValueError(
                f"Incompatible grids_shapeNd: shape={G.shape}, "
                f"len(unique_ids)={len(unique_ids_arr)}, max(cid)={max_cid}. "
                f"Expected either len==n_ids (stable order) or indexable by cid."
            )
    
        shape_for_id = _make_shape_map(grids_shapeNd, unique_ids)
    
        # sanity check: product of shape must equal number of rows per id
        for cid in unique_ids:
            shp = tuple(int(x) for x in shape_for_id[int(cid)])
            n_from_shape = int(np.prod(shp))
            n_rows = int(id2indices[int(cid)].size)
            if n_from_shape != n_rows:
                raise ValueError(
                    f"cid {int(cid)}: shape {shp} → {n_from_shape} points, "
                    f"but rifft_space_grid has {n_rows} rows for that id. "
                    f"Check grids_shapeNd alignment."
                )
    
        # ------------------------ hkl bounds from intervals -----------------
        if "reciprocal_space_intervals_all" not in self.parameters:
            raise ValueError("parameters['reciprocal_space_intervals_all'] must be provided")
        intervals   = self.parameters["reciprocal_space_intervals_all"]
        hkl_max_xyz = compute_hkl_max_from_intervals(intervals)
    
        # ------------------------ controls ---------------------------------
        guard_frac     = float(self.parameters.get("edge_guard_frac", 0.10))
        r_stage        = str(self.parameters.get("r_stage", "multiply"))     # "multiply" or "convolve"
        r_at_db        = float(self.parameters.get("r_cheb_at_db", 100.0))
        q_window_kind  = str(self.parameters.get("q_window_kind", "cheb")).lower()  # "cheb" | "lanczos"
        q_window_at_db = float(self.parameters.get("q_window_at_db", 100.0))        # only used if cheb
    
        # ------------------------ scatter large inputs ----------------------
        amp_future  = client.scatter(amplitudes,       broadcast=True, hash=False)
        grid_future = client.scatter(rifft_space_grid, broadcast=True, hash=False)
    
        # ------------------------ build tasks -------------------------------
        tasks = [
            dask.delayed(_process_one_id)(
                int(cid),
                amp_future,
                grid_future,
                id2indices[int(cid)],
                tuple(map(int, shape_for_id[int(cid)])),
                dim,
                size_aver,
                filter_kernel,  # module-level function
                PointDataPostprocessingProcessor.compute_weighted_positions_max,  # class static
                PointDataPostprocessingProcessor.compute_weighted_positions_min,  # class static
                hkl_max_xyz=hkl_max_xyz,
                guard_frac=guard_frac,
                r_stage=r_stage,
                r_at_db=r_at_db,
                q_window_kind=q_window_kind,
                q_window_at_db=q_window_at_db,
            )
            for cid in unique_ids
        ]
    
        # ------------------------ compute (sync vs distributed) -------------
        res = client.compute(tasks)
        try:
            # distributed path: res is list of futures
            wait(res)
            lines = client.gather(res)
        except Exception:
            # sync path: res already concrete results
            lines = res
    
        # ------------------------ normalize to list[str] --------------------
        flat: list[str] = []
        for item in lines:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                for s in item:
                    if s is None:
                        continue
                    s = s if isinstance(s, str) else str(s)
                    if not s.endswith("\n"):
                        s += "\n"
                    flat.append(s)
            else:
                s = item if isinstance(item, str) else str(item)
                if not s.endswith("\n"):
                    s += "\n"
                flat.append(s)
    
        # ------------------------ write output ------------------------------
        with open(outfile, "w") as f:
            self.write_header(f, dimensionality=dim)
            f.writelines(flat)
    
        logging.getLogger(__name__).info(
            "chunk %d – wrote %d central-point records → %s",
            chunk_id, len(flat), outfile,
        )

    def write_header(self, file, dimensionality):
         header = ['central_point_id', 'pos_min_weighted', 'pos_max_weighted']
         coord_headers = [f'pos_max_coord_{d}' for d in range(dimensionality)]
         coord_headers += [f'pos_min_coord_{d}' for d in range(dimensionality)]
         coord_headers += [f'displacement_vector_{d}' for d in range(dimensionality)]
         header.extend(coord_headers)
         header += ['amplitude_min_weighted', 'amplitude_max_weighted', 'pos_min', 'pos_max']
         file.write(' '.join(header) + '\n')

    @staticmethod
    def _mean_shift(points, weights, bandwidth: float | None = None, max_iter: int = 100, tol: float = 1e-4):
        points  = np.asarray(points,  dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        N, D = points.shape
        if N == 0: raise ValueError("mean-shift: no points supplied")
        if N == 1 or weights.sum() == 0.0: return points[0].copy()
        if bandwidth is None:
            std = np.std(points, axis=0, ddof=0)
            sigma = np.exp(np.mean(np.log(std + 1e-12)))
            bandwidth = 1.06 * sigma * N ** (-1.0 / (D + 4))
        if bandwidth <= 0.0 or not np.isfinite(bandwidth):
            bbox = points.ptp(axis=0).max(); bandwidth = max(bbox * 0.01, 1e-6)
        m = np.sum(points * weights[:, None], axis=0) / weights.sum()
        for _ in range(max_iter):
            d2 = np.sum((points - m) ** 2, axis=1)
            k  = np.exp(-0.5 * d2 / (bandwidth ** 2))
            kw = k * weights; s = kw.sum()
            if s < 1e-20: break
            m_new = np.sum(points * kw[:, None], axis=0) / s
            if np.linalg.norm(m_new - m) < tol: break
            m = m_new
        return m

    @staticmethod
    def compute_weighted_positions_max(r_vals, r_grid, variable=0.0):
        def _norm_pos(vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            tot = float(vals.sum())
            return vals / tot if tot > eps else np.full_like(vals, 1.0 / len(vals), dtype=float)
        mask = r_vals >= variable
        if not np.any(mask): return None, None, np.argmax(r_vals), 0.0
        vals = r_vals[mask]; points = r_grid[mask]; weights = _norm_pos(vals)
        centre = PointDataPostprocessingProcessor._mean_shift(points, weights)
        distances = np.linalg.norm(r_grid - centre, axis=1)
        nearest_idx = int(np.argmin(distances))
        argmax_idx  = int(np.argmax(r_vals))
        amp_weighted = float(np.dot(weights, vals))
        return centre, nearest_idx, argmax_idx, amp_weighted

    @staticmethod
    def compute_weighted_positions_min(r_vals, r_grid, variable=0.0):
        def _norm_pos(vals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            tot = float(vals.sum())
            return vals / tot if tot > eps else np.full_like(vals, 1.0 / len(vals), dtype=float)
        mask = r_vals <= variable
        if not np.any(mask): return None, None, np.argmin(r_vals), 0.0
        vals = -r_vals[mask]; points = r_grid[mask]; weights = _norm_pos(vals)
        centre = PointDataPostprocessingProcessor._mean_shift(points, weights)
        distances = np.linalg.norm(r_grid - centre, axis=1)
        nearest_idx = int(np.argmin(distances))
        argmin_idx  = int(np.argmin(r_vals))
        amp_weighted = float(np.dot(weights, -vals))
        return centre, nearest_idx, argmin_idx, amp_weighted
