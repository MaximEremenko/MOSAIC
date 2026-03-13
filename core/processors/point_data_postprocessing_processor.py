# -*- coding: utf-8 -*-
"""
processors/point_data_postprocessing_processor.py

Strictly linear per-site displacement estimator (DoG-based LS).

Changes to guarantee linearity wrt input RIFFT residual R:
  • Fixed, data-independent design (Derivative-of-Gaussian kernel).
  • No design selection by "energy", no α rescaling, no sign flips.
  • Optional odd-plane (tilt) removal kept — it's linear in O.
  • Sub-voxel centering, even/odd split, fixed radial weights — linear.
  • r_stage forcibly "none" (user said mask already applied in Q).
  • Amplitudes normalized ONLY by ntotal reciprocal points.

Result: u is an affine-linear map of the RIFFT residual R (actually linear,
since no constant term), so u[A] = u[M·A] + u[(1−M)·A] holds to numerical precision.

Author: Maksim Eremenko + linear DoG revision
"""

import os, csv, logging
import numpy as np
from collections import defaultdict
from scipy.signal.windows import chebwin
from scipy.signal import convolve
from scipy.fft import fft, fftshift

from numba import set_num_threads
set_num_threads(32)

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_scalar_from_store(rifft_saver, filename, key_candidates: list[str]) -> int | None:
    try:
        d = rifft_saver.load_data(filename)
    except FileNotFoundError:
        return None
    for k in key_candidates:
        if k in d:
            v = np.array(d[k])
            try: return int(np.ravel(v)[0])
            except Exception: pass
    return None

def _normalize_amplitudes_ntotal(amplitudes: np.ndarray, *, rifft_saver, chunk_id: int,
                                 logger: logging.Logger | None = None) -> np.ndarray:
    """
    Scale amplitudes by 1/ntotal only (total # of reciprocal-space points for the
    unmasked grid). This constant scale factor preserves superposition between
    masked/unmasked runs.
    """
    fn_tot  = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes_ntotal_reciprocal_space_points")
    ntot = _load_scalar_from_store(rifft_saver, fn_tot, ["ntotal_reciprocal_space_points","ntotal_reciprocal_points"])
    if not ntot or ntot <= 0:
        if logger: logger.warning("[normalize_amplitudes_ntotal] Missing ntotal; skipping.")
        return amplitudes
    scale = 1.0/float(ntot)
    try:
        if amplitudes.ndim==2 and amplitudes.shape[1]>=2:
            amplitudes[:,1] *= scale
        else:
            amplitudes[...] *= scale
    except Exception as e:
        if logger: logger.warning("[normalize_amplitudes_ntotal] scale failed: %s", e)
    return amplitudes

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate/grid helpers (Cartesian)
# ──────────────────────────────────────────────────────────────────────────────

def _axes_from_coords(coords: np.ndarray):
    coords = np.asarray(coords, float); D = coords.shape[1]
    axes_vals = [np.sort(np.unique(coords[:,a])) for a in range(D)]
    shape = tuple(int(v.size) for v in axes_vals)
    return shape, axes_vals

def _indices_from_axes(coords: np.ndarray, axes_vals: list[np.ndarray]) -> tuple[np.ndarray,...]:
    coords = np.asarray(coords, float); D = coords.shape[1]; idxs=[]
    for a in range(D):
        av = axes_vals[a]
        i = np.searchsorted(av, coords[:,a]); i = np.clip(i, 1, len(av)-1)
        left  = np.abs(coords[:,a]-av[i-1]); right = np.abs(coords[:,a]-av[i])
        i = np.where(right<left, i, i-1).astype(np.int64); idxs.append(i)
    return tuple(idxs)

def _regrid_patch_to_c(coords: np.ndarray, vals: np.ndarray):
    shape, axes_vals = _axes_from_coords(coords)
    flat = np.ravel_multi_index(_indices_from_axes(coords, axes_vals), shape, order="C")
    y_flat = np.empty(flat.size, float); y_flat[flat] = np.asarray(vals, float)
    return y_flat.reshape(shape, order="C"), shape, axes_vals, y_flat

def _fractional_center_index(center_val: float, axis_vals: np.ndarray) -> float:
    av = np.asarray(axis_vals, float); j = int(np.clip(np.searchsorted(av, float(center_val)), 1, av.size-1))
    j0=j-1; j1=j; dv = av[j1]-av[j0]
    t = 0.0 if abs(dv)<1e-30 else (float(center_val)-av[j0])/dv
    return j0 + t  # float index

def _fourier_shift_nd(arr: np.ndarray, shift: tuple[float,...]) -> np.ndarray:
    F = np.fft.fftn(arr)
    for ax,sh in enumerate(shift):
        n = arr.shape[ax]
        phase = np.exp(-2j*np.pi*sh*np.fft.fftfreq(n))
        shape=[1]*arr.ndim; shape[ax]=n
        F *= phase.reshape(shape)
    return np.fft.ifftn(F).real

def _center_patch_subvoxel(y_grid: np.ndarray, axes_vals: list[np.ndarray], center_abs: np.ndarray):
    mid = tuple((n-1)/2.0 for n in y_grid.shape)
    i_star = [_fractional_center_index(center_abs[a], axes_vals[a]) for a in range(y_grid.ndim)]
    delta = tuple(mid[a]-i_star[a] for a in range(y_grid.ndim))
    y_c = _fourier_shift_nd(y_grid, delta)
    # ensure odd sizes (drop 1 at high end if even)
    sl=[]
    for n in y_c.shape:
        if n%2==1: sl.append(slice(None))
        else:      sl.append(slice(0,n-1))
    return y_c[tuple(sl)], None

# ──────────────────────────────────────────────────────────────────────────────
# Windows & weights (R-space)
# ──────────────────────────────────────────────────────────────────────────────

def _radial_gaussian_weights_cart(shape: tuple[int,...], steps_cart: tuple[float,...], gamma: float=0.35) -> np.ndarray:
    coords_1d=[(np.arange(n)-(n-1)/2.0)*float(steps_cart[a]) for a,n in enumerate(shape)]
    if len(shape)==1:
        r2=coords_1d[0]**2
    elif len(shape)==2:
        x,y=np.meshgrid(coords_1d[0], coords_1d[1], indexing='ij'); r2=x*x+y*y
    else:
        x,y,z=np.meshgrid(coords_1d[0], coords_1d[1], coords_1d[2], indexing='ij'); r2=x*x+y*y+z*z
    edge=np.array([max(abs(s[0]),abs(s[-1])) for s in coords_1d], float)
    r_edge=float(np.linalg.norm(edge)); sigma=max(gamma*r_edge,1e-12)
    return np.exp(-0.5*(r2/(sigma*sigma)))

# ──────────────────────────────────────────────────────────────────────────────
# Fixed DoG template (linear design)
# ──────────────────────────────────────────────────────────────────────────────

def _dog_kernel_grad(window_shape: tuple[int, ...],
                     steps_phys: tuple[float, ...],
                     sigma_vox: tuple[float, ...]) -> tuple[np.ndarray, ...]:
    """
    Gradient of a separable Gaussian bump Φ with widths given in *voxels*,
    evaluated on the centered patch with physical spacings steps_phys (Å).
    Returns tuple (Gx, Gy, [Gz]) in units 1/Å, same shape as window_shape.
    """
    D = len(window_shape)
    coords = [np.arange(L) - (L - 1) / 2.0 for L in window_shape]
    meshes = np.meshgrid(*coords, indexing="ij")

    # physical coordinates per axis
    x = [meshes[a] * float(steps_phys[a]) for a in range(D)]
    sig_phys = [float(sigma_vox[a]) * float(steps_phys[a]) for a in range(D)]

    quad = np.zeros(window_shape, float)
    for a in range(D):
        if sig_phys[a] > 0:
            quad += (x[a] / sig_phys[a]) ** 2
    Phi = np.exp(-0.5 * quad)

    # ∂Φ/∂x_a = -(x_a / σ_a^2) Φ
    grads = []
    for a in range(D):
        if sig_phys[a] > 0:
            ga = -(x[a] / (sig_phys[a] ** 2)) * Phi
        else:
            ga = np.zeros(window_shape, float)
        grads.append(ga)

    # Normalise Φ so ΣΦ=1 ⇒ gradient scale predictable
    s = Phi.sum()
    if s > 0:
        for a in range(D):
            grads[a] /= s
    return tuple(grads)

def _estimate_u_linear_dog(
    y_grid: np.ndarray,               # per-site residual patch (already regridded)
    axes_vals: list[np.ndarray],      # per-axis absolute coordinates (Å)
    center_abs: np.ndarray,           # site in Å
    *,
    D: int,
    kernel_sigma_vox: tuple[float, ...] | None = None,
    weight_gamma: float = 0.35,
    lambda_reg: float = 1e-3,
    remove_odd_tilt: bool = True,
) -> np.ndarray:
    """
    Strictly linear DoG LS:
      O_clean ≈ -(G u), with fixed G from a DoG template.
      u = -(G^T W G + λI)^{-1} G^T W O_clean
    No α, no sign flips, no data-dependent branching.
    """
    # center residual
    y_c, _ = _center_patch_subvoxel(y_grid, axes_vals, center_abs[:D])
    flip_axes = tuple(range(D))

    # even/odd of residual
    O = 0.5*(y_c - np.flip(y_c, axis=flip_axes))

    # geometry + weights
    steps = tuple(float(np.mean(np.diff(axes_vals[a]))) if len(axes_vals[a])>1 else 1.0 for a in range(D))
    coords_1d = [(np.arange(n) - (n - 1) / 2.0) * float(steps[a]) for a, n in enumerate(y_c.shape)]

    if D == 1:
        s0 = coords_1d[0]; r2 = s0**2
    elif D == 2:
        s0, s1 = np.meshgrid(coords_1d[0], coords_1d[1], indexing='ij'); r2 = s0*s0 + s1*s1
    else:
        s0, s1, s2 = np.meshgrid(coords_1d[0], coords_1d[1], coords_1d[2], indexing='ij'); r2 = s0*s0 + s1*s1 + s2*s2

    W  = _radial_gaussian_weights_cart(y_c.shape, steps, gamma=weight_gamma)
    w  = W.ravel('C'); sw = np.sqrt(w + 1e-30)
    N  = O.size

    # optional odd tilt removal on outer ring — linear in O
    if remove_odd_tilt:
        r = np.sqrt(r2); r_max = float(np.max(r)) if r.size else 1.0
        mask = (r > 0.55 * r_max).ravel('C')

        if D == 1:
            Sodd = np.vstack([s0.ravel('C'), np.zeros(N), np.zeros(N)])
        elif D == 2:
            Sodd = np.vstack([s0.ravel('C'), s1.ravel('C'), np.zeros(N)])
        else:
            Sodd = np.vstack([s0.ravel('C'), s1.ravel('C'), s2.ravel('C')])

        Ao   = Sodd[:, mask] * sw[mask][None, :]
        rhsO = (O.ravel('C')[mask]) * w[mask]
        Ho   = Ao @ Ao.T + 1e-12*np.eye(3)
        fo   = Ao @ rhsO
        b_hat = np.linalg.solve(Ho, fo)
        O = O - (b_hat[0]*s0 + (b_hat[1]*s1 if D>=2 else 0.0) + (b_hat[2]*s2 if D==3 else 0.0))

    # Fixed DoG design
    if kernel_sigma_vox is None:
        _REL = 1.85  # tighter DoG (~1/3 of previous 0.6) → ~1/9 scale on u
        sigma_vox = tuple(max(_REL * ((n - 1) / 2.0), 1.0) for n in y_c.shape)
    else:
        sigma_vox = tuple(float(v) for v in kernel_sigma_vox[:D])

    Gk = _dog_kernel_grad(y_c.shape, steps, sigma_vox)  # tuple of length D
    # Build design: N×D (columns are axis-gradients flattened)
    G = np.stack([Gk[a].ravel('C') for a in range(D)], axis=1)

    # Weighted ridge LS: u = -(G^T W G + λI)^{-1} G^T W O
    GTW  = G.T * w[None, :]
    H    = GTW @ G + float(lambda_reg) * np.eye(D)
    rhs  = GTW @ (O.ravel('C'))
    try:
        uD = - np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        uD = - (np.linalg.pinv(H, rcond=1e-12) @ rhs)

    u = np.zeros(3, float)
    for a in range(D): u[a] = float(uD[a])
    return u

# ──────────────────────────────────────────────────────────────────────────────
# Q-space PSF (keep; depends only on hkl_max_xyz)
# ──────────────────────────────────────────────────────────────────────────────

def _kband_vector_unshifted(nS: int, S: int, hmax: float, guard_frac: float) -> np.ndarray:
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
    size_aver = np.asarray(size_aver, dtype=int)
    dim = len(size_aver)
    wd = []

    for ax in range(dim):
        S = int(size_aver[ax])
        if kind == "cheb":
            w = chebwin(S, at_db, sym=False).astype(float)
        else:
            raise ValueError("kind must be 'cheb'")

        Fw = fft(w, S) / (len(w) / 2.0)
        # build unshifted arrays for IFFT
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

    # window spectra (unshifted)
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
# Q-stage (R-local) apply — r_stage is forced to "none"
# ──────────────────────────────────────────────────────────────────────────────

def _infer_shape_from_coords(coords: np.ndarray) -> tuple[int, ...]:
    coords = np.asarray(coords, float)
    D = coords.shape[1]
    return tuple(int(np.unique(coords[:, a]).size) for a in range(D))
def _dc_preserving_convolve_local(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Normalized ('constant-preserving') convolution on a *cropped* patch.

    y = (x * k) / (1 * |k|)

    • Works with zero-padded 'same' mode but keeps DC exactly: constant inputs
      come out unchanged ⇒ integral is preserved on the crop.
    • Uses |k| for the normalization track to handle tiny negative side-lobes.
    """
    from scipy.signal import convolve
    x = np.asarray(arr)
    k = np.asarray(kernel)
    y = convolve(x, k, mode="same")
    w = convolve(np.ones_like(x, dtype=float), np.abs(k), mode="same")
    return y / (w + 1e-30)
from scipy import ndimage

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
    return Rvals

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
# CSV writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_displacements_csv(csv_path: str, ids: np.ndarray, U: np.ndarray) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(["central_point_id","ux","uy","uz","units=cartesian"])
        for i,(ux,uy,uz) in zip(ids.tolist(), U.tolist()):
            w.writerow([i, f"{ux:.6E}", f"{uy:.6E}", f"{uz:.6E}", "cartesian"])

# ──────────────────────────────────────────────────────────────────────────────
# Main processor
# ──────────────────────────────────────────────────────────────────────────────

class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        self.parameters = dict(parameters or {})
        # Defaults (locked for linear path)
        self.parameters.setdefault("normalize_amplitudes_by", "ntotal")  # enforced below
        self.parameters.setdefault("coords_are_fractional", False)       # Cartesian Å
        self.parameters.setdefault("displacement_method", "dog_linear")  # fixed linear DoG

        # LS knobs (used by linear DoG)
        self.parameters.setdefault("ls_weight_gamma", 0.35)
        self.parameters.setdefault("dog_lambda_reg", 1e-3)
        self.parameters.setdefault("dog_kernel_sigma_vox", None)  # None => auto

        # Q-only stage knobs
        self.parameters.setdefault("q_window_kind", "cheb")
        self.parameters.setdefault("q_window_at_db", 100.0)
        self.parameters.setdefault("edge_guard_frac", 0.10)

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
            broadcast_into_rows=self.parameters.get("broadcast_displacement_into_rows", False),
        )

    def load_amplitudes_and_generate_grid(self, chunk_id, point_data_list, rifft_saver):
        filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
        try:
            data = rifft_saver.load_data(filename)
            amplitudes = data.get('amplitudes', None)
            if amplitudes is None:
                print(f"Amplitudes not found in {filename}")
                return np.array([]), None, None

            # Always normalize by ntotal only
            amplitudes = _normalize_amplitudes_ntotal(
                amplitudes, rifft_saver=rifft_saver, chunk_id=chunk_id,
                logger=logging.getLogger(__name__),
            )

            grids=[]; grids_shapeNd=[]; central_point_ids=[]
            for pd in point_data_list:
                grid_points, grid_shapeNd = self.point_data_processor._generate_grid(
                    chunk_id=chunk_id,
                    dimensionality=len(pd['coordinates']),
                    step_in_frac=pd['step_in_frac'],
                    central_point=pd['coordinates'],
                    dist=pd['dist_from_atom_center'],
                    central_point_id=pd['central_point_id']
                )
                grids.append(grid_points); grids_shapeNd.append(grid_shapeNd)
                central_point_ids.extend([pd['central_point_id']]*len(grid_points))

            rifft_space_grid = (
                np.hstack((np.vstack(grids), np.array(central_point_ids)[:,None])) if grids else np.array([])
            )
            return rifft_space_grid, amplitudes, grids_shapeNd
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return np.array([]), None, None

    def compute_and_save_displacements(self, *, chunk_id:int, rifft_saver, point_data_list:list[dict],
                                       output_dir:str|None=None, broadcast_into_rows:bool=False):
        log = logging.getLogger(__name__)
        if output_dir is None:
            out_by_saver = getattr(rifft_saver, "output_dir", None)
            output_dir = out_by_saver or os.path.dirname(os.path.abspath(
                rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")))
        os.makedirs(output_dir, exist_ok=True)

        # load amplitudes + grid
        fn_amp = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            d = rifft_saver.load_data(fn_amp)
            amplitudes = d.get("amplitudes", None)
            rifft_space_grid = d.get("rifft_space_grid", None)
        except FileNotFoundError:
            d={}; amplitudes=None; rifft_space_grid=None

        if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid)==0:
            rifft_space_grid2, amplitudes2, _ = self.load_amplitudes_and_generate_grid(
                chunk_id, point_data_list, rifft_saver
            )
            if amplitudes is None: amplitudes=amplitudes2
            if rifft_space_grid is None: rifft_space_grid=rifft_space_grid2
            if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid)==0:
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")

            # Enforce ntotal-only normalization (again, in case we came this path)
            amplitudes = _normalize_amplitudes_ntotal(
                amplitudes, rifft_saver=rifft_saver, chunk_id=chunk_id,
                logger=log,
            )

        # Residual ΔR column (already RIFFT of masked/total A(q) as user stated)
        Rvals_all = amplitudes[:,1] if amplitudes.ndim==2 and amplitudes.shape[1]>=2 else np.ravel(amplitudes)

        rifft_space_grid = np.asarray(rifft_space_grid)
        D_all = rifft_space_grid.shape[1]-1
        coords_all = rifft_space_grid[:,:D_all]
        ids_all    = rifft_space_grid[:,-1].astype(int)

        # id -> center, steps (Å)
        id2center={}; id2step={}
        for pd in point_data_list:
            cid=int(pd["central_point_id"])
            id2center[cid]=np.asarray(pd["coordinates"], float)[:D_all]
            # dx_dy_dz are not needed by the linear solver directly, kept for completeness
            step=pd.get("step_in_frac",None)
            if step is None: raise KeyError("step_in_frac missing in point data.")
            if np.isscalar(step): vals=[float(step)]*D_all
            else: vals=[float(step[k]) for k in range(D_all)]
            while len(vals)<3: vals.append(1.0)
            id2step[cid]=tuple(vals[:3])

        # group rows by site id
        groups=defaultdict(list)
        for i,cid in enumerate(ids_all): groups[int(cid)].append(i)

        # Shared knobs (constant across masked/unmasked runs)
        weight_g  = float(self.parameters.get("ls_weight_gamma", 0.35))
        lam_reg   = float(self.parameters.get("dog_lambda_reg", 1e-3))
        sigma_vox = self.parameters.get("dog_kernel_sigma_vox", None)
        intervals   = self.parameters["reciprocal_space_intervals_all"]
        hkl_max_xyz = compute_hkl_max_from_intervals(intervals)
        guard_frac     = float(self.parameters.get("edge_guard_frac", 0.10))
        q_window_kind  = str(self.parameters.get("q_window_kind", "cheb")).lower()
        q_window_at_db = float(self.parameters.get("q_window_at_db", 100.0))
        size_aver      = np.asarray(self.parameters["supercell"], dtype=int)

        rows=[]
        for cid, idxs in groups.items():
            idxs=np.asarray(idxs,int)
            coords=coords_all[idxs,:]; Rvals=Rvals_all[idxs]
            center=id2center.get(int(cid))
            if center is None: continue

            # Per-site *Q-only* processing (linear)
            Rvals_proc = _apply_rq_pipeline_local(
                Rvals, coords,
                q_window_kind=q_window_kind,
                q_window_at_db=q_window_at_db,
                size_aver=size_aver,
                hkl_max_xyz=hkl_max_xyz,
                guard_frac=guard_frac,
            )

            # Regrid to dense small patch and solve with linear DoG
            y_grid, shape, axes_vals, _ = _regrid_patch_to_c(coords, Rvals_proc)
            D = len(shape)

            # sigma_vox handling
            if sigma_vox is not None:
                if np.isscalar(sigma_vox):
                    sigma_use = tuple(float(sigma_vox) for _ in range(D))
                else:
                    sigma_use = tuple(float(s) for s in sigma_vox)[:D]
            else:
                sigma_use = None

            u_vec = _estimate_u_linear_dog(
                y_grid, axes_vals, center,
                D=D,
                kernel_sigma_vox=sigma_use,
                weight_gamma=weight_g,
                lambda_reg=lam_reg,
                remove_odd_tilt=True,
            )

            rows.append({"central_point_id":int(cid), "ux":float(u_vec[0]),
                         "uy":float(u_vec[1]), "uz":float(u_vec[2])})

        if not rows: raise RuntimeError("No site displacements computed.")

        ids = np.fromiter((r["central_point_id"] for r in rows), count=len(rows), dtype=np.int64)
        U   = np.stack([[r["ux"], r["uy"], r["uz"]] for r in rows], axis=0).astype(np.float64)

        out_table={
            "central_point_id": ids,
            "u": U,
            "columns": np.array(["ux","uy","uz"], dtype=object),
            "coordinate_system": np.array(["cartesian"], dtype=object),
            "units": np.array(["angstrom","angstrom","angstrom"], dtype=object),
        }
        h5_path = os.path.join(output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.h5")
        csv_path= os.path.join(output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.csv")
        rifft_saver.save_data(out_table, h5_path)
        _write_displacements_csv(csv_path, ids, U)

        if broadcast_into_rows:
            cid2u={int(i):U[k,:] for k,i in enumerate(ids)}
            urows=np.stack([cid2u[int(c)] for c in ids_all], axis=0)
            if amplitudes.ndim==1: aug=np.column_stack([amplitudes, urows])
            else: aug=np.concatenate([amplitudes, urows], axis=1)
            d_aug=dict(d); d_aug["amplitudes_with_displacement"]=aug
            d_aug["amplitudes_with_displacement_columns"]=np.array(["<orig...>","ux","uy","uz"], dtype=object)
            h5_aug=os.path.join(output_dir, f"output_chunk_{chunk_id}_amplitudes_with_displacements.h5")
            rifft_saver.save_data(d_aug, h5_aug)

        return out_table

    # legacy (unused in linear path, kept for compatibility)
    def filter_from_window(self, window0, dimensionality, window1=None, window2=None, size_aver=None):
        if dimensionality==1:
            fd0=fft(window0, np.array(self.parameters["supercell"])[0])/(len(window0)/2.0)
            k0=np.abs(fftshift(fd0/np.abs(fd0).max())); return k0/k0.sum()
        if dimensionality==2:
            fd0=fft(window0, self.parameters["supercell"][0])/(len(window0)/2.0)
            fd1=fft(window1, self.parameters["supercell"][1])/(len(window1)/2.0)
            k0=np.abs(fftshift(fd0/np.abs(fd0).max()))
            k1=np.abs(fftshift(fd1/np.abs(fd1).max()))
            kern=k0[:,None]*k1[None,:]; return kern/kern.sum()
        if dimensionality==3:
            sc=np.array(self.parameters["supercell"])
            fd0=fft(window0, sc[0])/(len(window0)/2.0)
            fd1=fft(window1, sc[1])/(len(window1)/2.0)
            fd2=fft(window2, sc[2])/(len(window2)/2.0)
            k0=np.abs(fftshift(fd0/np.abs(fd0).max()))
            k1=np.abs(fftshift(fd1/np.abs(fd1).max()))
            k2=np.abs(fftshift(fd2/np.abs(fd2).max()))
            kern=k0[:,None,None]*k1[None,:,None]*k2[None,None,:]; return kern/kern.sum()
        raise ValueError("Unsupported dimensionality")
