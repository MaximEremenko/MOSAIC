from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal.windows import chebwin


def radial_gaussian_weights_cart(shape, steps_cart, gamma=0.35):
    coords_1d = [
        (np.arange(n) - (n - 1) / 2.0) * float(steps_cart[axis])
        for axis, n in enumerate(shape)
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

    edge = np.array([max(abs(s[0]), abs(s[-1])) for s in coords_1d], float)
    r_edge = float(np.linalg.norm(edge))
    sigma = max(gamma * r_edge, 1e-12)
    return np.exp(-0.5 * (r2 / (sigma * sigma)))


def kband_vector_unshifted(nS, S, hmax, guard_frac):
    nS = int(nS)
    S = int(S)
    if nS <= 0 or S <= 0:
        raise ValueError("nS and S must be positive")

    h = np.fft.fftfreq(nS, d=1.0 / S)
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


def window_spectrum_vectors_unshifted(size_aver, kind="cheb", at_db=100.0):
    size_aver = np.asarray(size_aver, dtype=int)
    wd = []
    for axis in range(len(size_aver)):
        size = int(size_aver[axis])
        if kind == "cheb":
            window = chebwin(size, at_db, sym=False).astype(float)
        else:
            raise ValueError("kind must be 'cheb'")
        Fw = fft(window, size) / (len(window) / 2.0)
        mag_unshift = np.fft.ifftshift(
            np.abs(fftshift(Fw / (np.abs(Fw).max() or 1.0)))
        )
        wd.append(mag_unshift)
    return wd


def qspace_psf_in_r(
    size_aver,
    hkl_max_xyz,
    *,
    guard_frac,
    window_kind="cheb",
    window_at_db=100.0,
):
    size_aver = np.asarray(size_aver, dtype=int)
    dim = len(size_aver)

    wd = window_spectrum_vectors_unshifted(
        size_aver, kind=window_kind, at_db=window_at_db
    )

    h_max = float(hkl_max_xyz[0]) if dim >= 1 else 0.0
    k_max = float(hkl_max_xyz[1]) if dim >= 2 else 0.0
    l_max = float(hkl_max_xyz[2]) if dim >= 3 else 0.0

    bx = kband_vector_unshifted(size_aver[0], size_aver[0], h_max, guard_frac)
    if dim >= 2:
        by = kband_vector_unshifted(size_aver[1], size_aver[1], k_max, guard_frac)
    if dim == 3:
        bz = kband_vector_unshifted(size_aver[2], size_aver[2], l_max, guard_frac)

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


def build_feature_vector_from_patch(
    y_grid,
    axes_vals,
    center_abs,
    *,
    D,
    weight_gamma=0.35,
    remove_odd_tilt=True,
    center_patch_subvoxel,
):
    y_c, _ = center_patch_subvoxel(y_grid, axes_vals, center_abs[:D])
    flip_axes = tuple(range(D))
    O = 0.5 * (y_c - np.flip(y_c, axis=flip_axes))

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
        r2 = s0**2
    elif D == 2:
        s0, s1 = np.meshgrid(coords_1d[0], coords_1d[1], indexing="ij")
        r2 = s0 * s0 + s1 * s1
    else:
        s0, s1, s2 = np.meshgrid(
            coords_1d[0], coords_1d[1], coords_1d[2], indexing="ij"
        )
        r2 = s0 * s0 + s1 * s1 + s2 * s2

    if remove_odd_tilt:
        r = np.sqrt(r2)
        r_max = float(np.max(r)) if r.size else 1.0
        mask = (r > 0.55 * r_max).ravel("C")
        if D == 1:
            Sodd = np.vstack([s0.ravel("C")])
        elif D == 2:
            Sodd = np.vstack([s0.ravel("C"), s1.ravel("C")])
        else:
            Sodd = np.vstack([s0.ravel("C"), s1.ravel("C"), s2.ravel("C")])
        Ao = Sodd[:, mask]
        rhs = O.ravel("C")[mask]
        Ho = Ao @ Ao.T + 1e-12 * np.eye(D)
        fo = Ao @ rhs
        b_hat = np.linalg.solve(Ho, fo)
        if D == 1:
            O = O - b_hat[0] * s0
        elif D == 2:
            O = O - (b_hat[0] * s0 + b_hat[1] * s1)
        else:
            O = O - (b_hat[0] * s0 + b_hat[1] * s1 + b_hat[2] * s2)

    W = radial_gaussian_weights_cart(y_c.shape, steps, gamma=weight_gamma)
    return (np.sqrt(W) * O).ravel("C")

