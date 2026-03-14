from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32


MAX_SOL = 256 * 16


@cuda.jit(device=True)
def corresponding_value_dev(n_val, a):
    if n_val == 0:
        return abs(a)
    return math.copysign(abs(n_val) + abs(a), n_val)


@cuda.jit(device=True)
def round8(x):
    return math.floor(x * 1e8 + 0.5) * 1e-8


@cuda.jit(device=True)
def find_val_in_interval_dev(cmin, cmax, a, tol, out):
    cmin_i = int(math.floor(cmin))
    cmax_i = int(math.ceil(cmax))
    cmin = round8(cmin)
    cmax = round8(cmax)

    cnt = 0
    for n_val in range(cmin_i - 2, cmax_i + 2):
        if n_val == 0:
            a_abs = abs(a)
            for val in (-a_abs, a_abs):
                if (cmin - tol) <= val <= (cmax + tol):
                    if cnt < MAX_SOL:
                        out[cnt] = val
                        cnt += 1
        else:
            expr_val = corresponding_value_dev(n_val, a)
            if cmin <= expr_val <= cmax:
                if cnt < MAX_SOL:
                    out[cnt] = expr_val
                    cnt += 1

    for i in range(cnt):
        out[i] = round8(out[i])
        j = i
        while j > 0 and out[j] < out[j - 1]:
            tmp = out[j]
            out[j] = out[j - 1]
            out[j - 1] = tmp
            j -= 1

    k = 1
    if cnt > 0:
        last = out[0]
        for i in range(1, cnt):
            if out[i] != last:
                out[k] = out[i]
                last = out[i]
                k += 1
    return k


@cuda.jit
def mask_kernel(points, radii, centers, cmin, cmax, mask):
    i = cuda.grid(1)
    if i >= points.shape[0]:
        return

    x = points[i, 0]
    y = points[i, 1]
    sol_x = cuda.local.array(shape=MAX_SOL, dtype=float32)
    sol_y = cuda.local.array(shape=MAX_SOL, dtype=float32)

    M = radii.shape[0]
    for m in range(M):
        r = radii[m]
        cx0 = centers[m, 0]
        cy0 = centers[m, 1]

        cnt_x = find_val_in_interval_dev(cmin[0], cmax[0], cx0, 1.0, sol_x)
        cnt_y = find_val_in_interval_dev(cmin[1], cmax[1], cy0, 1.0, sol_y)

        r2 = r * r
        inside = False
        for ix in range(cnt_x):
            dx = x - sol_x[ix]
            dx2 = dx * dx
            for iy in range(cnt_y):
                dy = y - sol_y[iy]
                if dx2 + dy * dy <= r2:
                    inside = True
                    break
            if inside:
                break

        if inside:
            mask[i] = True
            return


def compute_2d_mask_gpu(points, radii, centers, coord_min, coord_max, threads_per_block=256):
    N = points.shape[0]
    d_points = cuda.to_device(points.astype(np.float32))
    d_radii = cuda.to_device(radii.astype(np.float32))
    d_centers = cuda.to_device(centers.astype(np.float32))
    d_cmin = cuda.to_device(coord_min.astype(np.float32))
    d_cmax = cuda.to_device(coord_max.astype(np.float32))
    d_mask = cuda.device_array(N, dtype=np.bool_)

    blocks = (N + threads_per_block - 1) // threads_per_block
    mask_kernel[blocks, threads_per_block](
        d_points,
        d_radii,
        d_centers,
        d_cmin,
        d_cmax,
        d_mask,
    )
    return d_mask.copy_to_host()

