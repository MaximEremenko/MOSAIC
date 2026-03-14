from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.domain.masking.shape_math import find_val_in_interval


@njit
def flatten_candidate_centers(sols_x, sols_y, sols_z):
    nx = sols_x.shape[0]
    ny = sols_y.shape[0]
    nz = sols_z.shape[0]
    L = nx * ny * nz
    centers = np.empty((L, 3), dtype=sols_x.dtype)
    idx = 0
    for i in range(nx):
        cx = sols_x[i]
        for j in range(ny):
            cy = sols_y[j]
            for k in range(nz):
                cz = sols_z[k]
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                centers[idx, 2] = cz
                idx += 1
    return centers


@njit(parallel=True)
def compute_mask(data_points, special_radii, special_coords, coord_min, coord_max):
    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)

    for m in range(special_radii.shape[0]):
        radius = special_radii[m]
        radius_sq = radius * radius
        base_coord = special_coords[m]

        sols_x = find_val_in_interval(coord_min[0], coord_max[0], base_coord[0])
        sols_y = find_val_in_interval(coord_min[1], coord_max[1], base_coord[1])
        sols_z = find_val_in_interval(coord_min[2], coord_max[2], base_coord[2])
        candidate_centers = flatten_candidate_centers(sols_x, sols_y, sols_z)
        L = candidate_centers.shape[0]

        for p in prange(N):
            x = data_points[p, 0]
            y = data_points[p, 1]
            z = data_points[p, 2]
            inside = False
            for c in range(L):
                dx = x - candidate_centers[c, 0]
                dy = y - candidate_centers[c, 1]
                dz = z - candidate_centers[c, 2]
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    inside = True
                    break
            if inside:
                mask[p] = True
    return mask


@njit
def compute_1d_mask(data_points, radii, centers, coord_min, coord_max):
    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)

    for m in range(radii.shape[0]):
        radius = radii[m]
        center = centers[m]
        solutions = find_val_in_interval(coord_min, coord_max, center)

        for c in solutions:
            dx = np.abs(data_points[:, 0] - c)
            within = dx <= radius
            mask |= within

    return mask


@njit
def compute_2d_mask(data_points, radii, centers, coord_min, coord_max):
    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)

    for m in range(radii.shape[0]):
        radius = radii[m]
        center = centers[m]

        solutions_x = find_val_in_interval(coord_min[0], coord_max[0], center[0])
        solutions_y = find_val_in_interval(coord_min[1], coord_max[1], center[1])

        for cx in solutions_x:
            for cy in solutions_y:
                dx = data_points[:, 0] - cx
                dy = data_points[:, 1] - cy
                dist_sq = dx * dx + dy * dy
                within = dist_sq <= radius * radius
                mask |= within

    return mask
