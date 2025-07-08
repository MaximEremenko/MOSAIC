# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:01 2024

@author: Maksim Eremenko
"""

# strategies/shape_strategies.py
import matplotlib.pyplot as plt
import numpy as np
from interfaces.mask_strategy import IMaskStrategy

#from itertools import product

from numba import njit, prange, cuda
from typing import Dict, Any

import math, warnings
@njit
def corresponding_value(n_val, a):
    # Numba supports np.sign, np.abs
    if n_val == 0:
        return abs(a)
    else:
        return np.sign(n_val) * (abs(n_val) + abs(a))

@njit
def unique_sorted_rounded(arr):
    # arr is assumed to be sorted, we round and remove duplicates
    # First, round the array
    for i in range(len(arr)):
        arr[i] = round(arr[i], 8)

    # Remove duplicates from a sorted array
    if len(arr) == 0:
        return arr
    count = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            count += 1
    result = np.empty(count, dtype=arr.dtype)
    idx = 0
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            idx += 1
            result[idx] = arr[i]
    return result

# @njit
# def find_val_in_interval(coord_min, coord_max, a, tol=1.0):
#     # Largest integer <= coord_min
#     coord_min_int = np.int32(np.floor(coord_min))
#     # Smallest integer >= coord_max
#     coord_max_int = np.int32(np.ceil(coord_max))

#     coord_min = round(coord_min, 8)
#     coord_max = round(coord_max, 8)

#     # We'll store possible solutions in a preallocated array.
#     # max_solutions: let's assume the range is not too large. We'll pick a safe upper bound.
#     max_solutions = (coord_max_int - coord_min_int + 4)  # a bit more than needed
#     # Actually, to be safe, let's allow up to 200.
#     #max_solutions = 200
#     solutions = np.empty(max_solutions, dtype=np.float64)
#     count = 0

#     for n_val in range(coord_min_int - 2, coord_max_int + 2):
#         if n_val == 0:
#             # Check both +a and -a
#             # We'll first check the extended condition (coord_min-1 <= ... <= coord_max+1)
#             # but we only add solutions if they lie within [coord_min, coord_max]
#             a_abs = abs(a)
#             neg_a = -a_abs
#             pos_a = a_abs
#             # If both +a and -a fit in expanded range:
#             if (coord_min - 1 <= a <= coord_max + 1) and (coord_min - 1 <= -a <= coord_max + 1):
#                 # Check if they lie within the stricter [coord_min, coord_max]
#                 if (coord_min - tol) <= neg_a <= (coord_max + tol):
#                     solutions[count] = neg_a
#                     count += 1
#                 if (coord_min - tol) <= pos_a <= (coord_max + tol):
#                     solutions[count] = pos_a
#                     count += 1
#             else:
#                 # Check individually
#                 if (coord_min) <= pos_a <= (coord_max):
#                     solutions[count] = pos_a
#                     count += 1
#                 if (coord_min) <= neg_a <= (coord_max):
#                     solutions[count] = neg_a
#                     count += 1
#         else:
#             expr_val = corresponding_value(n_val, a)
#             if (coord_min) <= expr_val <= (coord_max):
#                 solutions[count] = expr_val
#                 count += 1

#     # Slice to actual count
#     solutions = solutions[:count]
#     # Sort solutions for uniqueness
#     if count > 1:
#         solutions = np.sort(solutions)
#     # Unique and round
#     solutions = unique_sorted_rounded(solutions)
#     return solutions
@njit
def find_val_in_interval(coord_min: float,
                         coord_max: float,
                         a: float,
                         tol: float = 1.0) -> np.ndarray:
    """
    Same logic as your original helper, but written to be numba-friendly
    (only NumPy scalars/arrays, no Python lists) so it can be called
    inside a parallel region.
    """
    lo_int = np.int32(np.floor(coord_min))
    hi_int = np.int32(np.ceil(coord_max))

    coord_min = round(coord_min, 8)
    coord_max = round(coord_max, 8)

    # generous upper bound on number of candidates
    max_sol = hi_int - lo_int + 4
    sols = np.empty(max_sol, dtype=np.float64)
    n = 0

    for n_val in range(lo_int - 2, hi_int + 2):

        if n_val == 0:
            a_abs = abs(a)
            for v in (-a_abs, a_abs):
                if (coord_min - tol) <= v <= (coord_max + tol):
                    sols[n] = v
                    n += 1
        else:
            expr = np.sign(n_val) * (abs(n_val) + abs(a))
            if coord_min <= expr <= coord_max:
                sols[n] = expr
                n += 1

    if n == 0:
        return sols[:0]

    # sort & unique (array is small – O(n²) is fine)
    sols = sols[:n]
    for i in range(n):
        sols[i] = round(sols[i], 8)
    sols.sort()

    # unique in-place
    w = 1
    for i in range(1, n):
        if sols[i] != sols[w - 1]:
            sols[w] = sols[i]
            w += 1
    return sols[:w]

@njit
def flatten_candidate_centers(sols_x, sols_y, sols_z):
    """
    Given three 1D arrays of candidate coordinates, builds an (L, 3) array
    of candidate centers for L = len(sols_x)*len(sols_y)*len(sols_z).
    """
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
    """
    data_points:    (N, 3) array of point coordinates.
    special_radii:  (M,) array of radii.
    special_coords: (M, 3) array of sphere center coordinates.
    coord_min:      (3,) lower bounds for periodic domain.
    coord_max:      (3,) upper bounds for periodic domain.

    Returns:
      mask: Boolean array of length N indicating whether each point is inside
            any sphere (including periodic copies).
    """
    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)
    
    # Loop over each special sphere (usually only one or two).
    for m in range(special_radii.shape[0]):
        radius = special_radii[m]
        radius_sq = radius * radius  # use squared radius for comparison
        base_coord = special_coords[m]
        
        # Compute the periodic solutions for x, y, and z.
        sols_x = find_val_in_interval(coord_min[0], coord_max[0], base_coord[0])
        sols_y = find_val_in_interval(coord_min[1], coord_max[1], base_coord[1])
        sols_z = find_val_in_interval(coord_min[2], coord_max[2], base_coord[2])
        
        # Flatten the three nested loops into one array of candidate centers.
        candidate_centers = flatten_candidate_centers(sols_x, sols_y, sols_z)
        L = candidate_centers.shape[0]
        
        # Parallel loop over all data points.
        for p in prange(N):
            x = data_points[p, 0]
            y = data_points[p, 1]
            z = data_points[p, 2]
            inside = False
            # Check the point against every candidate sphere center.
            for c in range(L):
                dx = x - candidate_centers[c, 0]
                dy = y - candidate_centers[c, 1]
                dz = z - candidate_centers[c, 2]
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    inside = True
                    break  # No need to check further candidates.
            if inside:
                mask[p] = True
    return mask
# @njit
# def compute_mask(data_points, special_radii, special_coords, coord_min, coord_max):
#     # data_points: (N,3)
#     # special_radii: (M,)
#     # special_coords: (M,3)
#     # Returns a boolean mask of length N.

#     N = data_points.shape[0]
#     mask = np.zeros(N, dtype=np.bool_)

#     for m in range(special_radii.shape[0]):
#         radius = special_radii[m]
#         base_coord = special_coords[m]

#         solutions_x = find_val_in_interval(coord_min[0], coord_max[0], base_coord[0])
#         solutions_y = find_val_in_interval(coord_min[1], coord_max[1], base_coord[1])
#         solutions_z = find_val_in_interval(coord_min[2], coord_max[2], base_coord[2])

#         # Nested loops instead of product
#         for i in range(len(solutions_x)):
#             cx = solutions_x[i]
#             for j in range(len(solutions_y)):
#                 cy = solutions_y[j]
#                 for k in range(len(solutions_z)):
#                     cz = solutions_z[k]

#                     # Compute distances: (x - cx)^2 + (y - cy)^2 + (z - cz)^2
#                     dx = data_points[:, 0] - cx
#                     dy = data_points[:, 1] - cy
#                     dz = data_points[:, 2] - cz
#                     dist_sq = dx*dx + dy*dy + dz*dz
#                     # Compare with radius^2 to avoid sqrt
#                     within = dist_sq <= radius*radius

#                     # Update mask
#                     for idx in range(N):
#                         if within[idx]:
#                             mask[idx] = True

#     return mask

class SphereShapeStrategy(IMaskStrategy):
    def __init__(self, spetial_points_param: Dict[str, Any]):
        self.spetial_points_param = spetial_points_param

    def generate_mask(self, data: np.ndarray) -> np.ndarray:
        coord_min = np.min(data, axis=0)
        coord_max = np.max(data, axis=0)

        # Extract special points info into Numba-friendly arrays
        specialPoints = self.spetial_points_param["specialPoints"]
        M = len(specialPoints)
        special_radii = np.empty(M, dtype=np.float64)
        special_coords = np.empty((M, 3), dtype=np.float64)
        for i, sp in enumerate(specialPoints):
            special_radii[i] = sp["radius"]
            special_coords[i, :] = sp["coordinate"]

        # Call the Numba-optimized function
        mask = compute_mask(data, special_radii, special_coords, coord_min, coord_max)
        return mask

    
        
class EllipsoidShapeStrategy(IMaskStrategy):
    def __init__(self, spetial_points: np.ndarray, axes: np.ndarray, theta: float, phi: float):
        self.axes = axes
        self.rotation_matrix = self._create_rotation_matrix(theta, phi)
        self.spetial_points = spetial_points
        
    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        """
        Generates a mask for points within multiple ellipsoids.

        Args:
            data_points (np.ndarray): Data points array.
            spetial_points (np.ndarray): Array of centers of the ellipsoids.

        Returns:
            np.ndarray: Boolean mask array.
        """
        mask = np.zeros(len(data_points), dtype=bool)
        for spetial_point in self.spetial_points:
            shifted_points = data_points - spetial_point
            rotated_points = shifted_points @ self.rotation_matrix.T
            scaled_points = rotated_points / self.axes
            distances = np.sum(scaled_points**2, axis=1)
            mask |= distances <= 1.0
        return mask

    def _create_rotation_matrix(self, theta: float, phi: float) -> np.ndarray:
        """
        Creates a rotation matrix based on theta and phi angles.

        Args:
            theta (float): Rotation angle theta in radians.
            phi (float): Rotation angle phi in radians.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        R_phi = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi),  np.cos(phi), 0],
            [0,            0,           1]
        ])
        R_theta = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0,             1, 0            ],
            [-np.sin(theta),0, np.cos(theta)]
        ])
        return R_theta @ R_phi
    
# @njit
# def compute_1d_mask(data_points, radii, centers, coord_min, coord_max):
#     """
#     Computes a 1D mask for points within multiple intervals.

#     Args:
#         data_points (np.ndarray): Array of shape (N,).
#         radii (np.ndarray): Radii of the intervals, shape (M,). Here, radius corresponds to half the interval length.
#         centers (np.ndarray): Interval centers, shape (M,).
#         coord_min (float): Minimum coordinate bound.
#         coord_max (float): Maximum coordinate bound.

#     Returns:
#         np.ndarray: Boolean mask of shape (N,).
#     """
#     N = data_points.shape[0]
#     M = radii.shape[0]
#     mask = np.zeros(N, dtype=np.bool_)

#     for m in range(M):
#         radius = radii[m]
#         center = centers[m]

#         # Define step size based on radius for sampling points
#         step = radius / 2.0

#         solutions = find_val_in_interval(coord_min, coord_max, center)

#         for c in solutions:
#             dx = np.abs(data_points[:, 0] - c)
#             dist = dx
#             within = (dist <= radius) 
#             mask |= within

#     return mask
@njit(parallel=True)
def compute_1d_mask(data_points: np.ndarray,
                    radii:       np.ndarray,
                    centers:     np.ndarray,
                    coord_min:   float,
                    coord_max:   float) -> np.ndarray:
    """
    Parallel-friendly version.

    Parameters
    ----------
    data_points : (N,) float64
        Coordinates of the points to test.
    radii, centers : (M,) float64
        Interval half–lengths **r** and their centers **c**.
    coord_min, coord_max : float
        Bounds of the periodic box.

    Returns
    -------
    mask : (N,) bool
        True for every point that falls in at least one interval.
    """
    N = data_points.shape[0]
    M = radii.shape[0]

    # one row per interval – each thread owns exactly one row
    tmp = np.zeros((M, N), dtype=np.bool_)

    for m in prange(M):                           # ← parallel loop
        r  = radii[m]
        cs = find_val_in_interval(coord_min, coord_max, centers[m])

        row = tmp[m]                              # private view
        for k in range(cs.size):                  # usually ≤ 3
            row |= np.abs(data_points - cs[k]) <= r

    # OR-reduce across intervals (serial, negligible cost)
    mask = np.zeros(N, dtype=np.bool_)
    for m in range(M):
        mask |= tmp[m]

    return mask  
# @njit
# def compute_2d_mask(data_points, radii, centers, coord_min, coord_max):
#     """
#     Computes a 2D mask for points within multiple circles.

#     Args:
#         data_points (np.ndarray): Array of shape (N, 2).
#         radii (np.ndarray): Radii of the circles, shape (M,).
#         centers (np.ndarray): Circle centers, shape (M, 2).
#         coord_min (np.ndarray): Minimum coordinate bounds (2,).
#         coord_max (np.ndarray): Maximum coordinate bounds (2,).

#     Returns:
#         np.ndarray: Boolean mask of shape (N,).
#     """
#     N = data_points.shape[0]
#     mask = np.zeros(N, dtype=np.bool_)


#     for m in range(radii.shape[0]):
#         radius = radii[m]
#         center = centers[m]

#         solutions_x = find_val_in_interval(coord_min[0], coord_max[0], center[0])
#         solutions_y = find_val_in_interval(coord_min[1], coord_max[1], center[1])

#         #solutions_x = np.arange(coord_min[0], coord_max[0] + 0.001, radius / 2)
#         #solutions_y = np.arange(coord_min[1], coord_max[1] + 0.001, radius / 2)

#         for cx in solutions_x:
#             for cy in solutions_y:
#                 dx = data_points[:, 0] - cx
#                 dy = data_points[:, 1] - cy
#                 dist_sq = dx * dx + dy * dy
#                 within = dist_sq <= radius * radius
#                 mask |= within

#     return mask

# @njit
# def _expand_centres(centres, radii, coord_min, coord_max):
#     """
#     For each input circle add the periodic images that might overlap the
#     simulation cell.  Returns flat arrays (K,2) and (K,) where K ≤ 9 × M.
#     """
#     xs_all = []
#     ys_all = []
#     rs_all = []

#     for m in range(radii.size):
#         xs = find_val_in_interval(coord_min[0], coord_max[0], centres[m, 0])
#         ys = find_val_in_interval(coord_min[1], coord_max[1], centres[m, 1])

#         for x in xs:
#             for y in ys:
#                 xs_all.append(x)
#                 ys_all.append(y)
#                 rs_all.append(radii[m])

#     out_xy = np.empty((len(xs_all), 2), dtype=np.float32)
#     out_xy[:, 0] = np.asarray(xs_all)
#     out_xy[:, 1] = np.asarray(ys_all)
#     out_r  = np.asarray(rs_all, dtype=np.float32)

#     return out_xy, out_r
@njit(parallel=True)
def _expand_centres(centres, radii, coord_min, coord_max):
    """
    Parallel version.
    For each input circle emit all periodic images that can overlap the
    simulation cell.  Output shapes:
        out_xy : (K, 2)  float32
        out_r  : (K,)    float32
    where K ≤ 9 × M.
    """
    M = radii.size

    # ── 1st pass: count how many images each circle produces ────────────
    counts = np.empty(M, dtype=np.int64)
    for m in range(M):
        xs = find_val_in_interval(coord_min[0], coord_max[0], centres[m, 0])
        ys = find_val_in_interval(coord_min[1], coord_max[1], centres[m, 1])
        counts[m] = xs.size * ys.size      # ≤ 9

    total = counts.sum()                   # global K

    # prefix sum → start index of each circle’s chunk
    offsets = np.empty(M + 1, dtype=np.int64)
    offsets[0] = 0
    for m in range(M):
        offsets[m + 1] = offsets[m] + counts[m]

    # ── allocate outputs once ───────────────────────────────────────────
    out_xy = np.empty((total, 2), dtype=np.float32)
    out_r  = np.empty(total,       dtype=np.float32)

    # ── 2nd pass: parallel fill  ───────────────────────────────────────
    for m in prange(M):
        xs = find_val_in_interval(coord_min[0], coord_max[0], centres[m, 0])
        ys = find_val_in_interval(coord_min[1], coord_max[1], centres[m, 1])

        idx = offsets[m]
        for i in range(xs.size):           # at most 3
            for j in range(ys.size):       # at most 3
                out_xy[idx, 0] = xs[i]
                out_xy[idx, 1] = ys[j]
                out_r[idx]     = radii[m]
                idx += 1

    return out_xy, out_r

@cuda.jit
def _mask_kernel(points, centres, radii, out_mask):
    """
    One thread ⇝ one data point.
    points  : (N,2)  float64
    centres : (K,2)  float64
    radii   : (K,)   float64
    out_mask: (N,)   uint8
    """
    i = cuda.grid(1)
    if i >= points.shape[0]:
        return

    x, y = points[i, 0], points[i, 1]

    hit = 0
    for j in range(centres.shape[0]):          # stays in registers
        dx = x - centres[j, 0]
        dy = y - centres[j, 1]
        if dx*dx + dy*dy <= radii[j]*radii[j]:
            hit = 1
            break                              # first match is enough
    out_mask[i] = hit
    
def compute_2d_mask_gpu(data_points: np.ndarray,   # (N,2)
                        radii:       np.ndarray,   # (M,)
                        centres:     np.ndarray,   # (M,2)
                        coord_min:   np.ndarray,   # (2,)
                        coord_max:   np.ndarray    # (2,)
                        ) -> np.ndarray:
    """
    GPU-accelerated Boolean mask.
    """
    # 1) build the periodic images on CPU (cheap, small arrays)
    exp_centres, exp_radii = _expand_centres(centres, radii,
                                             coord_min, coord_max)

    # 2) move everything to the device
    d_points  = cuda.to_device(data_points.astype(np.float64))
    d_centre  = cuda.to_device(exp_centres)        # (K,2)
    d_radii   = cuda.to_device(exp_radii)          # (K,)
    d_mask    = cuda.device_array(data_points.shape[0], dtype=np.uint8)

    # 3) launch – one thread per point
    threads_per_block = 128
    blocks_per_grid   = math.ceil(data_points.shape[0] / threads_per_block)

    _mask_kernel[blocks_per_grid, threads_per_block](
        d_points, d_centre, d_radii, d_mask
    )

    # 4) copy back, cast to bool
    return d_mask.copy_to_host().view(np.bool_)

@njit(parallel=True, fastmath=True)
def compute_2d_mask_cpu(data_points: np.ndarray,     # (N, 2)
                    radii:       np.ndarray,     # (M,)
                    centers:     np.ndarray,     # (M, 2)
                    coord_min:   np.ndarray,     # (2,)
                    coord_max:   np.ndarray      # (2,)
                    ) -> np.ndarray:
    """
    Parallel version of `compute_2d_mask`.

    Returns
    -------
    mask : (N,) bool – True if a point is inside ≥1 circle
    """
    N = data_points.shape[0]
    M = radii.size

    # one private row per circle
    tmp = np.zeros((M, N), dtype=np.bool_)

    for m in prange(M):                                   # ← parallel
        r  = radii[m]
        cx, cy = centers[m]

        xs = find_val_in_interval(coord_min[0], coord_max[0], cx)
        ys = find_val_in_interval(coord_min[1], coord_max[1], cy)

        row = tmp[m]
        r2  = r * r
        for ix in range(xs.size):
            dx = data_points[:, 0] - xs[ix]               # vectorised
            dx2 = dx * dx
            for iy in range(ys.size):
                dy  = data_points[:, 1] - ys[iy]
                within = dx2 + dy * dy <= r2
                row |= within                             # local write

    # OR-reduce across all circles (serial, cheap)
    mask = np.zeros(N, dtype=np.bool_)
    for m in range(M):
        mask |= tmp[m]
    return mask

#     return mask
# ── 2.  Public dispatching API ────────────────────────────────────────
def compute_2d_mask(data_points: np.ndarray,
                    radii:       np.ndarray,
                    centres:     np.ndarray,
                    coord_min:   np.ndarray,
                    coord_max:   np.ndarray,
                    *,
                    gpu_threshold: int = 20_000,
                    prefer_float32: bool = True) -> np.ndarray:
    """
    Front-door convenience wrapper.
    Tries the GPU first (if available & worth it); otherwise CPU.
    """
    # --- cast once according to precision preference --------------------
    dt = np.float32 if prefer_float32 else np.float64
    pts = data_points.astype(dt, copy=False)
    rad = radii.astype(dt, copy=False)
    ctr = centres.astype(dt, copy=False)

    # --- decide on GPU vs CPU ------------------------------------------
    gpu_wanted = cuda.is_available() and pts.shape[0] >= gpu_threshold

    if gpu_wanted:
        try:
            return compute_2d_mask_gpu(pts, rad, ctr, coord_min, coord_max)
        except cuda.cudadrv.error.CudaSupportError as err:
            warnings.warn(f"CUDA problem ({err}); falling back to CPU.")

    # --- CPU fallback ---------------------------------------------------
    return compute_2d_mask_cpu(pts, rad, ctr, coord_min, coord_max)


# ───────────────────────── 1. periodic expansion (CPU, parallel) ────────
@njit(parallel=True)
def _expand_centres3d(centres, radii, cmin, cmax):
    """
    centres : (M,3)  radii : (M,)
    Return all periodic images that can overlap the simulation box.
      out_xyz : (K,3) float64   out_r : (K,) float64
    K ≤ 27 × M  (3 images per axis)
    """
    M = radii.size

    # pass 1 – count images per sphere
    counts = np.empty(M, np.int64)
    for m in range(M):
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])
        counts[m] = xs.size * ys.size * zs.size    # ≤ 27

    K = counts.sum()

    # prefix sum → offsets
    offsets = np.empty(M + 1, np.int64)
    offsets[0] = 0
    for m in range(M):
        offsets[m + 1] = offsets[m] + counts[m]

    out_xyz = np.empty((K, 3), np.float64)
    out_r   = np.empty(K,       np.float64)

    # pass 2 – fill (parallel)
    for m in prange(M):
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])

        idx = offsets[m]
        for i in range(xs.size):
            for j in range(ys.size):
                for k in range(zs.size):
                    out_xyz[idx, 0] = xs[i]
                    out_xyz[idx, 1] = ys[j]
                    out_xyz[idx, 2] = zs[k]
                    out_r[idx]      = radii[m]
                    idx += 1

    return out_xyz, out_r


# ───────────────────────── 2. CUDA kernel (thread ↦ point) ──────────────
@cuda.jit
def _mask_kernel3d(points, centres, radii, out_mask):
    """
    points  : (N,3)  centres : (K,3)  radii : (K,)  out_mask : (N,)
    """
    p = cuda.grid(1)
    if p >= points.shape[0]:
        return

    x, y, z = points[p, 0], points[p, 1], points[p, 2]
    hit = 0
    for c in range(centres.shape[0]):
        dx = x - centres[c, 0]
        dy = y - centres[c, 1]
        dz = z - centres[c, 2]
        if dx*dx + dy*dy + dz*dz <= radii[c]*radii[c]:
            hit = 1
            break
    out_mask[p] = hit


# ───────────────────────── 3. public GPU wrapper  ───────────────────────
def compute_3d_mask_gpu(data_points: np.ndarray,   # (N,3)
                        radii:       np.ndarray,   # (M,)
                        centres:     np.ndarray,   # (M,3)
                        coord_min:   np.ndarray,   # (3,)
                        coord_max:   np.ndarray    # (3,)
                        ) -> np.ndarray:
    """
    GPU-accelerated Boolean mask for spheres with periodic images.
    """
    # build periodic images (CPU, fast)
    exp_centres, exp_r = _expand_centres3d(centres, radii,
                                           coord_min, coord_max)

    # move to device
    d_pts  = cuda.to_device(data_points.astype(np.float64))
    d_ctr  = cuda.to_device(exp_centres)
    d_rad  = cuda.to_device(exp_r)
    d_mask = cuda.device_array(data_points.shape[0], dtype=np.uint8)

    # launch kernel – 1 thread per point
    TPB    = 128
    blocks = math.ceil(data_points.shape[0] / TPB)
    _mask_kernel3d[blocks, TPB](d_pts, d_ctr, d_rad, d_mask)

    return d_mask.copy_to_host().view(np.bool_)


# ───────────────────────── 4. optional CPU fallback  ────────────────────
@njit(parallel=True, fastmath=True)
def compute_3d_mask_cpu(data_points, radii, centres, cmin, cmax):
    """
    Fully threaded CPU version (similar logic, slower than GPU).
    """
    N, M = data_points.shape[0], radii.size
    mask = np.zeros(N, np.bool_)

    for m in prange(M):
        r2 = radii[m]*radii[m]
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])
        for p in range(N):                # each thread iterates its slice
            hit = False
            for i in range(xs.size):
                dx = data_points[p, 0] - xs[i]
                dx2 = dx*dx
                for j in range(ys.size):
                    dy = data_points[p, 1] - ys[j]
                    for k in range(zs.size):
                        dz = data_points[p, 2] - zs[k]
                        if dx2 + dy*dy + dz*dz <= r2:
                            hit = True
                            break
                    if hit: break
                if hit: break
            if hit:
                mask[p] = True
    return mask


# ───────────────────────── 5. dispatcher (GPU first, else CPU) ──────────
def compute_3d_mask(data_points, radii, centres,
                    coord_min, coord_max,
                    gpu_threshold=20_000):
    """
    Call this everywhere; it picks the fastest backend automatically.
    """
    if cuda.is_available() and data_points.shape[0] >= gpu_threshold:
        try:
            return compute_3d_mask_gpu(data_points, radii, centres,
                                       coord_min, coord_max)
        except cuda.cudadrv.error.CudaSupportError as e:
            warnings.warn(f"CUDA unavailable ({e}); falling back to CPU")

    return compute_3d_mask_cpu(data_points, radii, centres,
                               coord_min, coord_max)

class CircleShapeStrategy(IMaskStrategy):
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param

    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        coord_min = np.min(data_points, axis=0)-1
        coord_max = np.max(data_points, axis=0)+1

        special_points = self.special_points_param["specialPoints"]
        num_circles = len(special_points)

        radii = np.empty(num_circles, dtype=np.float64)
        centers = np.empty((num_circles, 2), dtype=np.float64)
        for i, sp in enumerate(special_points):
            radii[i] = sp["radius"]
            centers[i, :] = sp["coordinate"]

        mask = compute_2d_mask(data_points, radii, centers, coord_min, coord_max)
        filtered_data = data_points[mask] 
        
        # plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c='blue', marker='o')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Filtered Points (mask)')
        # plt.grid(True)
        # plt.show()
        return mask
    
class IntervalShapeStrategy(IMaskStrategy):
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param
        
    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        """
        Applies a 1D interval mask to the data points.

        Args:
            data_points (np.ndarray): Array of shape (N,).
            central_points (np.ndarray): Array of central points, shape (M,).

        Returns:
            np.ndarray: Boolean mask of shape (N,).
        """
    
        coord_min = np.min(data_points) - 1
        coord_max = np.max(data_points) + 1

        special_points = self.special_points_param["specialPoints"]
        num_intervals = len(special_points)

        radii = np.empty(num_intervals, dtype=np.float64)
        centers = np.empty(num_intervals, dtype=np.float64)
        for i, sp in enumerate(special_points):
            radii[i] = sp["radius"]
            centers[i] = np.array(sp["coordinate"])#self.central_points[i]

        mask = compute_1d_mask(data_points, radii, centers, coord_min, coord_max)
        filtered_data = data_points[mask]
        
        # Plotting the filtered 1D points
        plt.figure(figsize=(10, 2))
        plt.scatter(filtered_data[:,0], np.zeros_like(filtered_data[:,0]), c='green', marker='o', label='Filtered Points')
        plt.xlabel('X-axis')
        plt.title('Filtered Points Within Intervals')
        plt.yticks([])  # Hide y-axis ticks for 1D visualization
        plt.legend()
        plt.grid(True)
        plt.show()
        return mask