# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:01 2024

@author: Maksim Eremenko
"""

# strategies/shape_strategies.py
import matplotlib.pyplot as plt
import numpy as np
from interfaces.shape_strategy import ShapeStrategy

from itertools import product
import numpy as np

from numba import njit, prange
from typing import Dict, Any

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

@njit
def find_val_in_interval(coord_min, coord_max, a):
    # Largest integer <= coord_min
    coord_min_int = np.int32(np.floor(coord_min))
    # Smallest integer >= coord_max
    coord_max_int = np.int32(np.ceil(coord_max))

    coord_min = round(coord_min, 8)
    coord_max = round(coord_max, 8)

    # We'll store possible solutions in a preallocated array.
    # max_solutions: let's assume the range is not too large. We'll pick a safe upper bound.
    max_solutions = (coord_max_int - coord_min_int + 4)  # a bit more than needed
    # Actually, to be safe, let's allow up to 200.
    max_solutions = 200
    solutions = np.empty(max_solutions, dtype=np.float64)
    count = 0

    for n_val in range(coord_min_int - 1, coord_max_int + 2):
        if n_val == 0:
            # Check both +a and -a
            # We'll first check the extended condition (coord_min-1 <= ... <= coord_max+1)
            # but we only add solutions if they lie within [coord_min, coord_max]
            a_abs = abs(a)
            neg_a = -a_abs
            pos_a = a_abs
            # If both +a and -a fit in expanded range:
            if (coord_min - 1 <= a <= coord_max + 1) and (coord_min - 1 <= -a <= coord_max + 1):
                # Check if they lie within the stricter [coord_min, coord_max]
                if coord_min <= neg_a <= coord_max:
                    solutions[count] = neg_a
                    count += 1
                if coord_min <= pos_a <= coord_max:
                    solutions[count] = pos_a
                    count += 1
            else:
                # Check individually
                if coord_min <= pos_a <= coord_max:
                    solutions[count] = pos_a
                    count += 1
                if coord_min <= neg_a <= coord_max:
                    solutions[count] = neg_a
                    count += 1
        else:
            expr_val = corresponding_value(n_val, a)
            if coord_min <= expr_val <= coord_max:
                solutions[count] = expr_val
                count += 1

    # Slice to actual count
    solutions = solutions[:count]
    # Sort solutions for uniqueness
    if count > 1:
        solutions = np.sort(solutions)
    # Unique and round
    solutions = unique_sorted_rounded(solutions)
    return solutions
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

class SphereShapeStrategy(ShapeStrategy):
    def __init__(self, spetial_points_param: Dict[str, Any]):
        self.spetial_points_param = spetial_points_param

    def apply(self, data_points: np.ndarray, spetial_points: np.ndarray) -> np.ndarray:
        coord_min = np.min(data_points, axis=0)
        coord_max = np.max(data_points, axis=0)

        # Extract special points info into Numba-friendly arrays
        specialPoints = self.spetial_points_param["specialPoints"]
        M = len(specialPoints)
        special_radii = np.empty(M, dtype=np.float64)
        special_coords = np.empty((M, 3), dtype=np.float64)
        for i, sp in enumerate(specialPoints):
            special_radii[i] = sp["radius"]
            special_coords[i, :] = sp["coordinate"]

        # Call the Numba-optimized function
        mask = compute_mask(data_points, special_radii, special_coords, coord_min, coord_max)
        return mask

    
        
class EllipsoidShapeStrategy(ShapeStrategy):
    def __init__(self, axes: np.ndarray, theta: float, phi: float):
        self.axes = axes
        self.rotation_matrix = self._create_rotation_matrix(theta, phi)

    def apply(self, data_points: np.ndarray, spetial_points: np.ndarray) -> np.ndarray:
        """
        Generates a mask for points within multiple ellipsoids.

        Args:
            data_points (np.ndarray): Data points array.
            spetial_points (np.ndarray): Array of centers of the ellipsoids.

        Returns:
            np.ndarray: Boolean mask array.
        """
        mask = np.zeros(len(data_points), dtype=bool)
        for spetial_point in spetial_points:
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
    
@njit
def compute_1d_mask(data_points, radii, centers, coord_min, coord_max):
    """
    Computes a 1D mask for points within multiple intervals.

    Args:
        data_points (np.ndarray): Array of shape (N,).
        radii (np.ndarray): Radii of the intervals, shape (M,). Here, radius corresponds to half the interval length.
        centers (np.ndarray): Interval centers, shape (M,).
        coord_min (float): Minimum coordinate bound.
        coord_max (float): Maximum coordinate bound.

    Returns:
        np.ndarray: Boolean mask of shape (N,).
    """
    N = data_points.shape[0]
    M = radii.shape[0]
    mask = np.zeros(N, dtype=np.bool_)

    for m in range(M):
        radius = radii[m]
        center = centers[m]

        # Define step size based on radius for sampling points
        step = radius / 2.0

        solutions = find_val_in_interval(coord_min, coord_max, center)

        for c in solutions:
            dx = data_points[:, 0] - c
            dist = dx
            within = (dist <= radius) 
            mask |= within

    return mask
   
@njit
def compute_2d_mask(data_points, radii, centers, coord_min, coord_max):
    """
    Computes a 2D mask for points within multiple circles.

    Args:
        data_points (np.ndarray): Array of shape (N, 2).
        radii (np.ndarray): Radii of the circles, shape (M,).
        centers (np.ndarray): Circle centers, shape (M, 2).
        coord_min (np.ndarray): Minimum coordinate bounds (2,).
        coord_max (np.ndarray): Maximum coordinate bounds (2,).

    Returns:
        np.ndarray: Boolean mask of shape (N,).
    """
    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)


    for m in range(radii.shape[0]):
        radius = radii[m]
        center = centers[m]

        solutions_x = find_val_in_interval(coord_min[0], coord_max[0], center[0])
        solutions_y = find_val_in_interval(coord_min[1], coord_max[1], center[1])

        #solutions_x = np.arange(coord_min[0], coord_max[0] + 0.001, radius / 2)
        #solutions_y = np.arange(coord_min[1], coord_max[1] + 0.001, radius / 2)

        for cx in solutions_x:
            for cy in solutions_y:
                dx = data_points[:, 0] - cx
                dy = data_points[:, 1] - cy
                dist_sq = dx * dx + dy * dy
                within = dist_sq <= radius * radius
                mask |= within

    return mask

class CircleShapeStrategy(ShapeStrategy):
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param

    def apply(self, data_points: np.ndarray, central_points: np.ndarray) -> np.ndarray:
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
        
        plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c='blue', marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Filtered Points (mask)')
        plt.grid(True)
        plt.show()
        return mask
    
class IntervalShapeStrategy(ShapeStrategy):
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param

    def apply(self, data_points: np.ndarray, central_points: np.ndarray) -> np.ndarray:
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
            centers[i] = central_points[i]

        mask = compute_1d_mask(data_points, radii, centers, coord_min, coord_max)
        filtered_data = data_points[mask]
        
        # Plotting the filtered 1D points
        plt.figure(figsize=(10, 2))
        plt.scatter(filtered_data, np.zeros_like(filtered_data), c='green', marker='o', label='Filtered Points')
        plt.xlabel('X-axis')
        plt.title('Filtered Points Within Intervals')
        plt.yticks([])  # Hide y-axis ticks for 1D visualization
        plt.legend()
        plt.grid(True)
        plt.show()
        return mask