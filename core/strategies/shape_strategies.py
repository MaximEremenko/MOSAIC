# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:01 2024

@author: Maksim Eremenko
"""

# strategies/shape_strategies.py

import numpy as np
from interfaces.shape_strategy import ShapeStrategy

from itertools import product
import numpy as np

from numba import njit
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
def compute_mask(data_points, special_radii, special_coords, coord_min, coord_max):
    # data_points: (N,3)
    # special_radii: (M,)
    # special_coords: (M,3)
    # Returns a boolean mask of length N.

    N = data_points.shape[0]
    mask = np.zeros(N, dtype=np.bool_)

    for m in range(special_radii.shape[0]):
        radius = special_radii[m]
        base_coord = special_coords[m]

        solutions_x = find_val_in_interval(coord_min[0], coord_max[0], base_coord[0])
        solutions_y = find_val_in_interval(coord_min[1], coord_max[1], base_coord[1])
        solutions_z = find_val_in_interval(coord_min[2], coord_max[2], base_coord[2])

        # Nested loops instead of product
        for i in range(len(solutions_x)):
            cx = solutions_x[i]
            for j in range(len(solutions_y)):
                cy = solutions_y[j]
                for k in range(len(solutions_z)):
                    cz = solutions_z[k]

                    # Compute distances: (x - cx)^2 + (y - cy)^2 + (z - cz)^2
                    dx = data_points[:, 0] - cx
                    dy = data_points[:, 1] - cy
                    dz = data_points[:, 2] - cz
                    dist_sq = dx*dx + dy*dy + dz*dz
                    # Compare with radius^2 to avoid sqrt
                    within = dist_sq <= radius*radius

                    # Update mask
                    for idx in range(N):
                        if within[idx]:
                            mask[idx] = True

    return mask

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
