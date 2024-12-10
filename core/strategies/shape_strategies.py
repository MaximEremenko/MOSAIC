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

class SphereShapeStrategy(ShapeStrategy):
    def __init__(self, spetial_points_param: dict):
        self.spetial_points_param = spetial_points_param

    def apply(self, data_points: np.ndarray, spetial_points: np.ndarray) -> np.ndarray:
        """
        Generates a mask for points within multiple spheres, including all integer-shifted 
        equivalents of the special points within the data_points domain.
        """

        # Helper function from the provided code
        def corresponding_value(n_val, a):
            if n_val == 0:
                return abs(a)
            else:
                return np.sign(n_val) * (np.abs(n_val) + np.abs(a))

        def find_val_in_interval(coord_min, coord_max, a):
            # Adjust the range boundaries to the nearest integers
            coord_min_int = np.floor(coord_min).astype(np.int32)  # Largest integer <= coord_min
            coord_max_int = np.ceil(coord_max).astype(np.int32)   # Smallest integer >= coord_max

            coord_min = np.round(coord_min, 8)
            coord_max = np.round(coord_max, 8)
    
            solutions = np.empty(0)
            for n_val in range(coord_min_int-2, coord_max_int + 2):
                if n_val == 0:
                    # Check both +a and -a if they fall within [coord_min, coord_max]
                    if (coord_min-2 <= a <= coord_max+2) and (coord_min-2 <= -a <= coord_max+2):
                        # both +a and -a fit
                        solutions = np.append(solutions, -abs(a))
                        solutions = np.append(solutions, abs(a))
                    elif (coord_min <= a <= coord_max):
                        solutions = np.append(solutions, abs(a))
                    elif coord_min <= -a <= coord_max:
                        solutions = np.append(solutions, -abs(a))
                else:
                    expr_val = corresponding_value(n_val, a)
                    if coord_min <= expr_val <= coord_max:
                        solutions = np.append(solutions, expr_val)
    
            solutions = np.unique(solutions, axis=0)
            solutions = np.round(solutions, 8)
            return solutions

        mask = np.zeros(len(data_points), dtype=bool)

        # Determine data_points bounding range
        coord_min = np.min(data_points, axis=0)
        coord_max = np.max(data_points, axis=0)

        # Iterate over each special point defined in the parameters
        for sp_point in self.spetial_points_param["specialPoints"]:
            radius = sp_point["radius"]
            base_coord = sp_point["coordinate"]

            # Find all integer-based solutions within the given intervals for each dimension
            solutions_x = find_val_in_interval(coord_min[0], coord_max[0], base_coord[0])
            solutions_y = find_val_in_interval(coord_min[1], coord_max[1], base_coord[1])
            solutions_z = find_val_in_interval(coord_min[2], coord_max[2], base_coord[2])

            # Generate all possible combinations of (x, y, z) from these solutions
            for cx, cy, cz in product(solutions_x, solutions_y, solutions_z):
                spetial_point = np.array([cx, cy, cz])
                distances = np.linalg.norm(data_points - spetial_point, axis=1)
                mask |= distances <= radius

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
