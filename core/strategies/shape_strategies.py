# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:01 2024

@author: Maksim Eremenko
"""

# strategies/shape_strategies.py

import numpy as np
from interfaces.shape_strategy import ShapeStrategy

class SphereShapeStrategy(ShapeStrategy):
    def __init__(self, radius: float):
        self.radius = radius

    def apply(self, data_points: np.ndarray, central_points: np.ndarray) -> np.ndarray:
        """
        Generates a mask for points within multiple spheres.

        Args:
            data_points (np.ndarray): Data points array.
            central_points (np.ndarray): Array of centers of the spheres.

        Returns:
            np.ndarray: Boolean mask array.
        """
        # Initialize mask
        mask = np.zeros(len(data_points), dtype=bool)
        # Calculate distances from all data points to each central point
        for central_point in central_points:
            distances = np.linalg.norm(data_points - central_point, axis=1)
            mask |= distances <= self.radius
        return mask

class EllipsoidShapeStrategy(ShapeStrategy):
    def __init__(self, axes: np.ndarray, theta: float, phi: float):
        self.axes = axes
        self.rotation_matrix = self._create_rotation_matrix(theta, phi)

    def apply(self, data_points: np.ndarray, central_points: np.ndarray) -> np.ndarray:
        """
        Generates a mask for points within multiple ellipsoids.

        Args:
            data_points (np.ndarray): Data points array.
            central_points (np.ndarray): Array of centers of the ellipsoids.

        Returns:
            np.ndarray: Boolean mask array.
        """
        mask = np.zeros(len(data_points), dtype=bool)
        for central_point in central_points:
            shifted_points = data_points - central_point
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
