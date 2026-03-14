# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:01 2024
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.qspace.masking.shape_cpu import (
    compute_1d_mask,
    compute_2d_mask,
    compute_mask,
)


class SphereShapeStrategy:
    def __init__(self, spetial_points_param: Dict[str, Any]):
        self.spetial_points_param = spetial_points_param

    def generate_mask(self, data: np.ndarray) -> np.ndarray:
        coord_min = np.min(data, axis=0)
        coord_max = np.max(data, axis=0)

        special_points = self.spetial_points_param["specialPoints"]
        M = len(special_points)
        special_radii = np.empty(M, dtype=np.float64)
        special_coords = np.empty((M, 3), dtype=np.float64)
        for i, point in enumerate(special_points):
            special_radii[i] = point["radius"]
            special_coords[i, :] = point["coordinate"]

        return compute_mask(data, special_radii, special_coords, coord_min, coord_max)


class EllipsoidShapeStrategy:
    def __init__(self, spetial_points: np.ndarray, axes: np.ndarray, theta: float, phi: float):
        self.axes = axes
        self.rotation_matrix = self._create_rotation_matrix(theta, phi)
        self.spetial_points = spetial_points

    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(data_points), dtype=bool)
        for spetial_point in self.spetial_points:
            shifted_points = data_points - spetial_point
            rotated_points = shifted_points @ self.rotation_matrix.T
            scaled_points = rotated_points / self.axes
            distances = np.sum(scaled_points**2, axis=1)
            mask |= distances <= 1.0
        return mask

    def _create_rotation_matrix(self, theta: float, phi: float) -> np.ndarray:
        R_phi = np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )
        R_theta = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        return R_theta @ R_phi


class CircleShapeStrategy:
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param

    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        coord_min = np.min(data_points, axis=0) - 1
        coord_max = np.max(data_points, axis=0) + 1

        special_points = self.special_points_param["specialPoints"]
        num_circles = len(special_points)

        radii = np.empty(num_circles, dtype=np.float64)
        centers = np.empty((num_circles, 2), dtype=np.float64)
        for i, point in enumerate(special_points):
            radii[i] = point["radius"]
            centers[i, :] = point["coordinate"]

        return compute_2d_mask(data_points, radii, centers, coord_min, coord_max)


class IntervalShapeStrategy:
    def __init__(self, special_points_param: Dict[str, Any]):
        self.special_points_param = special_points_param

    def generate_mask(self, data_points: np.ndarray) -> np.ndarray:
        coord_min = np.min(data_points) - 1
        coord_max = np.max(data_points) + 1

        special_points = self.special_points_param["specialPoints"]
        num_intervals = len(special_points)

        radii = np.empty(num_intervals, dtype=np.float64)
        centers = np.empty(num_intervals, dtype=np.float64)
        for i, point in enumerate(special_points):
            radii[i] = point["radius"]
            centers[i] = np.array(point["coordinate"])

        return compute_1d_mask(data_points, radii, centers, coord_min, coord_max)
