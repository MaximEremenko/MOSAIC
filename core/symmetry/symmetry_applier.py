# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:57:06 2024

@author: Maksim Eremenko
"""

# symmetry/symmetry_applier.py

import numpy as np
from mantid.geometry import PointGroupFactory

class SymmetryApplier:
    def __init__(self, symmetry_parameters: dict):
        self.point_group_symmetry = symmetry_parameters.get('point_group_symmetry')
        self.tolerance = symmetry_parameters.get('tolerance', 0.05)
        self.pg = PointGroupFactory.createPointGroup(self.point_group_symmetry)

    def apply(self, data_points: np.ndarray) -> np.ndarray:
        """
        Applies symmetry operations to generate a mask.

        Args:
            data_points (np.ndarray): Data points array.

        Returns:
            np.ndarray: Boolean mask array.
        """
        equivalents = []
        for point in data_points:
            equivalents.extend(self.pg.getEquivalents(point))
        equivalents = np.array(equivalents)
        mask = self._mask_for_points(data_points, equivalents)
        return mask

    def _mask_for_points(self, data_points: np.ndarray, equivalents: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(data_points[:, np.newaxis] - equivalents, axis=2)
        mask = np.any(distances <= self.tolerance, axis=1)
        return mask

