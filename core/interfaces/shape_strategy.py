# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:21:42 2024

@author: Maksim Eremenko
"""

# interfaces/shape_strategy.py

from abc import ABC, abstractmethod
import numpy as np

class ShapeStrategy(ABC):
    @abstractmethod
    def apply(self, data_points: np.ndarray, central_points: np.ndarray) -> np.ndarray:
        """
        Applies the shape-based mask to the data_points based on multiple central points.

        Args:
            data_points (np.ndarray): An array of data points.
            central_points (np.ndarray): An array of central points for the shapes.

        Returns:
            np.ndarray: A boolean array representing the mask.
        """
        pass
