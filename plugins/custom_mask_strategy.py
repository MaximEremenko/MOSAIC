# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:15:43 2024

@author: Maksim Eremenko
"""

#plugins/custom_mask_strategy.py

from strategies.base_mask_strategy import BaseMaskStrategy
import numpy as np

class CustomMaskStrategy(BaseMaskStrategy):
    def generate_mask(self, data_points: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Custom mask generation logic.

        Args:
            data_points (np.ndarray): Data points array.
            parameters (dict): Includes 'threshold'.

        Returns:
            np.ndarray: Boolean mask array.
        """
        threshold = parameters.get('threshold', 5.0)
        # Example logic: Mask points where the sum of coordinates exceeds the threshold
        mask = np.sum(data_points, axis=1) > threshold
        return mask
