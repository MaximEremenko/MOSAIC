# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:15:13 2024

@author: Maksim Eremenko
"""

# strategies/base_mask_strategy.py

from interfaces.mask_strategy import MaskStrategy
import numpy as np

class BaseMaskStrategy(MaskStrategy):
    def generate_mask(self, data_points: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Base method to generate a mask. Should be overridden by subclasses.

        Args:
            data_points (np.ndarray): Data points array.
            parameters (dict): Parameters for mask generation.

        Returns:
            np.ndarray: Boolean mask array.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def apply_mask(self, mask: np.ndarray, data_array: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Applies the mask to the data array.

        Args:
            mask (np.ndarray): Boolean mask array.
            data_array (np.ndarray): Data array to be masked.

        Returns:
            np.ndarray: Masked data array.
        """
        if len(mask) != len(data_array):
            raise ValueError("Mask and data array must be of the same length.")
        return data_array[mask]
