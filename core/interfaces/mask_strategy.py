# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:11:58 2024

@author: Maksim Eremenko
"""

# interfaces/mask_strategy.py

from abc import ABC, abstractmethod
import numpy as np

class MaskStrategy(ABC):
    @abstractmethod
    def generate_mask(self, data_points: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Generates a boolean mask for the given data_points using the provided parameters.

        Args:
            data_points (np.ndarray): An array of data points (1D, 2D, or 3D).
            parameters (dict): Parameters for mask generation.

        Returns:
            np.ndarray: A boolean array representing the mask.
        """
        pass

    @abstractmethod
    def apply_mask(self, mask: np.ndarray, data_array: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Applies the mask to the data array.

        Args:
            mask (np.ndarray): A boolean array.
            data_array (np.ndarray): The data array to mask.
            parameters (dict): Parameters that may affect mask application.

        Returns:
            np.ndarray: The masked data array.
        """
        pass
