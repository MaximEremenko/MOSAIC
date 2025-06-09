# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:11:58 2024

@author: Maksim Eremenko
"""

# interfaces/mask_strategy.py

from abc import ABC, abstractmethod
import numpy as np

class IMaskStrategy(ABC):
    """
    Strategy interface for generating a boolean mask over a collection of points.
    """

    @abstractmethod
    def generate_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Compute a boolean mask for the given points.

        Args:
            data (np.ndarray):
                - If you have an hkl grid, shape should be (N, 3).
                - If you have 1D data, shape should be (N,).
                - Any other shape you choose, as long as your implementation
                  knows how to interpret it.

        Returns:
            mask (np.ndarray): A 1D boolean array of length N.
        """
        pass
