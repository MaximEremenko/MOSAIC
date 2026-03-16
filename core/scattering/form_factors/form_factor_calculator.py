# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:14:13 2024

@author: Maksim Eremenko
"""

# form_factors/form_factor_calculator.py
import numpy as np
from abc import ABC, abstractmethod

class FormFactorCalculator(ABC):
    @abstractmethod
    def calculate(self, reciprocal_space_coordinates: np.ndarray, element: str, charge: int = 0) -> np.ndarray:
        pass
