# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:15:51 2024

@author: Maksim Eremenko
"""

# form_factors/default_form_factor_calculator.py
import numpy as np
from form_factors.form_factor_calculator import  FormFactorCalculator

class DefaultFormFactorCalculator(FormFactorCalculator):
    def calculate(self, reciprocal_space_coordinates: np.ndarray, element: str, charge: int = 0) -> np.ndarray:
        n_coordinates = reciprocal_space_coordinates.shape[0]
        return np.ones(n_coordinates)
