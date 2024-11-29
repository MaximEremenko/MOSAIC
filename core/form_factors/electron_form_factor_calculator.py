# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:19:27 2024

@author: Maksim Eremenko
"""

# form_factors/electron_form_factor_calculator.py
import numpy as np
from form_factors.form_factor_calculator import FormFactorCalculator
from form_factors.equation_strategy import EquationStrategy
from form_factors.parameter_provider import ParameterProvider

class ElectronFormFactorCalculator(FormFactorCalculator):
    def __init__(self, equation_strategy: EquationStrategy, parameter_provider: ParameterProvider):
        self.equation_strategy = equation_strategy
        self.parameter_provider = parameter_provider

    def calculate(self, reciprocal_space_coordinates: np.ndarray, element: str, charge: int = 0) -> np.ndarray:
        params = self.parameter_provider.get_parameters(element, charge=charge)
        q_vectors = np.linalg.norm(reciprocal_space_coordinates, axis=1)
        return self.equation_strategy.compute(q_vectors, params, charge=charge)
