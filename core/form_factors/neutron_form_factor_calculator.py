# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:16:57 2024

@author: Maksim Eremenko
"""

# form_factors/neutron_form_factor_calculator.py
import numpy as np
from form_factors.form_factor_calculator import  FormFactorCalculator
from form_factors.parameter_provider import ParameterProvider

class NeutronFormFactorCalculator(FormFactorCalculator):
    def __init__(self, parameter_provider: ParameterProvider):
        self.parameter_provider = parameter_provider

    def calculate(self, reciprocal_space_coordinates: np.ndarray, element: str, charge: int = 0) -> np.ndarray:
        scattering_length = self.parameter_provider.get_parameters(element, charge=charge)
        n_coordinates = reciprocal_space_coordinates.shape[0]
        return np.full(n_coordinates, scattering_length)
