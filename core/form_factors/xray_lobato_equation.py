# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:12:27 2024

@author: Maksim Eremenko
"""

# form_factors/xray_lobato_equation.py

import numpy as np
from form_factors.equation_strategy import EquationStrategy

class XRayLobatoEquation(EquationStrategy):
    def compute(self, q_vectors: np.ndarray, params: dict, charge: int = 0) -> np.ndarray:
        """
        Computes the form factor using the Lobato equation for X-ray scattering.

        Args:
            q_vectors (np.ndarray): Array of q-vector magnitudes.
            params (dict): Parameters specific to the element.
            charge (int, optional): Charge of the element. Default is 0.

        Returns:
            np.ndarray: Computed form factor values.
        """
        gsq = (q_vectors / (4 * np.pi)) ** 2
        form_factors = np.zeros_like(gsq)

        for a_i, b_i in zip(params['a'], params['b']):
            form_factors += a_i * (2 + b_i * gsq) / ((1 + b_i * gsq) ** 2)

        return form_factors
