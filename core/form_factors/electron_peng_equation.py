# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:21:42 2024

@author: Maksim Eremenko
"""

# form_factors/electron_peng_equation.py
import numpy as np
from form_factors.equation_strategy import  EquationStrategy

class ElectronPengEquation(EquationStrategy):
    def compute(self, q_vectors: np.ndarray, params: dict, charge: int = 0) -> np.ndarray:
        gsq = (q_vectors / (4 * np.pi)) ** 2
        form_factors = np.zeros_like(gsq)
        for a_i, b_i in zip(params['a'], params['b']):
            form_factors += a_i * np.exp(-b_i * gsq)
        return form_factors
