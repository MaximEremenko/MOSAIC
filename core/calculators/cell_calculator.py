# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:45:12 2024

@author: Maksim Eremenko
"""

# calculators/cell_calculator.py

import numpy as np
from typing import List, Dict

class CellCalculator:
    def calculate_vectors(self, cell_params: List[float]) -> np.ndarray:
        a, b, c, alpha_deg, beta_deg, gamma_deg = cell_params
        alpha = np.radians(alpha_deg)
        beta = np.radians(beta_deg)
        gamma = np.radians(gamma_deg)

        vectors = np.zeros((3, 3))

        # Calculate the vectors based on cell parameters
        vectors[0, 0] = a
        vectors[1, 0] = b * np.cos(gamma)
        vectors[1, 1] = b * np.sin(gamma)
        vectors[2, 0] = c * np.cos(beta)
        vectors[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        vectors[2, 2] = np.sqrt(
            c**2 - vectors[2, 0]**2 - vectors[2, 1]**2
        )

        return vectors

    def calculate_metric(self, vectors: np.ndarray) -> Dict:
        av = vectors[:, 0]
        bv = vectors[:, 1]
        cv = vectors[:, 2]

        volume = np.dot(av, np.cross(bv, cv))
        a_star = np.cross(bv, cv) / volume
        b_star = np.cross(cv, av) / volume
        c_star = np.cross(av, bv) / volume

        metric = {
            'reciprocal_vectors': np.array([a_star, b_star, c_star]),
            'volume': volume
        }
        return metric
