# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:45:12 2024

@author: Maksim Eremenko
"""

# calculators/cell_calculator.py

import numpy as np
from typing import List, Dict

import numpy as np
from typing import List, Dict

class CellCalculator:
    def calculate_vectors(self, cell_params: List[float]) -> np.ndarray:
        """
        Calculate lattice vectors for 1D, 2D, or 3D cases.

        Args:
            cell_params (List[float]): Cell parameters:
                - For 1D: [a]
                - For 2D: [a, b, gamma_deg]
                - For 3D: [a, b, c, alpha_deg, beta_deg, gamma_deg]

        Returns:
            np.ndarray: Lattice vectors as a numpy array.
                - 1D: shape (1, 1)
                - 2D: shape (2, 3)
                - 3D: shape (3, 3)
        """
        num_params = len(cell_params)

        if num_params == 1:
            # 1D case
            a = cell_params[0]
            vectors = np.array([[a]])  # 1D vector as shape (1, 1)

        elif num_params == 3:
            # 2D case
            a, b, gamma_deg = cell_params
            gamma = np.radians(gamma_deg)

            vectors = np.zeros((2, 2))  # 2D lattice in 2D space
            vectors[0, 0] = a
            vectors[1, 0] = b * np.cos(gamma)
            vectors[1, 1] = b * np.sin(gamma)

        elif num_params == 6:
            # 3D case
            a, b, c, alpha_deg, beta_deg, gamma_deg = cell_params
            alpha = np.radians(alpha_deg)
            beta = np.radians(beta_deg)
            gamma = np.radians(gamma_deg)

            vectors = np.zeros((3, 3))
            vectors[0, 0] = a
            vectors[1, 0] = b * np.cos(gamma)
            vectors[1, 1] = b * np.sin(gamma)
            vectors[2, 0] = c * np.cos(beta)
            vectors[2, 1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
            vectors[2, 2] = np.sqrt(c**2 - vectors[2, 0]**2 - vectors[2, 1]**2)

        else:
            raise ValueError("Unsupported number of cell parameters. Provide 1D, 2D, or 3D parameters.")

        return vectors

    def calculate_metric(self, vectors: np.ndarray) -> Dict:
        """
        Calculate the reciprocal lattice vectors and size metric 
        (length for 1D, area for 2D, volume for 3D).

        Args:
            vectors (np.ndarray): Lattice vectors as a numpy array.

        Returns:
            Dict: Contains 'reciprocal_vectors' and 'length'/'area'/'volume'.
        """
        dim = vectors.shape[0]

        if dim == 1:
            # 1D case
            a = vectors[0, 0]
            size_metric = a
            reciprocal_vectors = np.array([[2 * np.pi / a]])
            metric_key = 'length'  # Use 'length' for 1D

        elif dim == 2:
            # 2D case
            a_vector = vectors[0, :2]
            b_vector = vectors[1, :2]
            size_metric = np.linalg.det(np.vstack([a_vector, b_vector]))  # Area

            a_star = 2 * np.pi * np.array([b_vector[1], -b_vector[0]]) / size_metric
            b_star = 2 * np.pi * np.array([-a_vector[1], a_vector[0]]) / size_metric
            reciprocal_vectors = np.vstack([a_star, b_star])
            metric_key = 'area'  # Use 'area' for 2D

        elif dim == 3:
            # 3D case
            av = vectors[:, 0]
            bv = vectors[:, 1]
            cv = vectors[:, 2]

            size_metric = np.dot(av, np.cross(bv, cv))
            if size_metric == 0:
                raise ValueError("Vectors are degenerate; volume is zero.")

            a_star = 2 * np.pi * np.cross(bv, cv) / size_metric
            b_star = 2 * np.pi * np.cross(cv, av) / size_metric
            c_star = 2 * np.pi * np.cross(av, bv) / size_metric
            reciprocal_vectors = np.array([a_star, b_star, c_star])
            metric_key = 'volume'  # Use 'volume' for 3D

        else:
            raise ValueError("Unsupported vector dimension. Only 1D, 2D, or 3D is supported.")

        return {
            'reciprocal_vectors': reciprocal_vectors,
            metric_key: size_metric
        }

