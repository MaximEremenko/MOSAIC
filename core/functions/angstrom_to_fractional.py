# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:22:12 2024

@author: Maksim Eremenko
"""

import numpy as np

def angstrom_to_fractional(coords: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from Angstroms to fractional coordinates.

    :param coords: Coordinates in Angstroms (array of shape (N, D))
    :param vectors: Lattice vectors (array of shape (D, D))
    :return: Fractional coordinates (array of shape (N, D))
    """
    # Invert the lattice vectors matrix
    inv_vectors = np.linalg.inv(vectors.T)
    fractional_coords = np.dot(coords, inv_vectors)
    return fractional_coords
