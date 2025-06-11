# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:22:12 2024

@author: Maksim Eremenko
"""

import numpy as np

def angstrom_to_fractional(coords: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from Angstroms to fractional coordinates.

    Supports scalar, 1D, 2D, or 3D coordinates.

    :param coords: Coordinates in Angstroms (scalar, 1D, or array of shape (N, D))
    :param vectors: Lattice vectors (scalar for 1D, or array of shape (D, D))
    :return: Fractional coordinates (array of shape (N, D))
    """
    # Handle scalar 1D case
    if np.isscalar(coords) and np.isscalar(vectors):
        return np.array([[coords / vectors]])

    coords = np.atleast_1d(coords)
    vectors = np.atleast_2d(vectors)

    # If vectors is 1D array [a], promote to (1,1)
    if vectors.shape == (1,):
        vectors = vectors.reshape(1, 1)

    # Promote coords if it's 1D: treat as column vector
    if coords.ndim == 1:
        coords = coords[:, np.newaxis]

    if coords.ndim != 2 or vectors.ndim != 2:
        raise ValueError(f"Incompatible shapes: coords {coords.shape}, vectors {vectors.shape}")

    D = vectors.shape[0]
    if vectors.shape[1] != D:
        raise ValueError(f"Lattice vectors must be square (got shape {vectors.shape})")

    if coords.shape[1] != D:
        raise ValueError(f"Coordinate dimension mismatch: {coords.shape[1]} vs {D}")

    inv_vectors = np.linalg.inv(vectors.T)
    fractional_coords = coords @ inv_vectors
    return fractional_coords
