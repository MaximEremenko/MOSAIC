# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:20:13 2024

@author: Maksim Eremenko
"""
# data_structures/point_data.py

import numpy as np
from dataclasses import dataclass

@dataclass
class PointData:
    coordinates: np.ndarray                # Shape: (N, D)
    dist_from_atom_center: np.ndarray      # Shape: (N, D)
    step_in_frac: np.ndarray               # Shape: (N, D)
    central_point_ids: np.ndarray          # Shape: (N,)
    chunk_ids: np.ndarray                  # Shape: (N,)
    grid_amplitude_initialized: np.ndarray # Shape: (N,), dtype: bool

    def __post_init__(self):
        if self.chunk_ids is None or len(self.chunk_ids) == 0:
            self.chunk_ids = np.zeros(self.coordinates.shape[0], dtype=int)
        if self.grid_amplitude_initialized is None or len(self.grid_amplitude_initialized) == 0:
            self.grid_amplitude_initialized = np.zeros(self.coordinates.shape[0], dtype=bool)

