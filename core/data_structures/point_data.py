# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:20:13 2024

@author: Maksim Eremenko
"""
# data_structures/point_data.py
from dataclasses import dataclass
import numpy as np
@dataclass
class PointData:
    coordinates: np.ndarray               # Shape: (N, D)
    dist_from_atom_center: np.ndarray     # Shape: (N, D)
    step_in_frac: np.ndarray              # Shape: (N, D) or scalar
    central_point_ids: np.ndarray         # Shape: (N,)

