# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:51:00 2024

@author: Maksim Eremenko
"""

# functions/split_point_data.py

import numpy as np
from data_structures.point_data import PointData

def split_point_data(point_data: PointData, chunk_size: int) -> [PointData]:
    """
    Splits the PointData into smaller chunks.

    Args:
        point_data (PointData): The complete point data.
        chunk_size (int): The number of points per chunk.

    Returns:
        [PointData]: A list of PointData instances, each containing a subset of the data.
    """
    num_points = point_data.coordinates.shape[0]
    chunks = []
    for start in range(0, num_points, chunk_size):
        end = start + chunk_size
        chunk_coordinates = point_data.coordinates[start:end]
        chunk_dist = point_data.dist_from_atom_center[start:end]
        chunk_step = point_data.step_in_frac[start:end]
        chunk_ids = point_data.central_point_ids[start:end]

        chunk = PointData(
            coordinates=chunk_coordinates,
            dist_from_atom_center=chunk_dist,
            step_in_frac=chunk_step,
            central_point_ids=chunk_ids
        )
        chunks.append(chunk)
    return chunks
