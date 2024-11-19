# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:51:00 2024

@author: Maksim Eremenko
"""
# functions/split_point_data.py

import numpy as np
from data_structures.point_data import PointData
from typing import List
import math

def split_point_data(point_data: PointData, num_chunks: int) -> List[PointData]:
    """
    Splits the PointData into a specified number of chunks.

    Args:
        point_data (PointData): The complete point data.
        num_chunks (int): The desired number of chunks.

    Returns:
        List[PointData]: A list of PointData instances, each containing a subset of the data.
    """
    num_points = point_data.coordinates.shape[0]
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")
    if num_chunks > num_points:
        raise ValueError("Number of chunks cannot exceed the number of points.")

    # Calculate the number of points per chunk
    points_per_chunk = math.ceil(num_points / num_chunks)

    chunks = []
    for i in range(num_chunks):
        start = i * points_per_chunk
        end = min(start + points_per_chunk, num_points)
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
