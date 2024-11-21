# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:34:50 2024

@author: Maksim Eremenko
"""

# utilities/file_utils.py

import os
import re

def get_next_chunk_id(output_dir: str, suffix: str = '') -> int:
    """
    Determines the next available chunk ID based on existing files in the output directory.

    Args:
        output_dir (str): Directory where chunk files are stored.
        suffix (str): Optional suffix to filter specific files (e.g., '_amplitudes').

    Returns:
        int: The next chunk ID.
    """
    pattern = re.compile(rf'point_data_chunk_(\d+){re.escape(suffix)}\.hdf5$')
    existing_ids = []
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            existing_ids.append(int(match.group(1)))
    if existing_ids:
        return max(existing_ids) + 1
    else:
        return 0
