# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:50:36 2024

@author: Maksim Eremenko
"""

# utilities/utils.py

import os

def determine_configuration_file_type(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.rmc6f':
        return 'rmc6f'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
