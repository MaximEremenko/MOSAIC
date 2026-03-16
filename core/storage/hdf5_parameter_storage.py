# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:58:39 2024

@author: Maksim Eremenko
"""

import os
import json
import h5py
import logging

from core.config.contracts.base_interfaces import (
    IConfigDataLoader,
    IConfigDataSaver,
)


logger = logging.getLogger(__name__)

class HDF5ParameterSaver(IConfigDataSaver):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def save_data(self, data: dict):
        try:
            with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
                # Serialize the parameters dictionary to a JSON string
                json_str = json.dumps(data)
                # Use h5py's string dtype for UTF-8 encoding
                dt = h5py.string_dtype(encoding='utf-8')
                # Create a dataset named 'parameters' to store the JSON string
                hdf5_file.create_dataset('parameters', data=json_str, dtype=dt)
            logger.info("Parameters saved to %s", self.hdf5_file_path)
        except Exception:
            logger.exception("Failed to save parameters to HDF5 file: %s", self.hdf5_file_path)
            raise

class HDF5ParameterLoader(IConfigDataLoader):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def can_load_data(self) -> bool:
        return os.path.exists(self.hdf5_file_path)

    def load_data(self) -> dict:
        try:
            with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
                # Read the JSON string from the 'parameters' dataset
                json_str = hdf5_file['parameters'][()]
                # If the data is bytes, decode it to a string
                if isinstance(json_str, bytes):
                    json_str = json_str.decode('utf-8')
                # Deserialize the JSON string back to a dictionary
                data = json.loads(json_str)
            logger.info("Parameters loaded from %s", self.hdf5_file_path)
            return data
        except Exception:
            logger.exception("Failed to read parameters from HDF5 file: %s", self.hdf5_file_path)
            raise
