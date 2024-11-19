# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:21:35 2024

@author: Maksim Eremenko
"""

# interfaces/point_parameters_processor_interface.py

from abc import ABC, abstractmethod
from data_structures.point_data import PointData

class IPointParametersProcessor(ABC):
    @abstractmethod
    def process_parameters(self):
        """Process the input parameters and prepare point data."""
        pass

    @abstractmethod
    def get_point_data(self) -> PointData:
        """Return a list of PointData instances."""
        pass

    @abstractmethod
    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        """Save point data to an HDF5 file."""
        pass

    @abstractmethod
    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        """
        Load point data from an HDF5 file.
        Returns True if successful, False otherwise.
        """
        pass
