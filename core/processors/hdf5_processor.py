# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:34:03 2024

@author: Maksim Eremenko
"""

from interfaces.base_interfaces import IConfigurationFileProcessor
from data_storage.hdf5_data_storage import HDF5ConfigDataLoader
import pandas as pd
from typing import Optional, Dict
import numpy as np

class HDF5Processor(IConfigurationFileProcessor):
    def __init__(self, hdf5_file_path: str):
        self.data_loader = HDF5ConfigDataLoader(hdf5_file_path)
        self.data = None

    def process(self):
        self.data = self.data_loader.load_data()

    def get_coordinates(self) -> pd.DataFrame:
        return self.data['original_coords']

    def get_average_coordinates(self) -> pd.DataFrame:
        return self.data['average_coords']

    def get_supercell(self) -> np.ndarray:
        return self.data['supercell']

    def get_elements(self) -> pd.Series:
        return self.data['elements']
    
    def get_refnumbers(self) -> pd.Series:
        return self.data['refnumbers']
    
    def get_vectors(self) -> Optional[np.ndarray]:
        return self.data['vectors']

    def get_metric(self) -> Dict:
        return self.data['metric']
    
    def get_coeff(self) -> np.ndarray:
        return self.data['coeff']
        
    def get_cells_origin(self) -> pd.DataFrame:
        return self.data['cells_origin']