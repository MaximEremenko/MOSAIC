# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:35:37 2024

@author: Maksim Eremenko
"""
from typing import Optional
from interfaces.base_interfaces import IConfigDataSaver, IConfigDataLoader
from processors.configuration_file_data_processor import ConfigurationFileDataProcessor
import os
import h5py
import pandas as pd
import numpy as np

# class HDF5ConfigDataSaver(IConfigDataSaver):
#     def __init__(self, hdf5_file_path: str):
#         self.hdf5_file_path = hdf5_file_path

#     def save_data(self, data):
#         try:
#             with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
#                 # Save datasets
#                 hdf5_file.create_dataset('original_coords', data=data['original_coords'].to_numpy())
#                 hdf5_file.create_dataset('average_coords', data=data['average_coords'].to_numpy())
#                 # Save elements as fixed-length ASCII strings
#                 elements_encoded = np.array(data['elements'].to_list(), dtype='S')
#                 hdf5_file.create_dataset('elements', data=elements_encoded)
#                 hdf5_file.create_dataset('vectors', data=data['vectors'])
#                 # Save metric as a group
#                 metric_group = hdf5_file.create_group('metric')
#                 metric_group.create_dataset('reciprocal_vectors', data=data['metric']['reciprocal_vectors'])
#                 metric_group.create_dataset('volume', data=data['metric']['volume'])
#                 hdf5_file.create_dataset('supercell', data=data['supercell'])
#             print(f"Data successfully saved to {self.hdf5_file_path}")
#         except Exception as e:
#             print(f"Failed to save data to HDF5 file: {e}")
#             raise

# class HDF5ConfigDataLoader(IConfigDataLoader):
#     def __init__(self, hdf5_file_path: str):
#         self.hdf5_file_path = hdf5_file_path

#     def can_load_data(self) -> bool:
#         return os.path.exists(self.hdf5_file_path)

#     def load_data(self):
#         try:
#             with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
#                 # Read datasets
#                 original_coords = pd.DataFrame(
#                     hdf5_file['original_coords'][:], columns=['x', 'y', 'z']
#                 )
#                 average_coords = pd.DataFrame(
#                     hdf5_file['average_coords'][:], columns=['x', 'y', 'z']
#                 )
#                 elements = pd.Series(
#                     [elem.decode('utf-8') for elem in hdf5_file['elements'][:]], name='element'
#                 )
#                 vectors = hdf5_file['vectors'][:]
#                 # Read metric group
#                 metric_group = hdf5_file['metric']
#                 metric = {
#                     'reciprocal_vectors': metric_group['reciprocal_vectors'][:],
#                     'volume': metric_group['volume'][()]
#                 }
#                 supercell = hdf5_file['supercell'][:]
#             print(f"Data successfully loaded from {self.hdf5_file_path}")
#             return {
#                 'original_coords': original_coords,
#                 'average_coords': average_coords,
#                 'elements': elements,
#                 'vectors': vectors,
#                 'metric': metric,
#                 'supercell': supercell
#             }
#         except Exception as e:
#             print(f"Failed to read from HDF5 file: {e}")
#             raise
    
