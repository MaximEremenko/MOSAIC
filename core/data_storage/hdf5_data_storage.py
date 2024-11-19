# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:38:53 2024

@author: Maksim Eremenko
"""

# data_storage/hdf5_data_storage.py

from interfaces.base_interfaces import IConfigDataSaver, IConfigDataLoader
import os
import h5py
import pandas as pd
import numpy as np

class HDF5ConfigDataSaver(IConfigDataSaver):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def save_data(self, data):
        try:
            # Debugging: Print keys and data types
            print("Saving the following data to HDF5:")
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    print(f" - {key}: DataFrame with shape {value.shape}")
                elif isinstance(value, pd.Series):
                    print(f" - {key}: Series with length {len(value)}")
                elif isinstance(value, dict):
                    print(f" - {key}: Dictionary with keys {list(value.keys())}")
                elif isinstance(value, np.ndarray):
                    print(f" - {key}: ndarray with shape {value.shape}")
                else:
                    print(f" - {key}: {type(value)}")
            
            with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
                # Save datasets
                print("Creating 'original_coords' dataset.")
                hdf5_file.create_dataset('original_coords', data=data['original_coords'].to_numpy())
                
                print("Creating 'average_coords' dataset.")
                hdf5_file.create_dataset('average_coords', data=data['average_coords'].to_numpy())
                
                # Save elements as fixed-length ASCII strings
                print("Encoding 'elements' as fixed-length ASCII strings.")
                elements_encoded = np.array(data['elements'].to_list(), dtype='S')
                print("Creating 'elements' dataset.")
                hdf5_file.create_dataset('elements', data=elements_encoded)
                
                print("Creating 'refnumbers' dataset.")
                hdf5_file.create_dataset('refnumbers', data=data['refnumbers'].to_numpy())
                
                
                print("Creating 'vectors' dataset.")
                hdf5_file.create_dataset('vectors', data=data['vectors'])
                
                # Save metric as a group
                print("Creating 'metric' group.")
                metric_group = hdf5_file.create_group('metric')
                print("Creating 'reciprocal_vectors' dataset within 'metric' group.")
                metric_group.create_dataset('reciprocal_vectors', data=data['metric']['reciprocal_vectors'])
                print("Creating 'volume' dataset within 'metric' group.")
                metric_group.create_dataset('volume', data=data['metric']['volume'])
                
                print("Creating 'supercell' dataset.")
                hdf5_file.create_dataset('supercell', data=data['supercell'])
            
            print(f"Data successfully saved to {self.hdf5_file_path}")
        except Exception as e:
            print(f"Failed to save data to HDF5 file: {e}")
            raise

class HDF5ConfigDataLoader(IConfigDataLoader):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def can_load_data(self) -> bool:
        return os.path.exists(self.hdf5_file_path)

    def load_data(self):
        try:
            with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
                print(f"Reading 'original_coords' dataset from {self.hdf5_file_path}")
                original_coords = pd.DataFrame(
                    hdf5_file['original_coords'][:], columns=['x', 'y', 'z']
                )
                
                print(f"Reading 'average_coords' dataset from {self.hdf5_file_path}")
                average_coords = pd.DataFrame(
                    hdf5_file['average_coords'][:], columns=['x', 'y', 'z']
                )
                
                print(f"Reading 'elements' dataset from {self.hdf5_file_path}")
                elements = pd.Series(
                    [elem.decode('utf-8') for elem in hdf5_file['elements'][:]], name='element'
                )
                
                print(f"Reading 'refnumbers' dataset from {self.hdf5_file_path}")
 
                
                refnumbers = pd.DataFrame(
                    hdf5_file['refnumbers'][:], columns=['refnumbers']
                )
                
                
                print(f"Reading 'vectors' dataset from {self.hdf5_file_path}")
                vectors = hdf5_file['vectors'][:]
                
                print(f"Reading 'metric' group from {self.hdf5_file_path}")
                metric_group = hdf5_file['metric']
                metric = {
                    'reciprocal_vectors': metric_group['reciprocal_vectors'][:],
                    'volume': metric_group['volume'][()]
                }
                
                print(f"Reading 'supercell' dataset from {self.hdf5_file_path}")
                supercell = hdf5_file['supercell'][:]
            
            print(f"Data successfully loaded from {self.hdf5_file_path}")
            return {
                'original_coords': original_coords,
                'average_coords': average_coords,
                'elements': elements,
                'refnumbers': refnumbers,
                'vectors': vectors,
                'metric': metric,
                'supercell': supercell
            }
        except Exception as e:
            print(f"Failed to read from HDF5 file: {e}")
            raise
