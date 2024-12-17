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
                # Save original_coords and average_coords
                for key in ['original_coords', 'average_coords']:
                    coords = data[key]
                    print(f"Creating '{key}' dataset with shape {coords.shape}")
                    hdf5_file.create_dataset(key, data=coords.to_numpy())

                # Save elements as fixed-length ASCII strings
                elements_encoded = np.array(data['elements'].to_list(), dtype='S')
                print("Creating 'elements' dataset")
                hdf5_file.create_dataset('elements', data=elements_encoded)

                # Save refnumbers
                print("Creating 'refnumbers' dataset")
                hdf5_file.create_dataset('refnumbers', data=data['refnumbers'].to_numpy())

                # Save vectors
                vectors = data['vectors']
                print("Creating 'vectors' dataset")
                hdf5_file.create_dataset('vectors', data=vectors)

                # Save metric dynamically based on the key (length/area/volume)
                metric_group = hdf5_file.create_group('metric')
                print("Saving 'metric' group")
                metric = data['metric']
                metric_key = next(key for key in metric if key in ['length', 'area', 'volume'])
                print(f"Creating 'metric/{metric_key}' dataset")
                metric_group.create_dataset(metric_key, data=metric[metric_key])
                print("Creating 'metric/reciprocal_vectors' dataset")
                metric_group.create_dataset('reciprocal_vectors', data=metric['reciprocal_vectors'])

                # Save supercell
                print("Creating 'supercell' dataset")
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
            print(f"Loading data from HDF5: {self.hdf5_file_path}")
            with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
                # Load original_coords and average_coords
                def get_column_names(shape):
                    if shape[1] == 1:
                        return ['x']
                    elif shape[1] == 2:
                        return ['x', 'y']
                    else:
                        return ['x', 'y', 'z']

                original_coords_data = hdf5_file['original_coords'][:]
                original_coords = pd.DataFrame(original_coords_data, columns=get_column_names(original_coords_data.shape))
                print("Loaded 'original_coords'")

                average_coords_data = hdf5_file['average_coords'][:]
                average_coords = pd.DataFrame(average_coords_data, columns=get_column_names(average_coords_data.shape))
                print("Loaded 'average_coords'")

                # Load elements
                elements = pd.Series(
                    [elem.decode('utf-8') for elem in hdf5_file['elements'][:]],
                    name='element'
                )
                print("Loaded 'elements'")

                # Load refnumbers
                refnumbers = pd.Series(hdf5_file['refnumbers'][:], name='refnumbers')
                print("Loaded 'refnumbers'")

                # Load vectors
                vectors = hdf5_file['vectors'][:]
                print("Loaded 'vectors'")

                # Load metric group dynamically
                metric_group = hdf5_file['metric']
                metric_key = next(key for key in metric_group if key in ['length', 'area', 'volume'])
                metric = {
                    metric_key: metric_group[metric_key][()],
                    'reciprocal_vectors': metric_group['reciprocal_vectors'][:]
                }
                print(f"Loaded 'metric' with key '{metric_key}'")

                # Load supercell
                supercell = hdf5_file['supercell'][:]
                print("Loaded 'supercell'")

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
            print(f"Failed to load data from HDF5 file: {e}")
            raise

