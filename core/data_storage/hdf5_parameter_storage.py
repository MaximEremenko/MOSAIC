# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:58:39 2024

@author: Maksim Eremenko
"""

import h5py
from interfaces.base_interfaces import IConfigDataSaver, IConfigDataLoader
import os
import json
# class HDF5ParameterSaver(IConfigDataSaver):
#     def __init__(self, hdf5_file_path: str):
#         self.hdf5_file_path = hdf5_file_path

#     def save_data(self, data: dict):
#         try:
#             with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
#                 self._recursively_save_dict_contents_to_group(hdf5_file, '/', data)
#             print(f"Parameters successfully saved to {self.hdf5_file_path}")
#         except Exception as e:
#             print(f"Failed to save parameters to HDF5 file: {e}")
#             raise

#     def _recursively_save_dict_contents_to_group(self, h5file, path, dic):
#         for key, item in dic.items():
#             if isinstance(item, dict):
#                 #group = h5file.create_group(path + key)
#                 self._recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
#             else:
#                 h5file.create_dataset(path + key, data=item)
                
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
            print(f"Parameters successfully saved to {self.hdf5_file_path}")
        except Exception as e:
            print(f"Failed to save parameters to HDF5 file: {e}")
            raise         

# class HDF5ParameterLoader(IConfigDataLoader):
#     def __init__(self, hdf5_file_path: str):
#         self.hdf5_file_path = hdf5_file_path

#     def can_load_data(self) -> bool:
#         return os.path.exists(self.hdf5_file_path)

#     def load_data(self) -> dict:
#         try:
#             with h5py.File(self.hdf5_file_path, 'r') as hdf5_file:
#                 data = self._recursively_load_dict_contents_from_group(hdf5_file, '/')
#             print(f"Parameters successfully loaded from {self.hdf5_file_path}")
#             return data
#         except Exception as e:
#             print(f"Failed to read parameters from HDF5 file: {e}")
#             raise

#     def _recursively_load_dict_contents_from_group(self, h5file, path):
#         ans = {}
#         for key, item in h5file[path].items():
#             if isinstance(item, h5py.Dataset):
#                 ans[key] = item[()]
#             elif isinstance(item, h5py.Group):
#                 ans[key] = self._recursively_load_dict_contents_from_group(h5file, path + key + '/')
#         return ans

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
            print(f"Parameters successfully loaded from {self.hdf5_file_path}")
            return data
        except Exception as e:
            print(f"Failed to read parameters from HDF5 file: {e}")
            raise