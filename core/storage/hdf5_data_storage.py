# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:38:53 2024

@author: Maksim Eremenko
"""

# data_storage/hdf5_data_storage.py

from core.config.contracts.base_interfaces import (
    IConfigDataLoader,
    IConfigDataSaver,
)
import os
import h5py
import pandas as pd
import numpy as np
import logging

from core.runtime.log_utils import short_path


logger = logging.getLogger(__name__)

class HDF5ConfigDataSaver(IConfigDataSaver):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def save_data(self, data):
        try:
            logger.debug("Saving configuration data to HDF5: %s", self.hdf5_file_path)
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    logger.debug("%s: DataFrame with shape %s", key, value.shape)
                elif isinstance(value, pd.Series):
                    logger.debug("%s: Series with length %d", key, len(value))
                elif isinstance(value, dict):
                    logger.debug("%s: Dictionary with keys %s", key, list(value.keys()))
                elif isinstance(value, np.ndarray):
                    logger.debug("%s: ndarray with shape %s", key, value.shape)
                else:
                    logger.debug("%s: %s", key, type(value))
            
            with h5py.File(self.hdf5_file_path, 'w') as hdf5_file:
                # Save original_coords and average_coords
                for key in ['original_coords', 'average_coords', 'cells_origin']:
                    coords = data[key]
                    logger.debug("Creating '%s' dataset with shape %s", key, coords.shape)
                    hdf5_file.create_dataset(key, data=coords.to_numpy())

                # Save elements as fixed-length ASCII strings
                elements_encoded = np.array(data['elements'].to_list(), dtype='S')
                logger.debug("Creating 'elements' dataset")
                hdf5_file.create_dataset('elements', data=elements_encoded)

                # Save refnumbers
                logger.debug("Creating 'refnumbers' dataset")
                hdf5_file.create_dataset('refnumbers', data=data['refnumbers'].to_numpy())

                # Save vectors
                vectors = data['vectors']
                logger.debug("Creating 'vectors' dataset")
                hdf5_file.create_dataset('vectors', data=vectors)

                # Save metric dynamically based on the key (length/area/volume)
                metric_group = hdf5_file.create_group('metric')
                logger.debug("Saving 'metric' group")
                metric = data['metric']
                metric_key = next(key for key in metric if key in ['length', 'area', 'volume'])
                logger.debug("Creating 'metric/%s' dataset", metric_key)
                metric_group.create_dataset(metric_key, data=metric[metric_key])
                logger.debug("Creating 'metric/reciprocal_vectors' dataset")
                metric_group.create_dataset('reciprocal_vectors', data=metric['reciprocal_vectors'])

                # Save supercell
                logger.debug("Creating 'supercell' dataset")
                hdf5_file.create_dataset('supercell', data=data['supercell'])
                
                # Save coeff
                logger.debug("Creating 'coeff' dataset")
                hdf5_file.create_dataset('coeff', data=data['coeff'])
                
            logger.info("Configuration data saved to %s", short_path(self.hdf5_file_path))
        except Exception:
            logger.exception("Failed to save data to HDF5 file: %s", short_path(self.hdf5_file_path))
            raise


class HDF5ConfigDataLoader(IConfigDataLoader):
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path

    def can_load_data(self) -> bool:
        return os.path.exists(self.hdf5_file_path)

    def load_data(self):
        try:
            logger.debug("Loading configuration data from HDF5: %s", self.hdf5_file_path)
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
                logger.debug("Loaded 'original_coords'")

                average_coords_data = hdf5_file['average_coords'][:]
                average_coords = pd.DataFrame(average_coords_data, columns=get_column_names(average_coords_data.shape))
                logger.debug("Loaded 'average_coords'")
                
                cells_origin_data = hdf5_file['cells_origin'][:]
                cells_origin = pd.DataFrame(cells_origin_data, columns=get_column_names(cells_origin_data.shape))
                logger.debug("Loaded 'cells_origin'")
                
                # Load elements
                elements = pd.Series(
                    [elem.decode('utf-8') for elem in hdf5_file['elements'][:]],
                    name='element'
                )
                logger.debug("Loaded 'elements'")

                # Load refnumbers
                refnumbers = pd.Series(hdf5_file['refnumbers'][:], name='refnumbers')
                logger.debug("Loaded 'refnumbers'")

                # Load vectors
                vectors = hdf5_file['vectors'][:]
                logger.debug("Loaded 'vectors'")

                # Load metric group dynamically
                metric_group = hdf5_file['metric']
                metric_key = next(key for key in metric_group if key in ['length', 'area', 'volume'])
                metric = {
                    metric_key: metric_group[metric_key][()],
                    'reciprocal_vectors': metric_group['reciprocal_vectors'][:]
                }
                logger.debug("Loaded 'metric' with key '%s'", metric_key)

                # Load supercell
                supercell = hdf5_file['supercell'][:]
                logger.debug("Loaded 'supercell'")
                
                coeff = hdf5_file['coeff'][:]
                logger.debug("Loaded 'coeff'")
            logger.info("Configuration data loaded from %s", short_path(self.hdf5_file_path))
            return {
                'original_coords': original_coords,
                'average_coords': average_coords,
                'cells_origin' : cells_origin,
                'elements': elements,
                'refnumbers': refnumbers,
                'vectors': vectors,
                'metric': metric,
                'supercell': supercell,
                'coeff': coeff
            }
        except Exception:
            logger.exception("Failed to load data from HDF5 file: %s", short_path(self.hdf5_file_path))
            raise

