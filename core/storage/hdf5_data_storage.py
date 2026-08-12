# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:38:53 2024

@author: Maksim Eremenko
"""

# data_storage/hdf5_data_storage.py

from core.config.contracts.base_interfaces import (
    IConfigDataLoader,
)
import os
import h5py
import pandas as pd
import numpy as np
import logging

from core.runtime.log_utils import short_path


logger = logging.getLogger(__name__)

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

