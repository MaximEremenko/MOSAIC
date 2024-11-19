# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:14:06 2024

@author: Maksim Eremenko
"""

# readers/configuration_data_accessor.py

import os
#import pandas as pd
#import numpy as np
#import h5py  # For HDF5 file handling
#from typing import Optional
#from managers.configuration_processor_manager import ConfigurationProcessorManager
#from processors.configuration_file_data_processor import ConfigurationFileDataProcessor
from interfaces.base_interfaces import IDataLoader, IDataProcessor, IDataSaver
from factories.hdf5_factory import HDF5DataLoader, HDF5DataSaver
#factories.hdf5_factory import  
from interfaces.base_interfaces import IDataLoaderFactory, IDataProcessorFactory, IDataSaverFactory

class ConfigurationDataAccessor:
    def __init__(self, 
                 data_loader: IDataLoader, 
                 data_processor: IDataProcessor, 
                 data_saver: IDataSaver):
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.data_saver = data_saver

        # Initialize data attributes
        self.original_coords = None
        self.average_coords = None
        self.elements = None
        self.vectors = None
        self.metric = None
        self.supercell = None

    def load_data(self):
        if self.data_loader.can_load_data():
            self._load_from_loader()
        else:
            self._process_data()
            self._save_data()

    def _load_from_loader(self):
        data = self.data_loader.load_data()
        self._set_data_attributes(data)

    def _process_data(self):
        data = self.data_processor.process_data()
        self._set_data_attributes(data)

    def _save_data(self):
        data = {
            'original_coords': self.original_coords,
            'average_coords': self.average_coords,
            'elements': self.elements,
            'vectors': self.vectors,
            'metric': self.metric,
            'supercell': self.supercell
        }
        self.data_saver.save_data(data)

    def _set_data_attributes(self, data):
        self.original_coords = data['original_coords']
        self.average_coords = data['average_coords']
        self.elements = data['elements']
        self.vectors = data['vectors']
        self.metric = data['metric']
        self.supercell = data['supercell']
