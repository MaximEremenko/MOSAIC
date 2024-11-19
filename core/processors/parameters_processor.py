# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:02:14 2024

@author: Maksim Eremenko
"""

from interfaces.base_interfaces import IConfigurationFileProcessor
from interfaces.parameter_interfaces import IParameterReader, IParameterParser
from data_storage.hdf5_parameter_storage import HDF5ParameterSaver, HDF5ParameterLoader
from typing import Optional

class ParametersProcessor(IConfigurationFileProcessor):
    def __init__(self, reader: IParameterReader, parser: Optional[IParameterParser] = None, hdf5_file_path: Optional[str] = None):
        self.reader = reader
        self.parser = parser
        self.hdf5_file_path = hdf5_file_path or 'parameters.hdf5'
        self.data = None
        self.data_loader = HDF5ParameterLoader(self.hdf5_file_path)
        self.data_saver = HDF5ParameterSaver(self.hdf5_file_path)

    def process(self):
        if self.data_loader.can_load_data():
            print(f"Loading parameters from HDF5 file: {self.hdf5_file_path}")
            self.data = self.data_loader.load_data()
        elif self.reader:
            print("Reading parameters using the provided reader.")
            data = self.reader.read()
            if self.parser:
                self.data = self.parser.parse(data)
            else:
                self.data = data
            self.data_saver.save_data(self.data)
        else:
            raise ValueError("No reader available to read parameters.")
        

    def get_parameters(self) -> dict:
        return self.data