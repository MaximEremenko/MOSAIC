# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:50:11 2024

@author: Maksim Eremenko
"""

# factories/configuration_processor_factory.py

from interfaces.base_interfaces import IConfigurationProcessorFactory
from processors.rmc6f_processor import RMC6fProcessor
from processors.configuration2d_file_processor import ConfigurationFileProcessor2D

from processors.rmc6f_average_structure_calculator import RMC6fAverageStructureCalculator
from processors.rmc6f_average_structure_reader import RMC6fAverageStructureReader
from typing import Optional

from factories.hdf5_processor_factory import HDF5ProcessorFactory
from factories.parameters_processor_factory import ParametersProcessorFactory

class RMC6fProcessorFactory(IConfigurationProcessorFactory):
    def create_processor(self,
                         file_path: str,
                         processor_type: str = 'calculate',
                         average_file_path: Optional[str] = None) -> RMC6fProcessor:
        if processor_type == 'calculate':
            data_processor = RMC6fAverageStructureCalculator()
        elif processor_type == 'read':
            if average_file_path is None:
                raise ValueError("An average file path must be provided when processor_type is 'read'.")
            data_processor = RMC6fAverageStructureReader(average_file_path)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")

        return RMC6fProcessor(file_path, data_processor)
    
class Processor2DFactory(IConfigurationProcessorFactory):
    def create_processor(self, file_path: str, processor_type: str = 'read', average_file_path: str = None):
        return ConfigurationFileProcessor2D(file_path)

class ConfigurationProcessorFactoryProvider:
    _factories = {
        'rmc6f': RMC6fProcessorFactory(),
        'hdf5': HDF5ProcessorFactory(),
        'f2d': Processor2DFactory(),
        # Add more factories for other configuration file types as needed
    }

    @staticmethod
    def get_factory(file_type: str) -> IConfigurationProcessorFactory:
        factory = ConfigurationProcessorFactoryProvider._factories.get(file_type)
        if factory:
            return factory
        else:
            raise ValueError(f"Unsupported file type: {file_type}")