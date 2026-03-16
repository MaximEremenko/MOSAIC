# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:50:11 2024

@author: Maksim Eremenko
"""

# factories/configuration_processor_factory.py

from core.config.contracts.base_interfaces import (
    IConfigurationProcessorFactory,
)
from core.config.processors.file_processor_1d import (
    ConfigurationFileProcessor1D,
)
from core.config.processors.file_processor_2d import (
    ConfigurationFileProcessor2D,
)
from core.config.processors.rmc6f_average_structure_calculator import (
    RMC6fAverageStructureCalculator,
)
from core.config.processors.rmc6f_average_structure_reader import (
    RMC6fAverageStructureReader,
)
from core.config.processors.rmc6f_processor import RMC6fProcessor
from typing import Optional

from core.config.factories.hdf5_factory import HDF5ProcessorFactory

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

class Processor1DFactory(IConfigurationProcessorFactory):
    def create_processor(self, file_path: str, processor_type: str = 'read', average_file_path: str = None):
        return ConfigurationFileProcessor1D(file_path)

class ConfigurationProcessorFactoryProvider:
    _factories = {
        'rmc6f': RMC6fProcessorFactory(),
        'hdf5': HDF5ProcessorFactory(),
        'f2d': Processor2DFactory(),
        'f1d': Processor1DFactory(),
    }

    @staticmethod
    def register_factory(
        file_type: str, factory: IConfigurationProcessorFactory
    ) -> None:
        ConfigurationProcessorFactoryProvider._factories[file_type] = factory

    @staticmethod
    def get_factory(file_type: str) -> IConfigurationProcessorFactory:
        factory = ConfigurationProcessorFactoryProvider._factories.get(file_type)
        if factory:
            return factory
        raise ValueError(f"Unsupported file type: {file_type}")
