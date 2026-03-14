# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:03:23 2024

@author: Maksim Eremenko
"""

# processors/parameters_processor.py

from typing import Optional

from core.infrastructure.storage.hdf5_parameter_storage import (
    HDF5ParameterLoader,
    HDF5ParameterSaver,
)
from core.infrastructure.configuration.contracts.parameter_interfaces import (
    IParameterParser,
    IParameterReader,
)
from core.infrastructure.configuration.parsers.json_parameter_parser import (
    JSONParameterParser,
)
from core.infrastructure.configuration.readers.api_parameter_reader import (
    APIParameterReader,
)
from core.infrastructure.configuration.readers.json_parameter_reader import (
    JSONParameterReader,
)

class ParametersProcessor:
    def __init__(self,
                 reader: Optional[IParameterReader] = None,
                 parser: Optional[IParameterParser] = None,
                 hdf5_file_path: Optional[str] = None):
        self.reader = reader
        self.parser = parser
        self.hdf5_file_path = hdf5_file_path or 'parameters.hdf5'
        self.data = None
        self.data_loader = HDF5ParameterLoader(self.hdf5_file_path)
        self.data_saver = HDF5ParameterSaver(self.hdf5_file_path)

    def process(self):
        if self.reader:
            import logging
            logging.getLogger(__name__).debug("Reading parameters using the configured reader.")
            data = self.reader.read()
            if self.parser:
                self.data = self.parser.parse(data)
            else:
                self.data = data
            return

        if self.data_loader.can_load_data():
            self.data = self.data_loader.load_data()
            return

        raise ValueError(
            f"No parameter source available for '{self.hdf5_file_path}'."
        )

    def get_parameters(self) -> dict:
        return self.data

class ParametersProcessorFactory:
    def create_processor(self,
                         source: str,
                         source_type: str = 'file',
                         hdf5_file_path: Optional[str] = None) -> ParametersProcessor:
        if source_type == 'file':
            reader = JSONParameterReader(source)
            parser = JSONParameterParser()
        elif source_type == 'api':
            reader = APIParameterReader(source)
            parser = JSONParameterParser()
        elif source_type == 'hdf5':
            reader = None  # Reader is not needed; data will be loaded from HDF5
            parser = None
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return ParametersProcessor(reader, parser, hdf5_file_path)

class ParametersProcessorFactoryProvider:
    _factory = ParametersProcessorFactory()

    @staticmethod
    def get_factory() -> ParametersProcessorFactory:
        return ParametersProcessorFactoryProvider._factory
