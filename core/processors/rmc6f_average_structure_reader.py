# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:46:24 2024

@author: Maksim Eremenko
"""

# processors/rmc6f_average_structure_reader.py

from interfaces.base_interfaces import IConfigurationDataProcessor
from readers.rmc6f_file_reader import RMC6fFileReader
from parsers.rmc6f_data_parser import RMC6fDataParser
from utilities.rmc6f_metadata_extractor import RMC6fMetadataExtractor
import pandas as pd
import numpy as np

class RMC6fAverageStructureReader(IConfigurationDataProcessor):
    def __init__(self, average_file_path: str):
        self.average_file_path = average_file_path
        self.reader = RMC6fFileReader(self.average_file_path)
        self.parser = RMC6fDataParser()
        self.metadata_extractor = RMC6fMetadataExtractor()

    def process(self, data_frame: pd.DataFrame, supercell: np.ndarray) -> pd.DataFrame:
        # Read and parse the average file
        content = self.reader.read()
        average_data_frame = self.parser.parse(content)
        header_lines = self.parser.header_lines
        average_metadata = self.metadata_extractor.extract(header_lines)

        # Ensure that the required columns are present
        required_columns = {'atomNumber', 'x', 'y', 'z', 'element'}
        if not required_columns.issubset(average_data_frame.columns):
            raise ValueError("Required columns are missing from the average coordinates file.")

        # Ensure consistency between main data_frame and average_data_frame
        if not data_frame['atomNumber'].equals(average_data_frame['atomNumber']):
            raise ValueError("Atom numbers in main file and average file do not match.")

        # Merge average coordinates into the original data_frame
        data_frame[['x', 'y', 'z']] = average_data_frame[['x', 'y', 'z']].values
        data_frame['element'] = average_data_frame['element']

        return data_frame
