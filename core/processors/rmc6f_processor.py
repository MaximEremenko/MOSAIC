# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:47:06 2024

@author: Maksim Eremenko
"""

# processors/rmc6f_processor.py

from interfaces.base_interfaces import IConfigurationFileProcessor, IConfigurationDataProcessor
from readers.rmc6f_file_reader import RMC6fFileReader
from parsers.rmc6f_data_parser import RMC6fDataParser
from utilities.rmc6f_metadata_extractor import RMC6fMetadataExtractor
from calculators.cell_calculator import CellCalculator
import pandas as pd
import numpy as np
from typing import Optional, Dict

class RMC6fProcessor( ):
    def __init__(self, file_path: str, data_processor: IConfigurationDataProcessor):
        self.file_path = file_path
        self.data_processor = data_processor
        self.reader = RMC6fFileReader(self.file_path)
        self.parser = RMC6fDataParser()
        self.metadata_extractor = RMC6fMetadataExtractor()
        self.cell_calculator = CellCalculator()
        # Initialize other attributes
        self.content = None
        self.data_frame = None
        self.original_coordinates = None
        self.header_lines = None
        self.metadata = None
        self.supercell = None
        self.vectors = None
        self.metric = None
        self.average_coordinates = None
        self.elements = None
        self.cell_ids = None
    def process(self):
        # Read the file
        self.content = self.reader.read()

        # Parse the data
        self.data_frame = self.parser.parse(self.content)
        self.header_lines = self.parser.header_lines



        # Extract metadata
        self.metadata = self.metadata_extractor.extract(self.header_lines)

        # Store supercell dimensions
        self.supercell = self.metadata.get('supercell')
        if self.supercell is None:
            raise ValueError("Supercell dimensions are missing in the metadata.")

        # Calculate cell vectors and metric
        cell_params = self.metadata.get('cell_params')
        if cell_params:
            self.vectors = self.cell_calculator.calculate_vectors(cell_params)
            self.metric = self.cell_calculator.calculate_metric(self.vectors)
        else:
            raise ValueError("Cell parameters are missing in the metadata.")
            
        # Store the original coordinates before any processing
        self.original_coordinates = self.data_frame[['x', 'y', 'z']].copy()@self.vectors
        self.original_coordinates.columns = self.data_frame[['x', 'y', 'z']].columns
        # Process data to calculate or read average coordinates
        self.data_frame = self.data_processor.process(self.data_frame, self.supercell)

        # Update average_coordinates and elements
        self.average_coordinates = self.data_frame[['x', 'y', 'z']].copy()@self.vectors
        self.average_coordinates.columns = self.data_frame[['x', 'y', 'z']].columns
        self.elements = self.data_frame['element']
        self.refNumbers = self.data_frame['refNumber']
        
        self.cell_ids = self.data_frame[['cellRefNumX', 'cellRefNumY', 'cellRefNumZ']]
        self.cell_ids = self.data_frame[['cellRefNumX', 'cellRefNumY', 'cellRefNumZ']].rename(
                            columns={'cellRefNumX': 'x', 'cellRefNumY': 'y', 'cellRefNumZ': 'z'}
        )

    def get_coordinates(self) -> pd.DataFrame:
        if self.original_coordinates is not None:
            return self.original_coordinates
        else:
            raise ValueError("Original coordinates are not available. Ensure that 'process()' has been called.")

    def get_average_coordinates(self) -> pd.DataFrame:
        return self.average_coordinates[['x', 'y', 'z']]
  
    def get_cells_origin (self) -> pd.DataFrame:
        cells_origin = (self.cell_ids/self.supercell)@self.vectors
        cells_origin.columns = ['x', 'y', 'z']
        return cells_origin
    
    def get_supercell(self) -> np.ndarray:
        if self.supercell is not None:
            return self.supercell
        else:
            raise ValueError("Supercell dimensions are not available. Ensure that 'process()' has been called.")

    def get_elements(self) -> pd.Series:
        return self.elements
    
    def get_cell_ids(self) -> pd.Series:
        return self.cell_ids
    
    def get_refnumbers(self) -> pd.Series:
        return self.refNumbers
    
    def get_vectors(self) -> Optional[np.ndarray]:
        return self.vectors

    def get_metric(self) -> Dict:
        return self.metric
