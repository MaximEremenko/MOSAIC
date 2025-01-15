# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:25:26 2024

@author: Maksim Eremenko
"""
#processors/configuration2d_file_processor.py
from interfaces.base_interfaces import IConfigurationFileProcessor
from calculators.cell_calculator import CellCalculator
from readers.file_reader2d import FileReader2D
from parsers.data_2d_parser import DataParser2D
import pandas as pd

class ConfigurationFileProcessor2D(IConfigurationFileProcessor):
    def __init__(self, file_path: str):
        """
        Processor for 2D configuration files.
        
        Args:
            file_path (str): Path to the input file.
        """
        self.file_path = file_path
        self.reader = FileReader2D(file_path)
        self.parser = DataParser2D()
        self.cell_calculator = CellCalculator()
        self.metadata = {}
        self.data = None
        self.vectors = None
        self.metric = None

    def process(self):
        """
        Reads the file, parses metadata and data, and computes lattice vectors and metrics.
        """
        # Step 1: Read file
        content = self.reader.read()

        # Step 2: Parse file
        self.metadata, self.data = self.parser.parse(content)

        # Step 3: Calculate vectors and metrics
        cell_params = self.metadata['cell_params']
        self.vectors = self.cell_calculator.calculate_vectors(cell_params)
        self.metric = self.cell_calculator.calculate_metric(self.vectors)
        self.data[['X', 'Y']]      = (self.data[['X', 'Y']].values).dot(self.vectors)
        self.data[['Xav', 'Yav']]  = (self.data[['Xav', 'Yav']].values).dot(self.vectors)
    def get_metadata(self):
        """
        Returns metadata extracted from the file.

        Returns:
            dict: Metadata containing 'supercell' and 'cell_params'.
        """
        return self.metadata

    def get_data(self):
        """
        Returns the parsed data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data table from the file.
        """
        return self.data

    def get_vectors(self):
        """
        Returns the lattice vectors.

        Returns:
            np.ndarray: Calculated lattice vectors.
        """
        return self.vectors

    def get_metric(self):
        """
        Returns the calculated reciprocal lattice and volume/area.

        Returns:
            dict: Metric containing reciprocal vectors and size (volume/area).
        """
        return self.metric

    def get_coordinates(self) -> pd.DataFrame:
        """
        Returns the 'X' and 'Y' coordinates from the parsed data.

        Returns:
            pd.DataFrame: DataFrame containing coordinates.
        """
        return self.data[['X', 'Y']]

    def get_average_coordinates(self) -> pd.DataFrame:
        """
        Returns the average coordinates ('Xav' and 'Yav') from the parsed data.

        Returns:
            pd.DataFrame: DataFrame containing average coordinates.
        """
        return self.data[['Xav', 'Yav']]

    def get_elements(self) -> pd.Series:
        """
        Returns the elements column.

        Returns:
            pd.Series: Series containing the element data.
        """
        return self.data['Element']

    def get_supercell(self) -> list:
        """
        Returns the supercell dimensions.

        Returns:
            list: List containing the supercell dimensions.
        """
        return self.metadata['supercell']

    def get_refnumbers(self) -> pd.Series:
        """
        Returns the reference numbers (RefNumber) column.

        Returns:
            pd.Series: Series containing reference numbers.
        """
        return self.data['RefNumber']
