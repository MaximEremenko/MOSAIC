# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:25:26 2024

@author: Maksim Eremenko
"""
#processors/configuration2d_file_processor.py
from interfaces.base_interfaces import IConfigurationFileProcessor
from calculators.cell_calculator import CellCalculator
from readers.file_reader1d import FileReader1D
from parsers.data_1d_parser import DataParser1D
import pandas as pd

class ConfigurationFileProcessor1D(IConfigurationFileProcessor):
    def __init__(self, file_path: str):
        """
        Processor for 2D configuration files.
        
        Args:
            file_path (str): Path to the input file.
        """
        self.file_path = file_path
        self.reader = FileReader1D(file_path)
        self.parser = DataParser1D()
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

        # Transform X and Xav based on the lattice vectors
        self.data[['X']] = (self.data[['X']].values).dot(self.vectors)
        self.data[['Xav']] = (self.data[['Xav']].values).dot(self.vectors)

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
        Returns the 'X' coordinates from the parsed data.

        Returns:
            pd.DataFrame: DataFrame containing coordinates.
        """
        return self.data[['X']]

    def get_average_coordinates(self) -> pd.DataFrame:
        """
        Returns the average coordinates ('Xav') from the parsed data.

        Returns:
            pd.DataFrame: DataFrame containing average coordinates.
        """
        return self.data[['Xav']]

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

    def get_coeff(self) -> pd.Series or None:
        """
        Returns the 'Coeff' column if present; otherwise returns None.

        Returns:
            pd.Series or None: The Coeff column if present, else None.
        """
        if 'Coeff' in self.data.columns:
            return self.data['Coeff']
        return None
