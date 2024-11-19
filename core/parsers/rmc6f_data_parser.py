# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:00:59 2024

@author: Maksim Eremenko
"""

# parsers/rmc6f_data_parser.py

from interfaces.base_interfaces import IConfigurationFileParser
import pandas as pd
import warnings
from io import StringIO

class RMC6fDataParser(IConfigurationFileParser):
    def __init__(self):
        self.header_lines = []

    def parse(self, content: str) -> pd.DataFrame:
        lines = content.splitlines()
        header_lines = []
        data_lines = []
        skiprows = None

        for i, line in enumerate(lines):
            if "Atoms:" in line:
                skiprows = i + 1  # Data starts after "Atoms:"
                header_lines = lines[:i+1]
                data_lines = lines[i+1:]
                break
            if i >= 150:
                break

        if skiprows is None:
            raise ValueError("Could not find 'Atoms:' section in the file.")

        self.header_lines = header_lines  # Store for metadata extraction

        data_str = '\n'.join(data_lines)
        df = pd.read_csv(
            StringIO(data_str),
            header=None,
            sep='\s+',
            engine='python'
        )

        if df.shape[1] == 10:
            df.columns = ['atomNumber', 'element', 'id', 'x', 'y', 'z',
                          'refNumber', 'cellRefNumX', 'cellRefNumY', 'cellRefNumZ']
        elif df.shape[1] == 9:
            df.columns = ['atomNumber', 'element', 'x', 'y', 'z',
                          'refNumber', 'cellRefNumX', 'cellRefNumY', 'cellRefNumZ']
        else:
            warnings.warn("Unsupported RMC6f format")
            raise ValueError("Unsupported RMC6f format.")

        return df
