# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:23:08 2024

@author: Maksim Eremenko
"""
#parsers/data_1d_parser.py

from interfaces.base_interfaces import IConfigurationFileParser
import pandas as pd
import re
from io import StringIO
from typing import Tuple

class DataParser1D(IConfigurationFileParser):
    def __init__(self):
        self.header_lines = []

    def parse(self, content: str) -> Tuple[dict, pd.DataFrame]:
        lines = content.splitlines()
        metadata = {}
        data_lines = []

        # Extract headers
        for i, line in enumerate(lines):
            if line.startswith("Supercell dimensions"):
                metadata['supercell'] = self._extract_supercell(line)
            elif line.startswith("Cell (Ang/deg)"):
                metadata['cell_params'] = self._extract_cell_params(line)
            elif "Element" in line:  # Start of data table
                data_lines = lines[i:]
                break

        # Read data into DataFrame
        data_str = '\n'.join(data_lines)
        df = pd.read_csv(StringIO(data_str), sep='\s+')

        return metadata, df

    def _extract_supercell(self, line: str) -> list:
        match = re.search(r"Supercell dimensions:\s+(\d+)", line)
        return [int(match.group(1))]

    def _extract_cell_params(self, line: str) -> list:
        match = re.search(r"Cell \(Ang/deg\):\s+([\d\.\-E]+)", line)
        return [float(match.group(1))]
