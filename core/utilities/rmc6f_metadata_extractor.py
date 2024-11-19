# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:44:49 2024

@author: Maksim Eremenko
"""
# utilities/rmc6f_metadata_extractor.py

from interfaces.base_interfaces import IMetadataExtractor
from typing import List, Dict
import numpy as np

class RMC6fMetadataExtractor(IMetadataExtractor):
    def extract(self, header_lines: List[str]) -> Dict:
        metadata = {}
        for line in header_lines:
            if "Supercell" in line:
                supercell = [int(s) for s in line.split() if s.isdigit()]
                metadata['supercell'] = np.array(supercell)
            elif "Cell" in line:
                cell_params = [float(s) for s in line.split() if self._is_float(s)]
                if len(cell_params) == 6:
                    metadata['cell_params'] = cell_params
                else:
                    raise ValueError("Invalid cell parameters.")
            elif "Atom types present:" in line:
                atom_types = line.replace("Atom types present:", "").strip().split()
                metadata['atom_types'] = atom_types
            elif "Metadata owner:" in line:
                metadata['owner'] = line.replace("Metadata owner:", "").strip()
            elif "Metadata date:" in line:
                metadata['date'] = line.replace("Metadata date:", "").strip()
        return metadata

    def _is_float(self, string: str) -> bool:
        try:
            float(string)
            return True
        except ValueError:
            return False

