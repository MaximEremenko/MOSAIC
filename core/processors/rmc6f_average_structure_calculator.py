# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:45:47 2024

@author: Maksim Eremenko
"""
# processors/rmc6f_average_structure_calculator.py

from interfaces.base_interfaces import IConfigurationDataProcessor
import pandas as pd
import numpy as np

class RMC6fAverageStructureCalculator(IConfigurationDataProcessor):
    def process(self, data_frame: pd.DataFrame, supercell: np.ndarray) -> pd.DataFrame:
        df_average = data_frame.copy()
        elements = df_average['element'].unique()
        ref_numbers = df_average['refNumber'].unique()

        for element in elements:
            for ref_num in ref_numbers:
                mask = (df_average['element'] == element) & (df_average['refNumber'] == ref_num)
                if not mask.any():
                    continue

                cell_refs = df_average.loc[mask, ['cellRefNumX', 'cellRefNumY', 'cellRefNumZ']] / supercell
                delta = df_average.loc[mask, ['x', 'y', 'z']] - cell_refs.values
                delta = delta.map(lambda x: x + 1 if x < -0.5 else x - 1 if x > 0.5 else x)
                avg_delta = delta.mean()
                corrected_positions = cell_refs + avg_delta.values
                corrected_positions = corrected_positions.map(
                    lambda x: x + 1 if x < 0 else x - 1 if x > 1 else x
                )

                df_average.loc[mask, ['x', 'y', 'z']] = corrected_positions.values

        return df_average