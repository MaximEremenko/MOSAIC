# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:22:04 2024

@author: Maksim Eremenko
"""

# form_factors/json_parameter_provider.py
import json
from form_factors.parameter_provider import ParameterProvider

class JSONParameterProvider(ParameterProvider):
    def __init__(self, filename: str):
        with open(filename, 'r') as file:
            self.parameters = json.load(file)

    def get_parameters(self, element: str, charge: int = 0) -> dict:
        element_data = self.parameters.get(element.lower())
        if element_data is None:
            raise ValueError(f"Parameters for element '{element}' not found.")
        charge_key = str(charge)
        params = element_data.get(charge_key)
        if params is None:
            params = element_data.get('0')
            if params is None:
                raise ValueError(f"Parameters for element '{element}' with charge {charge} not found.")
        return params
