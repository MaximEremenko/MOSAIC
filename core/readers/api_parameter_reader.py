# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:01:28 2024

@author: Maksim Eremenko
"""

import requests
from interfaces.parameter_interfaces import IParameterReader

class APIParameterReader(IParameterReader):
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint

    def read(self) -> dict:
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            print(f"Failed to fetch parameters from API: {e}")
            raise