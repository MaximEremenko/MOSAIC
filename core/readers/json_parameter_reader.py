# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:51:08 2024

@author: Maksim Eremenko
"""

# readers/json_parameter_reader.py

import json
from interfaces.parameter_interfaces import IParameterReader

class JSONParameterReader(IParameterReader):
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path

    def read(self) -> dict:
        try:
            with open(self.json_file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            raise
            
            
