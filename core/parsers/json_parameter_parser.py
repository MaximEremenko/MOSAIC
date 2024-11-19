# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:52:04 2024

@author: Maksim Eremenko
"""

# parsers/json_parameter_parser.py

from interfaces.parameter_interfaces import IParameterParser

class JSONParameterParser(IParameterParser):
    def parse(self, data: dict) -> dict:
        #TODO:  Validation and transformation logic here if needed
        return data
    