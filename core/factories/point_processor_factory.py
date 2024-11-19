# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:29:44 2024

@author: Maksim Eremenko
"""

# processors/point_processor_factory.py

from typing import Optional
from interfaces.point_parameters_processor_interface import IPointParametersProcessor
from processors.from_average_point_processor import FromAveragePointProcessor
from processors.central_point_processor import CentralPointProcessor
from processors.full_list_point_processor import FullListPointProcessor

class PointProcessorFactory:
    @staticmethod
    def create_processor(method: str, parameters: dict, average_structure: Optional[dict] = None) -> Optional[IPointParametersProcessor]:
        if method == 'from_average':
            rspace_info = parameters.get('rspace_info', {})
            num_chunks = rspace_info.get('num_chunks', 10)  # Default to 10 if not specified
            return FromAveragePointProcessor(parameters, average_structure, num_chunks=num_chunks)
        elif method == 'central':
            return CentralPointProcessor(parameters)
        elif method == 'full_list':
            return FullListPointProcessor(parameters)
        else:
            return None
