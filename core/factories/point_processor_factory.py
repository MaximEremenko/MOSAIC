# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:29:44 2024

@author: Maksim Eremenko
"""

# processors/point_processor_factory.py

from typing import Optional

from core.application.point_processor_registry import POINT_PROCESSOR_REGISTRY
from core.domain.interfaces.point_processors import PointProcessor

class PointProcessorFactory:
    @staticmethod
    def create_processor(
        method: str,
        parameters: dict,
        average_structure: Optional[dict] = None,
    ) -> PointProcessor:
        return POINT_PROCESSOR_REGISTRY.create(method, parameters, average_structure)
