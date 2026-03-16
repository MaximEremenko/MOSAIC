# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:17:45 2024

@author: Maksim Eremenko
"""

# form_factors/parameter_provider.py
from abc import ABC, abstractmethod

class ParameterProvider(ABC):
    @abstractmethod
    def get_parameters(self, element: str, charge: int = 0) -> dict:
        pass