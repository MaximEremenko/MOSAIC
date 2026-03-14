# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:19:59 2024

@author: Maksim Eremenko
"""

# form_factors/equation_strategy.py
import numpy as np
from abc import ABC, abstractmethod

class EquationStrategy(ABC):
    @abstractmethod
    def compute(self, q_vectors: np.ndarray, params: dict, charge: int = 0) -> np.ndarray:
        pass
