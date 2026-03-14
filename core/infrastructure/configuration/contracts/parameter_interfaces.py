# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:34:35 2024

@author: Maksim Eremenko
"""

from abc import ABC, abstractmethod

class IParameterReader(ABC):
    @abstractmethod
    def read(self) -> dict:
        pass

class IParameterParser(ABC):
    @abstractmethod
    def parse(self, data: dict) -> dict:
        pass