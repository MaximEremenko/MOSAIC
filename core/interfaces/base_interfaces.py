# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:42:04 2024

@author: Maksim Eremenko
"""

# interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
#import h5py

class IFileReader(ABC):
    @abstractmethod
    def read(self) -> str:
        pass
    

    
class IConfigurationFileParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> pd.DataFrame:
        pass

class IMetadataExtractor(ABC):
    @abstractmethod
    def extract(self, header_lines: List[str]) -> Dict:
        pass

class IConfigurationDataProcessor(ABC):
    @abstractmethod
    def process(self, data_frame: pd.DataFrame, supercell: Optional[np.ndarray]) -> pd.DataFrame:
        pass

class IConfigurationFileProcessor(ABC):
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def get_coordinates(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_average_coordinates(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_supercell(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_elements(self) -> pd.Series:
        pass

    @abstractmethod
    def get_vectors(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def get_metric(self) -> Dict:
        pass

class IConfigurationProcessorFactory(ABC):
    @abstractmethod
    def create_processor(self, 
                         file_path: str,
                         processor_type: str = 'calculate',
                         average_file_path: Optional[str] = None) -> 'IConfigurationFileProcessor':
        pass

class IConfigDataSaver(ABC):
    @abstractmethod
    def save_data(self, data):
        pass
 
class IConfigDataLoader(ABC):
    @abstractmethod
    def can_load_data(self) -> bool:
        pass

    @abstractmethod
    def load_data(self):
        pass

class IParametersFileProcessor(ABC):
    @abstractmethod
    def process(self):
        pass
    
class IParametersProcessorFactory(ABC):
    @abstractmethod
    def create_processor(self, 
                         file_path: str) -> 'IParametersFileProcessor':
        pass

class IParameterReader(ABC):
    @abstractmethod
    def read(self) -> dict:
        pass

class IParameterParser(ABC):
    @abstractmethod
    def parse(self, data: dict) -> dict:
        pass
    


