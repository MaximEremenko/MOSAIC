# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:25:33 2024

@author: Maksim Eremenko
"""
from typing import Optional
from interfaces.base_interfaces import IDataProcessor


class ConfigurationFileDataProcessor(IDataProcessor):
    def __init__(self, file_path: str, processor_type: str, average_file_path: Optional[str]):
        self.file_path = file_path
        self.processor_type = processor_type
        self.average_file_path = average_file_path

    def process_data(self):
        # Process configuration files using the processor
        pass
