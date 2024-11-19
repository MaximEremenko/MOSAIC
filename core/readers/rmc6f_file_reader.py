# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:57:06 2024

@author: Maksim Eremenko
"""
# readers/rmc6f_file_reader.py

from interfaces.base_interfaces import IFileReader

class RMC6fFileReader(IFileReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> str:
        with open(self.file_path, 'r') as file:
            content = file.read()
        return content