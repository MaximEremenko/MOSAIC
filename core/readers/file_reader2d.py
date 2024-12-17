# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:21:19 2024

@author: Maksim Eremenko
"""
# readers/file_reader2d.py
from interfaces.base_interfaces import IFileReader

class FileReader2D(IFileReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> str:
        with open(self.file_path, 'r') as file:
            content = file.read()
        return content
