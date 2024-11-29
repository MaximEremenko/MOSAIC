# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:18:47 2024

@author: Maksim Eremenko
"""

# plugins/plugin_manager.py

import importlib
import os
from interfaces.mask_strategy import MaskStrategy

class PluginManager:
    def __init__(self, plugin_directory: str):
        self.plugin_directory = plugin_directory
        self.plugins = {}

    def load_plugins(self):
        for filename in os.listdir(self.plugin_directory):
            if filename.endswith('_strategy.py'):
                module_name = filename[:-3]
                module = importlib.import_module(f'strategies.{module_name}')
                for attr in dir(module):
                    cls = getattr(module, attr)
                    if isinstance(cls, type) and issubclass(cls, MaskStrategy) and cls != MaskStrategy:
                        self.plugins[cls.__name__] = cls()
