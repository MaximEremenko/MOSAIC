# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:34:39 2024

@author: Maksim Eremenko
"""

from interfaces.base_interfaces import IConfigurationProcessorFactory, IConfigurationFileProcessor
from processors.hdf5_processor import HDF5Processor
from typing import Optional

class HDF5ProcessorFactory(IConfigurationProcessorFactory):
    def create_processor(self,
                         file_path: str,
                         processor_type: str = 'calculate',
                         average_file_path: Optional[str] = None) -> IConfigurationFileProcessor:
        """
        For HDF5ProcessorFactory, the processor_type and average_file_path are irrelevant.
        The file_path here refers to the HDF5 file path.
        """
        return HDF5Processor(file_path)