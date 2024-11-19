# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:57:59 2024

@author: Maksim Eremenko
"""
# data_saver.py

import os
import numpy as np

class RIFFTInDataSaver:
    """
    Saves data to files and generates file paths.
    """

    def __init__(self, output_dir, file_extension='npy'):
        """
        Initializes the RIFFTInDataSaver.

        Args:
            output_dir (str): Directory where files will be saved.
            file_extension (str): Extension of the saved files (e.g., 'npy', 'hdf5').
        """
        self.output_dir = output_dir
        self.file_extension = file_extension

    def generate_filename(self, chunk_id, suffix=''):
        """
        Generates a filename based on chunk ID and an optional suffix.

        Args:
            chunk_id (int): ID of the chunk.
            suffix (str): Optional suffix to differentiate file types (e.g., '_grid', '_amplitudes').

        Returns:
            str: Generated filename.
        """
        if suffix:
            filename = f'point_data_chunk_{chunk_id}{suffix}.{self.file_extension}'
        else:
            filename = f'point_data_chunk_{chunk_id}.{self.file_extension}'
        return filename

    def save_data(self, data, filename):
        """
        Saves the provided data to a file with the given filename.

        Args:
            data (np.ndarray): Data to be saved.
            filename (str): Name of the file.

        Returns:
            str: Full path to the saved file.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_path = os.path.join(self.output_dir, filename)
        if self.file_extension == 'npy':
            np.save(file_path, data)
        elif self.file_extension == 'hdf5':
            import h5py
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('data', data=data)
        else:
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
        return file_path
