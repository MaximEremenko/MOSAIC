# data_storage/rifft_in_data_saver.py

import os
import h5py
import numpy as np
import logging

class RIFFTInDataSaver:
    """
    Saves data to HDF5 files and generates file paths.
    """

    def __init__(self, output_dir, file_extension='hdf5'):
        """
        Initializes the RIFFTInDataSaver.

        Args:
            output_dir (str): Directory where files will be saved.
            file_extension (str): Extension of the saved files (e.g., 'npy', 'hdf5').
        """
        self.output_dir = output_dir
        self.file_extension = file_extension
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store the HDF5 file path for saving grid_amplitude_initialized
        self.hdf5_file_path = os.path.join(output_dir, 'point_data.hdf5')

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

    def save_data(self, data, filename, append=False):
        """
        Saves the provided data to a file with the given filename.

        Args:
            data (dict): Dictionary containing datasets to be saved.
            filename (str): Name of the file.
            append (bool): Whether to append data to existing datasets.

        Returns:
            str: Full path to the saved file.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_path = os.path.join(self.output_dir, filename)
        if self.file_extension == 'hdf5':
            try:
                if append and os.path.exists(file_path):
                    mode = 'a'  # Append mode
                    self.logger.debug(f"Appending data to existing HDF5 file: {file_path}")
                else:
                    mode = 'w'  # Overwrite/Create mode
                    self.logger.debug(f"{'Overwriting' if os.path.exists(file_path) else 'Creating'} HDF5 file: {file_path}")

                with h5py.File(file_path, mode) as h5file:
                    for dataset_name, dataset_data in data.items():
                        if append and dataset_name in h5file:
                            # Resize dataset to accommodate new data
                            if dataset_data.ndim > 1:
                                new_shape = (h5file[dataset_name].shape[0] + dataset_data.shape[0],) + h5file[dataset_name].shape[1:]
                            else:
                                new_shape = (h5file[dataset_name].shape[0] + dataset_data.shape[0],)
                            h5file[dataset_name].resize(new_shape)
                            h5file[dataset_name][-dataset_data.shape[0]:] = dataset_data
                            self.logger.debug(f"Appended {dataset_data.shape[0]} entries to dataset '{dataset_name}'. New shape: {h5file[dataset_name].shape}")
                        else:
                            # Create or overwrite dataset with maxshape to allow future appends
                            if dataset_data.ndim > 1:
                                maxshape = (None,) + dataset_data.shape[1:]
                            else:
                                maxshape = (None,)
                            h5file.create_dataset(dataset_name, data=dataset_data, maxshape=maxshape, chunks=True)
                            self.logger.debug(f"Dataset '{dataset_name}' {'created' if mode == 'w' else 'overwritten'} with shape {dataset_data.shape}")
                self.logger.debug(f"Data saved to HDF5 file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save data to HDF5 file: {e}")
                raise e
        elif self.file_extension == 'npy':
            if isinstance(data, np.ndarray):
                if append and os.path.exists(file_path):
                    # NPY files do not support appending. Consider using a different format or manage appending externally.
                    self.logger.error("Appending to NPY files is not supported.")
                    raise NotImplementedError("Appending to NPY files is not supported.")
                else:
                    np.save(file_path, data)
                    self.logger.debug(f"Data saved to NPY file: {file_path}")
            else:
                self.logger.error("Only NumPy arrays can be saved as NPY files.")
                raise ValueError("Only NumPy arrays can be saved as NPY files.")
        else:
            self.logger.error(f"Unsupported file extension: {self.file_extension}")
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
        return file_path
    def load_data(self, filename):
        """
        Loads data from a file with the given filename.
    
        Args:
            filename (str): Name of the file to load data from.
    
        Returns:
            dict or np.ndarray: The loaded data. If the file is HDF5, a dictionary of datasets is returned.
                                If the file is NPY, a NumPy array is returned.
    
        Raises:
            ValueError: If the file extension is unsupported or if the file does not exist.
        """
        file_path = os.path.join(self.output_dir, filename)
    
        if not os.path.exists(file_path):
            self.logger.info(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
    
        if self.file_extension == 'hdf5':
            try:
                self.logger.debug(f"Loading data from HDF5 file: {file_path}")
                data = {}
                with h5py.File(file_path, 'r') as h5file:
                    for dataset_name in h5file.keys():
                        data[dataset_name] = h5file[dataset_name][:]
                        self.logger.debug(f"Loaded dataset '{dataset_name}' with shape {data[dataset_name].shape}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load data from HDF5 file: {e}")
                raise e
        elif self.file_extension == 'npy':
            try:
                self.logger.debug(f"Loading data from NPY file: {file_path}")
                data = np.load(file_path, allow_pickle=True)
                self.logger.debug(f"Loaded NPY array with shape {data.shape}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load data from NPY file: {e}")
                raise e
        else:
            self.logger.error(f"Unsupported file extension: {self.file_extension}")
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
