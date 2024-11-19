# processors/full_list_point_processor.py

import logging
import os
import h5py
import numpy as np
from typing import List
from interfaces.point_parameters_processor_interface import IPointParametersProcessor
from data_structures.point_data import PointData
from functions.angstrom_to_fractional import angstrom_to_fractional

logger = logging.getLogger(__name__)

class FullListPointProcessor(IPointParametersProcessor):
    def __init__(self, parameters: dict, average_structure: dict):
        self.parameters = parameters
        self.average_structure = average_structure
        self.point_data: PointData = None  # Now a single PointData instance
        self.vectors = average_structure.get('vectors')
        self.hdf5_file_path = self.parameters.get('hdf5_file_path', 'point_data.hdf5')

    def process_parameters(self):
        # Check if HDF5 file exists
        if os.path.exists(self.hdf5_file_path):
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                logger.info(f"Loaded point data from {self.hdf5_file_path}")
                return

        logger.info("Processing parameters for 'full_list' method.")
        rspace_info = self.parameters.get('rspace_info', {})
        points_params = rspace_info.get('points', [])

        all_coordinates = []
        all_dist_from_atom_center = []
        all_step_in_angstrom = []

        for point_info in points_params:
            file_name = point_info.get('file name')
            if not file_name:
                logger.error("Missing 'file name' in point parameters.")
                continue

            if not os.path.exists(file_name):
                logger.error(f"Points file not found: {file_name}")
                continue

            points = self._read_points_file(file_name)
            if points.size == 0:
                logger.warning(f"No points loaded from {file_name}")
                continue

            dimension = points.shape[1] if points.ndim > 1 else 1

            # Convert coordinates to fractional
            fractional_coords = angstrom_to_fractional(points, self.vectors)

            # Append data to the arrays
            all_coordinates.append(fractional_coords)
            num_points = fractional_coords.shape[0]
            zero_array = np.zeros((num_points, dimension))
            all_dist_from_atom_center.append(zero_array)
            all_step_in_angstrom.append(zero_array)

        if all_coordinates:
            # Concatenate all data
            coordinates_array = np.vstack(all_coordinates)
            dist_from_atom_center_array = np.vstack(all_dist_from_atom_center)
            step_in_angstrom_array = np.vstack(all_step_in_angstrom)

            # Create a single PointData instance
            self.point_data = PointData(
                coordinates=coordinates_array,
                dist_from_atom_center=dist_from_atom_center_array,
                step_in_angstrom=step_in_angstrom_array
            )
        else:
            logger.error("No point data generated.")

        # Save to HDF5
        self.save_point_data_to_hdf5(self.hdf5_file_path)

    def _read_points_file(self, file_name: str) -> np.ndarray:
        try:
            points = np.loadtxt(file_name)
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            logger.info(f"Loaded {points.shape[0]} points from {file_name}.")
            return points
        except Exception as e:
            logger.error(f"Failed to read points from file {file_name}: {e}")
            return np.array([])

    def get_point_data(self) -> PointData:
        return self.point_data

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        try:
            if self.point_data is None:
                logger.error("No point data to save.")
                return
            with h5py.File(hdf5_file_path, 'w') as h5file:
                h5file.create_dataset('coordinates', data=self.point_data.coordinates)
                h5file.create_dataset('dist_from_atom_center', data=self.point_data.dist_from_atom_center)
                h5file.create_dataset('step_in_angstrom', data=self.point_data.step_in_angstrom)
            logger.info(f"Point data saved to HDF5 file: {hdf5_file_path}")
        except Exception as e:
            logger.error(f"Failed to save point data to HDF5 file: {e}")

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        try:
            with h5py.File(hdf5_file_path, 'r') as h5file:
                coordinates = h5file['coordinates'][:]
                dist_from_atom_center = h5file['dist_from_atom_center'][:]
                step_in_angstrom = h5file['step_in_angstrom'][:]
                self.point_data = PointData(
                    coordinates=coordinates,
                    dist_from_atom_center=dist_from_atom_center,
                    step_in_angstrom=step_in_angstrom
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load point data from HDF5 file: {e}")
            return False
