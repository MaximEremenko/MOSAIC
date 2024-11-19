# processors/from_average_point_processor.py

import logging
import os
import h5py
import numpy as np
from typing import List
from interfaces.point_parameters_processor_interface import IPointParametersProcessor
from data_structures.point_data import PointData
from functions.angstrom_to_fractional import angstrom_to_fractional

logger = logging.getLogger(__name__)

class FromAveragePointProcessor(IPointParametersProcessor):
    def __init__(self, parameters: dict, average_structure: dict):
        self.parameters = parameters
        self.average_structure = average_structure
        self.point_data: PointData = None  # Now includes central_point_ids
        self.metric = average_structure.get('metric')
        self.vectors = average_structure.get('vectors')
        self.hdf5_file_path = self.parameters.get('hdf5_file_path', 'point_data.hdf5')

    def process_parameters(self):
        # Check if HDF5 file exists
        if os.path.exists(self.hdf5_file_path):
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                logger.info(f"Loaded point data from {self.hdf5_file_path}")
                return

        logger.info("Processing parameters for 'from_average' method.")
        rspace_info = self.parameters.get('rspace_info', {})
        points_params = rspace_info.get('points', [])

        average_coords = self.average_structure.get('average_coords')
        elements = self.average_structure.get('elements')
        refnumbers = self.average_structure.get('refnumbers')

        if average_coords is None or elements is None or refnumbers is None:
            logger.error("Average structure data is missing required components.")
            return

        # Combine average_coords, elements, and refnumbers into a DataFrame
        structure_df = average_coords.copy()
        structure_df['element'] = elements
        structure_df['refnumbers'] = refnumbers

        all_coordinates = []
        all_dist_from_atom_center = []
        all_step_in_frac = []
        all_central_point_ids = []  # New list to store original point IDs

        for point_info in points_params:
            element_symbol = point_info.get('elementSymbol')
            reference_number = point_info.get('referenceNumber')
            dist_from_atom_center = point_info.get('distFromAtomCenter')
            step_in_angstrom = point_info.get('stepInAngstrom')

            if None in (element_symbol, reference_number, dist_from_atom_center, step_in_angstrom):
                logger.error("Missing required point parameters.")
                continue

            # Create a mask to filter rows where 'refnumbers' and 'element' match
            mask = (structure_df['refnumbers'] == reference_number) & (structure_df['element'] == element_symbol)
            matched_rows = structure_df[mask]

            if not matched_rows.empty:
                dimension = len(dist_from_atom_center)
                coord_columns = ['x', 'y', 'z'][:dimension]
                central_points = matched_rows[coord_columns].values  # Shape: (N, dimension)

                # central_points are already in fractional coordinates
                fractional_coords = central_points  # No need to convert

                # Convert dist_from_atom_center and step_in_angstrom to fractional units
                dist_from_atom_center_array = np.array(dist_from_atom_center).reshape(1, -1)  # Shape: (1, D)
                step_in_angstrom_array = np.array(step_in_angstrom).reshape(1, -1)          # Shape: (1, D)

                # Convert to fractional units
                fractional_dist = angstrom_to_fractional(dist_from_atom_center_array, self.vectors)  # Shape: (1, D)
                fractional_step = angstrom_to_fractional(step_in_angstrom_array, self.vectors)      # Shape: (1, D)

                # Repeat dist and step for each central point
                num_points = fractional_coords.shape[0]
                all_coordinates.append(fractional_coords)
                all_dist_from_atom_center.append(np.tile(fractional_dist, (num_points, 1)))  # Shape: (num_points, D)
                all_step_in_frac.append(np.tile(fractional_step, (num_points, 1)))          # Shape: (num_points, D)

                # Collect the original point IDs
                central_point_ids = matched_rows['refnumbers'].values  # Shape: (N,)
                all_central_point_ids.append(central_point_ids)
            else:
                logger.warning(f"Reference atom not found for element {element_symbol} and reference number {reference_number}")

        if all_coordinates:
            # Concatenate all data
            coordinates_array = np.vstack(all_coordinates)                           # Shape: (Total_N, D)
            dist_from_atom_center_array = np.vstack(all_dist_from_atom_center)         # Shape: (Total_N, D)
            step_in_frac_array = np.vstack(all_step_in_frac)                           # Shape: (Total_N, D)
            central_point_ids_array = np.concatenate(all_central_point_ids)           # Shape: (Total_N,)

            # Create a single PointData instance with central_point_ids
            self.point_data = PointData(
                coordinates=coordinates_array,
                dist_from_atom_center=dist_from_atom_center_array,
                step_in_frac=step_in_frac_array,
                central_point_ids=central_point_ids_array
            )
        else:
            logger.error("No point data generated.")

        # Save to HDF5
        self.save_point_data_to_hdf5(self.hdf5_file_path)

    def get_point_data(self) -> PointData:
        return self.point_data

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        try:
            with h5py.File(hdf5_file_path, 'w') as h5file:
                h5file.create_dataset('coordinates', data=self.point_data.coordinates)
                h5file.create_dataset('dist_from_atom_center', data=self.point_data.dist_from_atom_center)
                h5file.create_dataset('step_in_frac', data=self.point_data.step_in_frac)
                h5file.create_dataset('central_point_ids', data=self.point_data.central_point_ids)
            logger.info(f"Point data saved to HDF5 file: {hdf5_file_path}")
        except Exception as e:
            logger.error(f"Failed to save point data to HDF5 file: {e}")

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        try:
            with h5py.File(hdf5_file_path, 'r') as h5file:
                coordinates = h5file['coordinates'][:]
                dist_from_atom_center = h5file['dist_from_atom_center'][:]
                step_in_frac_array = h5file['step_in_frac'][:]
                central_point_ids = h5file['central_point_ids'][:]
                self.point_data = PointData(
                    coordinates=coordinates,
                    dist_from_atom_center=dist_from_atom_center,
                    step_in_frac=step_in_frac_array,
                    central_point_ids=central_point_ids
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load point data from HDF5 file: {e}")
            return False
