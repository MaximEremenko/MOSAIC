# processors/from_average_point_processor.py
from typing import Optional
import logging
import os
import h5py
import numpy as np
from interfaces.point_parameters_processor_interface import IPointParametersProcessor
from data_structures.point_data import PointData
from functions.angstrom_to_fractional import angstrom_to_fractional



class FromAveragePointProcessor(IPointParametersProcessor):
    def __init__(self, parameters: dict, average_structure: dict, num_chunks: int = 10):
        self.parameters = parameters
        self.average_structure = average_structure
        self.num_chunks = num_chunks
        self.point_data: PointData = None  # Includes central_point_ids and grid_amplitude_initialized
        self.metric = average_structure.get('metric')
        self.vectors = average_structure.get('vectors')
        self.hdf5_file_path = self.parameters.get('hdf5_file_path', 'point_data.hdf5')
        self.logger = logging.getLogger(__name__)
    def process_parameters(self):
        # Extract 'fresh_start' flag from rspace_info
        rspace_info = self.parameters.get('rspace_info', {})
        fresh_start = rspace_info.get('fresh_start', True)  # Default to True if not specified

        # Check if HDF5 file exists and handle accordingly
        if os.path.exists(self.hdf5_file_path) and not fresh_start:
            self.logger.info(f"HDF5 file found: {self.hdf5_file_path}. Attempting to load existing data for appending.")
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                self.logger.info(f"Existing point data loaded from {self.hdf5_file_path}. New points will be appended.")
            else:
                self.logger.warning(f"Failed to load existing point data from {self.hdf5_file_path}. Proceeding to create new data.")
                fresh_start = True  # Fallback to fresh start if loading fails

        if fresh_start or self.point_data is None:
            self.logger.info("Processing parameters for 'from_average' method with fresh start.")
            # Initialize empty PointData
            self.point_data = PointData(
                coordinates=np.empty((0, 3)),
                dist_from_atom_center=np.empty((0, 3)),
                step_in_frac=np.empty((0, 3)),
                central_point_ids=np.empty((0,), dtype=int),
                chunk_ids=np.empty((0,), dtype=int),
                grid_amplitude_initialized=np.empty((0,), dtype=bool)
            )

        # Proceed to process parameters
        self.logger.info("Processing parameters for 'from_average' method.")
        points_params = rspace_info.get('points', [])

        average_coords = self.average_structure.get('average_coords')
        elements = self.average_structure.get('elements')
        refnumbers = self.average_structure.get('refnumbers')

        if average_coords is None or elements is None or refnumbers is None:
            self.logger.error("Average structure data is missing required components.")
            return

        # Combine average_coords, elements, and refnumbers into a DataFrame
        # Assuming average_coords is a pandas DataFrame or similar structure
        # If not, adjust accordingly
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
                self.logger.error("Missing required point parameters.")
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

                # Collect the original point IDs (assuming DataFrame index)
                central_point_ids = matched_rows.index.values  # Shape: (N,)
                all_central_point_ids.append(central_point_ids)
            else:
                self.logger.warning(f"Reference atom not found for element {element_symbol} and reference number {reference_number}")

        if all_coordinates:
            # Concatenate all data
            coordinates_array = np.vstack(all_coordinates)                           # Shape: (Total_N, D)
            dist_from_atom_center_array = np.vstack(all_dist_from_atom_center)         # Shape: (Total_N, D)
            step_in_frac_array = np.vstack(all_step_in_frac)                           # Shape: (Total_N, D)
            central_point_ids_array = np.concatenate(all_central_point_ids)           # Shape: (Total_N,)
            grid_amplitude_initialized_array = np.zeros(central_point_ids_array.shape[0], dtype=bool)

            # Create a new PointData instance
            new_point_data = PointData(
                coordinates=coordinates_array,
                dist_from_atom_center=dist_from_atom_center_array,
                step_in_frac=step_in_frac_array,
                central_point_ids=central_point_ids_array,
                chunk_ids=np.empty((central_point_ids_array.shape[0],), dtype=int),  # Placeholder, to assign later
                grid_amplitude_initialized=grid_amplitude_initialized_array
            )

            # Assign chunk_ids to new points based on num_chunks
            self._assign_chunk_ids(new_point_data)

            # Merge new_point_data with existing self.point_data
            self._merge_point_data(new_point_data)
        else:
            self.logger.error("No point data generated.")

        # Save to HDF5
        self.save_point_data_to_hdf5(self.hdf5_file_path)

    def _assign_chunk_ids(self, new_point_data: PointData, max_chunk_size: Optional[int] = None):
        """
        Assigns chunk_ids to new points based on the specified number of chunks.

        Args:
            new_point_data (PointData): The new point data to assign chunk_ids to.
            max_chunk_size (int, optional): If provided, overrides num_chunks by specifying maximum points per chunk.
        """
        num_new_points = new_point_data.coordinates.shape[0]
        num_chunks = self.num_chunks

        if max_chunk_size:
            num_chunks = int(np.ceil(num_new_points / max_chunk_size))
            self.logger.debug(f"Overriding num_chunks to {num_chunks} based on max_chunk_size={max_chunk_size}")

        self.logger.debug(f"Assigning chunk_ids to {num_new_points} new points with num_chunks={num_chunks}")

        if self.point_data is None or self.point_data.chunk_ids.size == 0:
            current_max_chunk_id = -1
        else:
            current_max_chunk_id = self.point_data.chunk_ids.max()

        # Calculate the size of each chunk
        base_chunk_size = num_new_points // num_chunks
        remainder = num_new_points % num_chunks

        chunk_sizes = [base_chunk_size + 1 if i < remainder else base_chunk_size for i in range(num_chunks)]
        self.logger.debug(f"Chunk sizes: {chunk_sizes}")

        start = 0
        for i, size in enumerate(chunk_sizes):
            end = start + size
            chunk_id = current_max_chunk_id + 1 + i
            new_point_data.chunk_ids[start:end] = chunk_id
            self.logger.debug(f"Assigned chunk_id={chunk_id} to points {start} to {end-1}")
            start = end

    def _merge_point_data(self, new_point_data: PointData):
        """
        Merges new_point_data with existing self.point_data, avoiding duplicates.

        Args:
            new_point_data (PointData): The new point data to merge.
        """
        if self.point_data is None or self.point_data.central_point_ids.size == 0:
            self.point_data = new_point_data
            self.logger.debug("Initialized self.point_data with new_point_data.")
            return

        # Existing point_ids
        existing_ids = set(self.point_data.central_point_ids)

        # Identify unique new points
        unique_mask = ~np.isin(new_point_data.central_point_ids, list(existing_ids))
        unique_indices = np.where(unique_mask)[0]

        if unique_indices.size == 0:
            self.logger.info("No new unique points to add.")
            return

        # Append unique new points
        self.point_data.coordinates = np.vstack((self.point_data.coordinates, new_point_data.coordinates[unique_indices]))
        self.point_data.dist_from_atom_center = np.vstack((self.point_data.dist_from_atom_center, new_point_data.dist_from_atom_center[unique_indices]))
        self.point_data.step_in_frac = np.vstack((self.point_data.step_in_frac, new_point_data.step_in_frac[unique_indices]))
        self.point_data.central_point_ids = np.concatenate((self.point_data.central_point_ids, new_point_data.central_point_ids[unique_indices]))
        self.point_data.chunk_ids = np.concatenate((self.point_data.chunk_ids, new_point_data.chunk_ids[unique_indices]))
        self.point_data.grid_amplitude_initialized = np.concatenate((self.point_data.grid_amplitude_initialized, new_point_data.grid_amplitude_initialized[unique_indices]))
        self.logger.debug(f"Added {unique_indices.size} new unique points to self.point_data.")

    def get_point_data(self) -> PointData:
        return self.point_data

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        try:
            # Prepare the data to be saved
            data_to_save = {
                'coordinates': self.point_data.coordinates,
                'dist_from_atom_center': self.point_data.dist_from_atom_center,
                'step_in_frac': self.point_data.step_in_frac,
                'central_point_ids': self.point_data.central_point_ids,
                'chunk_ids': self.point_data.chunk_ids,
                'grid_amplitude_initialized': self.point_data.grid_amplitude_initialized.astype(int)  # Save as integers (0 or 1)
            }

            # Save data to HDF5
            with h5py.File(hdf5_file_path, 'w') as h5file:
                for key, value in data_to_save.items():
                    h5file.create_dataset(key, data=value)
            self.logger.info(f"Point data saved to {hdf5_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save point data to HDF5 file: {e}")

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        try:
            with h5py.File(hdf5_file_path, 'r') as h5file:
                coordinates = h5file['coordinates'][:]
                dist_from_atom_center = h5file['dist_from_atom_center'][:]
                step_in_frac_array = h5file['step_in_frac'][:]
                central_point_ids = h5file['central_point_ids'][:]
                chunk_ids = h5file['chunk_ids'][:]
                grid_amplitude_initialized = h5file['grid_amplitude_initialized'][:].astype(bool)
                self.point_data = PointData(
                    coordinates=coordinates,
                    dist_from_atom_center=dist_from_atom_center,
                    step_in_frac=step_in_frac_array,
                    central_point_ids=central_point_ids,
                    chunk_ids=chunk_ids,
                    grid_amplitude_initialized=grid_amplitude_initialized
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load point data from HDF5 file: {e}")
            return False
