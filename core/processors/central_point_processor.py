import logging
import os
import numpy as np
import h5py
from interfaces.point_parameters_processor_interface import IPointParametersProcessor
from data_structures.point_data import PointData
from functions.angstrom_to_fractional import angstrom_to_fractional

logger = logging.getLogger(__name__)

class CentralPointProcessor(IPointParametersProcessor):
    def __init__(self, parameters: dict, average_structure: dict, num_chunks: int = 10):
        """
        Initializes the CentralPointProcessor.

        Args:
            parameters (dict): Input parameters including rspace_info and HDF5 file paths.
            average_structure (dict): Information about average structure (coordinates, vectors, etc.).
            num_chunks (int): Number of chunks for dividing point data.
        """
        self.parameters = parameters
        self.average_structure = average_structure
        self.num_chunks = num_chunks
        self.point_data: PointData = None
        self.vectors = average_structure.get('vectors')
        self.hdf5_file_path = self.parameters.get('hdf5_file_path', 'point_data.hdf5')

    def process_parameters(self):
        """
        Processes parameters to generate PointData, either loading from HDF5 or generating from scratch.
        """
        # Load existing PointData if available
        if os.path.exists(self.hdf5_file_path):
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                logger.info(f"Loaded point data from {self.hdf5_file_path}")
                return

        # Generate PointData from parameters
        rspace_info = self.parameters.get('rspace_info', {})
        points_params = rspace_info.get('points', [])
        all_coordinates, all_distances, all_steps, all_ids = [], [], [], []

        for point_info in points_params:
            file_name = point_info.get('filename')
            distances = point_info.get('distFromAtomCenter')
            steps = point_info.get('stepInAngstrom')

            if not file_name or distances is None or steps is None:
                logger.error("Missing required point parameters.")
                continue

            if not os.path.exists(file_name):
                logger.error(f"Central points file not found: {file_name}")
                continue

            central_points = self._read_central_points(file_name)
            if central_points.size == 0:
                logger.warning(f"No central points loaded from {file_name}")
                continue

            fractional_coords = angstrom_to_fractional(central_points, self.vectors)
            num_points = fractional_coords.shape[0]

            all_coordinates.append(fractional_coords)
            all_distances.append(np.tile(distances, (num_points, 1)))
            all_steps.append(np.tile(steps, (num_points, 1)))
            all_ids.extend(range(len(all_coordinates[-1])))

        if all_coordinates:
            # Aggregate data into a single PointData instance
            self.point_data = PointData(
                coordinates=np.vstack(all_coordinates),
                dist_from_atom_center=np.vstack(all_distances),
                step_in_frac=np.vstack(all_steps),
                central_point_ids=np.array(all_ids),
                chunk_ids=np.zeros(len(all_ids), dtype=int),
                grid_amplitude_initialized=np.zeros(len(all_ids), dtype=bool),
            )
            # Assign chunk IDs
            self._assign_chunk_ids()
            # Save PointData to HDF5
            self.save_point_data_to_hdf5(self.hdf5_file_path)
        else:
            logger.error("No point data generated.")

    def _read_central_points(self, file_name: str) -> np.ndarray:
        """
        Reads central points from a file.

        Args:
            file_name (str): Path to the file containing central points.

        Returns:
            np.ndarray: Central points array.
        """
        try:
            points = np.loadtxt(file_name)
            return points if points.ndim > 1 else np.expand_dims(points, axis=0)
        except Exception as e:
            logger.error(f"Failed to read central points from file {file_name}: {e}")
            return np.array([])

    def _assign_chunk_ids(self):
        """
        Divides PointData into chunks based on num_chunks.
        """
        num_points = self.point_data.coordinates.shape[0]
        points_per_chunk = max(1, num_points // self.num_chunks)
        chunk_ids = np.arange(num_points) // points_per_chunk
        self.point_data.chunk_ids = chunk_ids

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        """
        Saves PointData to an HDF5 file.

        Args:
            hdf5_file_path (str): Path to save the HDF5 file.
        """
        try:
            with h5py.File(hdf5_file_path, 'w') as h5file:
                h5file.create_dataset('coordinates', data=self.point_data.coordinates)
                h5file.create_dataset('dist_from_atom_center', data=self.point_data.dist_from_atom_center)
                h5file.create_dataset('step_in_frac', data=self.point_data.step_in_frac)
                h5file.create_dataset('central_point_ids', data=self.point_data.central_point_ids)
                h5file.create_dataset('chunk_ids', data=self.point_data.chunk_ids)
                h5file.create_dataset('grid_amplitude_initialized', data=self.point_data.grid_amplitude_initialized.astype(int))
            logger.info(f"Point data saved to {hdf5_file_path}")
        except Exception as e:
            logger.error(f"Failed to save point data to HDF5 file: {e}")

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        """
        Loads PointData from an HDF5 file.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with h5py.File(hdf5_file_path, 'r') as h5file:
                self.point_data = PointData(
                    coordinates=h5file['coordinates'][:],
                    dist_from_atom_center=h5file['dist_from_atom_center'][:],
                    step_in_frac=h5file['step_in_frac'][:],
                    central_point_ids=h5file['central_point_ids'][:],
                    chunk_ids=h5file['chunk_ids'][:],
                    grid_amplitude_initialized=h5file['grid_amplitude_initialized'][:].astype(bool),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to load point data from HDF5 file: {e}")
            return False

    def get_point_data(self) -> PointData:
        """
        Returns the processed PointData.

        Returns:
            PointData: The processed point data.
        """
        return self.point_data
