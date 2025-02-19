# processors/point_data_processor.py

import numpy as np
import logging
from processors.rifft_grid_generator import GridGenerator1D, GridGenerator2D, GridGenerator3D
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from data_structures.point_data import PointData
from typing import Optional, List
import h5py

class PointDataProcessor:
    """
    Processes point data by expanding points and saving grid points and amplitudes by chunks.
    """

    def __init__(self, data_saver: RIFFTInDataSaver, save_rifft_coordinates: bool = False, max_chunk_size: int = 100000):
        """
        Initializes the PointDataProcessor with a DataSaver instance and a global flag.

        Args:
            data_saver (RIFFTInDataSaver): Instance responsible for saving data to files.
            save_rifft_coordinates (bool): Flag to determine if grid points should be saved separately.
            max_chunk_size (int): Maximum number of points per chunk.
        """
        self.data_saver = data_saver
        self.save_rifft_coordinates = save_rifft_coordinates
        self.max_chunk_size = max_chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.point_data: Optional[PointData] = None

    def process_point_data(self, point_data: PointData):
        """
        Processes the point data by generating grid points and amplitudes for chunks that haven't been processed.

        Args:
            point_data (PointData): The point data to process.
        """
        self.point_data = point_data

        # Identify unique chunk_ids
        unique_chunk_ids = np.unique(self.point_data.chunk_ids)
        self.logger.info(f"Found {len(unique_chunk_ids)} unique chunk_ids.")

        for chunk_id in unique_chunk_ids:
            # Find points in this chunk that haven't been initialized
            mask = (self.point_data.chunk_ids == chunk_id) & (~self.point_data.grid_amplitude_initialized)
            num_uninitialized = np.sum(mask)

            if num_uninitialized == 0:
                self.logger.debug(f"Chunk {chunk_id} already processed. Skipping.")
                continue

            self.logger.info(f"Processing chunk {chunk_id} with {num_uninitialized} uninitialized points.")

            # Process points in this chunk
            self._process_chunk(chunk_id, mask)

        # After processing all chunks, save the updated grid_amplitude_initialized
        self.save_grid_amplitude_initialized()

    def _process_chunk(self, chunk_id: int, mask: np.ndarray):
        """
        Processes a single chunk of points.

        Args:
            chunk_id (int): The ID of the chunk.
            mask (np.ndarray): Boolean array indicating which points in the chunk are uninitialized.
        """
        indices = np.where(mask)[0]
        coordinates = self.point_data.coordinates[indices]
        dist_from_atom_center = self.point_data.dist_from_atom_center[indices]
        step_in_frac = self.point_data.step_in_frac[indices]
        central_point_ids = self.point_data.central_point_ids[indices]

        num_points = len(indices)
        dimensionality = coordinates.shape[1]
        self.logger.debug(f"Chunk {chunk_id}: Processing {num_points} uninitialized points with dimensionality {dimensionality}.")

        all_grid_data = []
        all_amplitude_data = []

        for i in range(num_points):
            central_point = coordinates[i]
            dist = dist_from_atom_center[i]
            step = step_in_frac[i]
            central_point_id = central_point_ids[i]

            grid_points = self._generate_grid(chunk_id, dimensionality, step, central_point, dist, central_point_id)
            amplitude_data = self._generate_amplitude(chunk_id, central_point_id, grid_points)

            # Collect data for this chunk
            all_grid_data.append(grid_points)
            all_amplitude_data.append(amplitude_data)

        # Merge all grid_points and amplitudes for this chunk
        merged_grid_points = np.vstack(all_grid_data) if self.save_rifft_coordinates else None
        merged_amplitude_data = np.vstack(all_amplitude_data)
        
        
        total_reciprocal_points_filename =  self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_ntotal_reciprocal_space_points')
        self.data_saver.save_data({'ntotal_reciprocal_points': np.zeros([1], dtype = np.int64)}, total_reciprocal_points_filename)    
        
        
        # Save the data for this chunk
        self._save_chunk_data(chunk_id, merged_grid_points, merged_amplitude_data, np.zeros([1], dtype = int))

        # Mark all points in this chunk as initialized
        self.point_data.grid_amplitude_initialized[mask] = True
        self.logger.debug(f"Chunk {chunk_id}: All uninitialized points marked as initialized.")

    def _generate_grid(self, chunk_id: int, dimensionality, step_in_frac, central_point, dist, central_point_id):
        """
        Generates grid points around a central point.

        Args:
            chunk_id (int): The ID of the chunk.
            dimensionality (int): Dimensionality of the data (1, 2, or 3).
            step_in_frac (float or array-like): Step sizes for each dimension.
            central_point (np.ndarray): Coordinates of the central point.
            dist (np.ndarray): Distances from the central point.
            central_point_id (int or str): Original ID of the central point.

        Returns:
            np.ndarray: Array of grid points generated around the central point.
        """
        self.logger.debug(f"Chunk {chunk_id}: Generating grid for central_point_id={central_point_id} with step_in_frac={step_in_frac} and dist={dist}")

        grid_generator = self.grid_generator_factory(dimensionality, step_in_frac)
        grid_points = grid_generator.generate_grid_around_point(np.array(central_point), np.array(dist))
        self.logger.debug(f"Chunk {chunk_id}: Generated {grid_points.shape[0]} grid points for central_point_id={central_point_id}")

        return grid_points

    def _generate_amplitude(self, chunk_id: int, central_point_id, grid_points):
        """
        Generates amplitude data for a set of grid points.

        Args:
            chunk_id (int): The ID of the chunk.
            central_point_id (int or str): Original ID of the central point.
            grid_points (np.ndarray): Array of grid points for which amplitudes are to be generated.

        Returns:
            np.ndarray: Array containing central_point_id and corresponding amplitude values.
        """
        self.logger.debug(f"Chunk {chunk_id}: Generating amplitude data for central_point_id={central_point_id}")

        # Assign zero values to amplitude
        amplitude = np.zeros(grid_points.shape[0], dtype=np.complex128)

        # Prepare amplitude data: central_point_id, amplitude
        amplitude_data = np.hstack((
            np.full((amplitude.shape[0], 1), central_point_id),
            amplitude.reshape(-1, 1)
        ))
        self.logger.debug(f"Chunk {chunk_id}: Generated amplitude data with shape {amplitude_data.shape}")

        return amplitude_data

    def _save_chunk_data(self, chunk_id: int, grid_points: Optional[np.ndarray], amplitude_data: np.ndarray, nreciprocal_space_points: [np.ndarray]):
        """
        Saves the grid points and amplitude data for a chunk.

        Args:
            chunk_id (int): The ID of the chunk.
            grid_points (np.ndarray): Generated grid points, or None.
            amplitude_data (np.ndarray): Generated amplitude data.
        """
        if self.save_rifft_coordinates and grid_points is not None:
            grid_filename = self.data_saver.generate_filename(chunk_id, suffix='_grid')
            self.data_saver.save_data({'grid_points': grid_points}, grid_filename)
            self.logger.info(f"Chunk {chunk_id}: Grid points saved to {grid_filename}")

        amplitude_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes')
        self.data_saver.save_data({'amplitudes': amplitude_data}, amplitude_filename)
        nreciprocal_space_points_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_nreciprocal_space_points')
        self.data_saver.save_data({'nreciprocal_space_points': nreciprocal_space_points}, nreciprocal_space_points_filename)
        self.logger.info(f"Chunk {chunk_id}: Amplitudes saved to {amplitude_filename}")

    def grid_generator_factory(self, dimensionality, step_in_frac):
        """
        Factory method to get the appropriate GridGenerator based on dimensionality.

        Args:
            dimensionality (int): Dimensionality of the data (1, 2, or 3).
            step_in_frac (float or array-like): Step sizes for each dimension.

        Returns:
            GridGenerator*: Instance of the appropriate GridGenerator class.
        """
        if dimensionality == 1:
            return GridGenerator1D(step_in_frac)
        elif dimensionality == 2:
            return GridGenerator2D(step_in_frac)
        elif dimensionality == 3:
            return GridGenerator3D(step_in_frac)
        else:
            self.logger.error(f"Unsupported dimensionality: {dimensionality}")
            raise ValueError(f"Unsupported dimensionality: {dimensionality}")

    def save_grid_amplitude_initialized(self):
        """
        Saves the updated `grid_amplitude_initialized` array to the HDF5 file.
        """
        hdf5_file_path = self.data_saver.hdf5_file_path
        try:
            with h5py.File(hdf5_file_path, 'a') as h5file:
                if 'grid_amplitude_initialized' in h5file:
                    del h5file['grid_amplitude_initialized']
                h5file.create_dataset('grid_amplitude_initialized', data=self.point_data.grid_amplitude_initialized.astype(int))
            self.logger.info(f"Updated grid_amplitude_initialized saved to {hdf5_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save updated grid_amplitude_initialized: {e}")
