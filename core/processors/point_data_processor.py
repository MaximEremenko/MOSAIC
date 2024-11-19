# processors/point_data_processor.py

import numpy as np
import logging
from processors.rifft_grid_generator import GridGenerator1D, GridGenerator2D, GridGenerator3D

class PointDataProcessor:
    """
    Processes point data by expanding points and saving grid points and amplitudes to separate files.
    """

    def __init__(self, data_saver, save_rifft_coordinates=False):
        """
        Initializes the PointDataProcessor with a DataSaver instance and a global flag.

        Args:
            data_saver (RIFFTInDataSaver): Instance responsible for saving data to files.
            save_rifft_coordinates (bool): Flag to determine if grid points should be saved separately.
        """
        self.data_saver = data_saver
        self.save_rifft_coordinates = save_rifft_coordinates
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_point_data_chunk(self, chunk_id, point_data_chunk):
        """
        Processes a point data chunk by generating grid points and amplitudes, then saving them separately.

        Args:
            chunk_id (int): ID of the chunk.
            point_data_chunk (dict): Contains 'coordinates', 'dist_from_atom_center', 'step_in_frac', and 'central_point_ids'.

        Returns:
            dict or None: Dictionary with paths to saved 'grid_points' and 'amplitudes' files,
                          or None if no data was generated.
        """
        self.logger.debug(f"Processing chunk_id: {chunk_id}")

        try:
            # Validate the input data
            self._validate_point_data_chunk(point_data_chunk)

            # Generate grid and amplitude data
            all_grid_data, all_amplitude_data = self._generate_grid_and_amplitude(point_data_chunk)

            # Save the generated data
            saved_files = self._save_data(chunk_id, all_grid_data, all_amplitude_data)

            return saved_files if saved_files else None

        except Exception as e:
            self.logger.exception(f"An error occurred during point data processing: {e}")
            return None

    def _validate_point_data_chunk(self, point_data_chunk):
        """
        Validates the input point data chunk.

        Args:
            point_data_chunk (dict): Contains 'coordinates', 'dist_from_atom_center', 'step_in_frac', and 'central_point_ids'.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If shapes of the data do not align.
        """
        self.logger.debug("Validating point_data_chunk.")

        # Extract data
        coordinates = point_data_chunk.get('coordinates')                  # Expected Shape: (N, 3)
        dist_from_atom_center = point_data_chunk.get('dist_from_atom_center')  # Expected Shape: (N, 3)
        step_in_frac = point_data_chunk.get('step_in_frac')                # Expected Shape: (N, 3) or scalar
        central_point_ids = point_data_chunk.get('central_point_ids')      # Expected Shape: (N,)

        # Check for missing keys
        if coordinates is None or dist_from_atom_center is None or step_in_frac is None or central_point_ids is None:
            self.logger.error("Missing required keys in point_data_chunk.")
            raise KeyError("point_data_chunk must contain 'coordinates', 'dist_from_atom_center', 'step_in_frac', and 'central_point_ids'.")

        # Check shapes
        if coordinates.shape[0] != dist_from_atom_center.shape[0]:
            self.logger.error("Mismatch in number of points between 'coordinates' and 'dist_from_atom_center'.")
            raise ValueError("Number of points in 'coordinates' and 'dist_from_atom_center' must match.")

        if not np.isscalar(step_in_frac) and coordinates.shape[0] != step_in_frac.shape[0]:
            self.logger.error("Mismatch in number of points between 'coordinates' and 'step_in_frac'.")
            raise ValueError("Number of points in 'coordinates' and 'step_in_frac' must match if 'step_in_frac' is an array.")

        if coordinates.shape[0] != central_point_ids.shape[0]:
            self.logger.error("Mismatch in number of points between 'coordinates' and 'central_point_ids'.")
            raise ValueError("Number of points in 'coordinates' and 'central_point_ids' must match.")

        self.logger.debug("Validation successful.")

    def _generate_grid_and_amplitude(self, point_data_chunk):
        """
        Generates grid points and corresponding amplitude data for each point.

        Args:
            point_data_chunk (dict): Contains 'coordinates', 'dist_from_atom_center', 'step_in_frac', and 'central_point_ids'.

        Returns:
            tuple: Two lists containing grid data and amplitude data respectively.
        """
        self.logger.debug("Generating grid and amplitude data.")

        coordinates = point_data_chunk['coordinates']
        dist_from_atom_center = point_data_chunk['dist_from_atom_center']
        step_in_frac = point_data_chunk['step_in_frac']
        central_point_ids = point_data_chunk['central_point_ids']

        num_points, dimensionality = coordinates.shape
        self.logger.debug(f"Number of points: {num_points}, Dimensionality: {dimensionality}")

        all_grid_data = []
        all_amplitude_data = []

        for i in range(num_points):
            self.logger.debug(f"Processing point {i+1}/{num_points}")
            central_point = coordinates[i]
            dist = dist_from_atom_center[i]  # Shape: (3,)

            # Determine step size for the current point
            step = step_in_frac[i] if not np.isscalar(step_in_frac) else step_in_frac

            # Handle step_in_frac appropriately
            grid_points = self._generate_grid(dimensionality, step, central_point, dist, i, central_point_ids[i])

            if grid_points.size == 0:
                self.logger.debug(f"Point {i}: No grid points generated, skipping.")
                continue  # Skip if no grid points generated

            # Generate amplitude data
            amplitude_data = self._generate_amplitude(central_point_ids[i], grid_points)

            if self.save_rifft_coordinates:
                # Prepare grid data: central_point_id, grid_points (x, y, z)
                central_point_id = central_point_ids[i]
                central_point_ids_array = np.full((grid_points.shape[0], 1), central_point_id)
                grid_data = np.hstack((central_point_ids_array, grid_points))
                all_grid_data.append(grid_data)

            # Append amplitude data
            all_amplitude_data.append(amplitude_data)

        self.logger.debug("Grid and amplitude data generation complete.")
        return all_grid_data, all_amplitude_data

    def _generate_grid(self, dimensionality, step_in_frac, central_point, dist, point_index, central_point_id):
        """
        Generates grid points around a central point.

        Args:
            dimensionality (int): Dimensionality of the data (1, 2, or 3).
            step_in_frac (float or array-like): Step sizes for each dimension.
            central_point (np.ndarray): Coordinates of the central point.
            dist (np.ndarray): Distances from the central point.
            point_index (int): Index of the current point being processed.
            central_point_id (int or str): Original ID of the central point.

        Returns:
            np.ndarray: Array of grid points generated around the central point.
        """
        self.logger.debug(f"Point {central_point_id}: Generating grid with step_in_frac={step_in_frac} and dist={dist}")

        grid_generator = self.grid_generator_factory(dimensionality, step_in_frac)
        grid_points = grid_generator.generate_grid_around_point(central_point, dist)
        self.logger.debug(f"Point {central_point_id}: Generated {grid_points.shape[0]} grid points")

        return grid_points

    def _generate_amplitude(self, central_point_id, grid_points):
        """
        Generates amplitude data for a set of grid points.

        Args:
            central_point_id (int or str): Original ID of the central point.
            grid_points (np.ndarray): Array of grid points for which amplitudes are to be generated.

        Returns:
            np.ndarray: Array containing central_point_id and corresponding amplitude values.
        """
        self.logger.debug(f"Point {central_point_id}: Generating amplitude data.")

        # Assign zero values to amplitude
        amplitude = np.zeros(grid_points.shape[0])

        # Prepare amplitude data: central_point_id, amplitude
        amplitude_data = np.hstack((
            np.full((amplitude.shape[0], 1), central_point_id),
            amplitude.reshape(-1, 1)
        ))
        self.logger.debug(f"Point {central_point_id}: Generated amplitude data with shape {amplitude_data.shape}")

        return amplitude_data

    def _save_data(self, chunk_id, all_grid_data, all_amplitude_data):
        """
        Saves the generated grid points and amplitudes to separate files.

        Args:
            chunk_id (int): ID of the chunk.
            all_grid_data (list): List of NumPy arrays containing grid data.
            all_amplitude_data (list): List of NumPy arrays containing amplitude data.

        Returns:
            dict: Dictionary with paths to saved 'grid_points' and 'amplitudes' files.
        """
        self.logger.debug("Saving grid and amplitude data to files.")

        saved_files = {}

        if self.save_rifft_coordinates and all_grid_data:
            # Merge all grid data and save to a separate file
            merged_grid_data = np.vstack(all_grid_data)
            grid_filename = self.data_saver.generate_filename(chunk_id, suffix='_grid')
            grid_file_path = self.data_saver.save_data(merged_grid_data, grid_filename)
            saved_files['grid_points'] = grid_file_path
            self.logger.info(f"Chunk {chunk_id}: Grid points saved to {grid_file_path}")

        if all_amplitude_data:
            # Merge all amplitude data and save to a file
            merged_amplitude_data = np.vstack(all_amplitude_data)
            amplitude_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes')
            amplitude_file_path = self.data_saver.save_data(merged_amplitude_data, amplitude_filename)
            saved_files['amplitudes'] = amplitude_file_path
            self.logger.info(f"Chunk {chunk_id}: Amplitudes saved to {amplitude_file_path}")

        self.logger.debug("Data saving complete.")
        return saved_files

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
