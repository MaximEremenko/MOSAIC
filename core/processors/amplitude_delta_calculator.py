# processors/amplitude_delta_calculator.py

import numpy as np
from dask import delayed, compute
import logging
import os

from utilities.nufft_wrapper import execute_nufft, execute_inverse_nufft
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from managers.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def compute_amplitudes_delta(
    parameters: dict,
    FormFactorFactoryProducer,
    MaskStrategy,  # <-- MaskStrategy is now an instance, not a class
    MaskStrategyParameters: dict,
    db_manager: DatabaseManager,
    output_dir: str,
):
   
    def initialize_rifft_amplitudes(
        db_manager: DatabaseManager,
        rifft_saver: RIFFTInDataSaver,
        chunk_ids: list
    ) -> dict:
        """
        Initializes rifft_amplitudes by loading existing data from files based on chunk_ids.

        Args:
            db_manager (DatabaseManager): Instance to manage database operations.
            rifft_saver (RIFFTInDataSaver): Instance to handle data saving/loading.
            chunk_ids (list): List of chunk_id integers.

        Returns:
            dict: Dictionary mapping chunk_id to rifft_amplitudes (numpy arrays).
        """
        rifft_amplitudes = {}
        for chunk_id in chunk_ids:
            filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
            file_path = os.path.join(rifft_saver.output_dir, filename)
            if os.path.exists(file_path):
                try:
                    data = rifft_saver.load_data(filename)
                    rifft_amplitudes[chunk_id] = data.get('amplitudes', np.array([]))
                    logger.debug(f"Loaded rifft_amplitudes for chunk_id: {chunk_id} from file: {filename}")
                except Exception as e:
                    logger.error(f"Error loading rifft_amplitudes for chunk_id: {chunk_id} from file: {filename}: {e}")
                    rifft_amplitudes[chunk_id] = np.array([])  # Initialize empty if load fails
            else:
                rifft_amplitudes[chunk_id] = np.array([])  # Initialize empty if file does not exist
                logger.debug(f"No existing rifft_amplitudes for chunk_id: {chunk_id}. Initialized empty array.")
        return rifft_amplitudes

    """
    Computes delta amplitudes using NUFFT with nested parallelism.

    Args:
        parameters (dict): Dictionary containing required inputs such as hkl_intervals, vectors, supercell, etc.
        FormFactorFactoryProducer: Factory to create form factor calculators.
        MaskStrategy (MaskStrategy): Instance implementing the MaskStrategy interface.
        MaskStrategyParameters (dict): Parameters for the masking strategy.
        db_manager (DatabaseManager): Instance of DatabaseManager for database interactions.
        output_dir (str): Directory where rifft data will be saved.
    """

    # Extract inputs from parameters
    hkl_intervals = parameters["hkl_intervals"]
    point_data_list = parameters["point_data_list"]
    original_coords = parameters["original_coords"]  # Shape: (N, D)
    average_coords = parameters["average_coords"]    # Shape: (N, D)
    elements = parameters["elements"]                # Shape: (N,)
    vectors = parameters["vectors"]                  # Reciprocal space matrix B_ (D, D)
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0)
    def vec2spacemat(vectors):
        av = vectors[:, 0]
        bv = vectors[:, 1]
        cv = vectors[:, 2]

        # basis vectors of reciprocal space
        a_ = np.cross(bv, cv)/(np.dot(av, np.cross(bv, cv)))
        b_ = np.cross(cv, av)/(np.dot(av, np.cross(bv, cv)))
        c_ = np.cross(av, bv)/(np.dot(av, np.cross(bv, cv)))

        B_ = np.array([a_, b_, c_])
        B = np.linalg.inv(B_)
        G_ = np.matmul(B_.T, B_)
        G = np.matmul(B.T, B)
        return B, B_, G, G_
    B, B_, G, G_ = vec2spacemat(vectors/supercell)
    # Initialize complex coefficients
    M = original_coords.shape[0]
    c = np.ones(M, dtype=np.complex128)

    # Initialize RIFFTInDataSaver
    rifft_saver = RIFFTInDataSaver(output_dir=output_dir, file_extension='hdf5')

    # Removed redundant and undefined masking lines
    # Removed:
    # mask = MaskStrategy.generate_mask(data_points, MaskStrategyParameters)
    # masked_data = MaskStrategy.apply_mask(mask, data_array, MaskStrategyParameters)

    # Retrieve pending parts and chunk IDs from the database
    pending_parts = db_manager.get_pending_parts()  # List of hkl_interval dictionaries
    chunk_ids = db_manager.get_pending_chunk_ids()  # List of chunk_id integers

    # Identify unique elements
    unique_elements = np.unique(elements)

    # Initialize rifft_amplitudes dictionary to hold partial results
    rifft_amplitudes = initialize_rifft_amplitudes(db_manager, rifft_saver, chunk_ids)

    #@delayed
    def process_ihkl_element(
        ihkl: dict,
        element: str,
        B_: np.ndarray,
        mask_strategy: any,
        mask_parameters: dict
    ):
        """
        Processes a single (ihkl, element) pair.

        Args:
            ihkl (dict): HKL interval information.
            element (str): Chemical element symbol.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_strategy (MaskStrategy): Instance of mask strategy.
            mask_parameters (dict): Parameters for the mask strategy.

        Returns:
            tuple: (ihkl_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
        """
        logger.debug(f"Processing ihkl: {ihkl}, element: {element}")

        # Generate q-space grid
        q_space_grid = generate_q_space_grid_sync(ihkl, B_, mask_parameters, mask_strategy, supercell)
        if q_space_grid.size == 0:
            logger.debug(f"No q-space points after masking for ihkl: {ihkl}, element: {element}")
            return None  # Skip if no points

        # Calculate form factors
        #ff_calculator = FormFactorFactoryProducer.get_factory('default').create_calculator(method='default')
        ff = FormFactorFactoryProducer.calculate(q_space_grid, element, charge=charge)

        # Extract mask for elements
        mask_elements = (elements == element)
        if not np.any(mask_elements):
            logger.warning(f"No points found for element: {element}")
            return None

        # Perform NUFFT calculations
        q_amplitudes = ff * execute_nufft(original_coords[mask_elements], c[mask_elements], q_space_grid, eps=1e-5)
        q_amplitudes_av = execute_nufft(average_coords[mask_elements], c[mask_elements], q_space_grid, eps=1e-5)
        q_amplitudes_delta = execute_nufft(original_coords[mask_elements] - average_coords[mask_elements], c[mask_elements], q_space_grid, eps=1e-5)

        # Final computation
        q_amplitudes_av_final = ff * q_amplitudes_av * q_amplitudes_delta / c.size

        logger.debug(f"Completed NUFFT computations for ihkl: {ihkl}, element: {element}")

        return (ihkl['id'], element, q_space_grid, q_amplitudes, q_amplitudes_av_final)

    #@delayed
    def process_chunk_id(
        chunk_id: int,
        ihkl_id: int,
        q_space_grid: np.ndarray,
        q_amplitudes: np.ndarray,
        q_amplitudes_av: np.ndarray,
        rifft_saver: RIFFTInDataSaver,
        point_data_list: list  # Now includes 'id's
    ):
        """
        Processes a single chunk_id associated with a (ihkl, element) pair.
    
        Args:
            chunk_id (int): Chunk identifier.
            ihkl_id (int): HKL interval identifier.
            q_space_grid (np.ndarray): Array of q-space coordinates.
            q_amplitudes (np.ndarray): Array of q_amplitudes.
            q_amplitudes_av (np.ndarray): Array of q_amplitudes_av.
            rifft_saver (RIFFTInDataSaver): Instance to handle data saving/loading.
            point_data_list (list): List of point_data dictionaries with 'id's.
    
        Returns:
            None
        """
        logger.debug(f"Processing chunk_id: {chunk_id} for ihkl_id: {ihkl_id}")
    
        # Retrieve all point_data associated with this chunk_id
        chunk_data = [pd for pd in point_data_list if pd["chunk_id"] == chunk_id]
        if not chunk_data:
            logger.warning(f"No point data found for chunk_id: {chunk_id}")
            return
    
        # Generate RIFFT grid
        rifft_space_grid = generate_rifft_grid_sync(chunk_data, supercell)
        if rifft_space_grid.size == 0:
            logger.warning(f"No rifft_space_grid points generated for chunk_id: {chunk_id}")
            return
    
        # Perform inverse NUFFT
        r_amplitudes_partial = execute_inverse_nufft(
            q_coords=q_space_grid,
            c=q_amplitudes - q_amplitudes_av,
            real_coords=rifft_space_grid,
            eps=1e-15
        )
    
        # Generate filename based on chunk_id
        filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
    
        # Load existing rifft amplitudes if file exists
        try:
            existing_data = rifft_saver.load_data(filename)
            rifft_amplitudes_chunk = existing_data.get('_amplitudes', np.array([]))
            rifft_amplitudes_chunk = np.concatenate((rifft_amplitudes_chunk, r_amplitudes_partial))
        except FileNotFoundError:
            rifft_amplitudes_chunk = r_amplitudes_partial  # Initialize if file does not exist
    
        # Save updated rifft amplitudes
        rifft_saver.save_data({'rift_amplitudes': rifft_amplitudes_chunk}, filename, append=False)
        logger.debug(f"Saved rifft amplitudes for chunk_id: {chunk_id} to file: {filename}")
    
        # Update database to mark this (point, hkl) association as saved
        # Assuming all points in chunk_data are associated with ihkl_id
        updates = [(1, pd['id'], ihkl_id) for pd in chunk_data]
        db_manager.update_saved_status_batch(updates)
        logger.debug(f"Updated saved status for chunk_id: {chunk_id} and ihkl_id: {ihkl_id}")

    #@delayed
    def generate_q_space_grid(
        ihkl: dict,
        B_: np.ndarray,
        mask_parameters: dict,
        mask_strategy: any,
        supercell: np.ndarray
    ) -> np.ndarray:
        """
        Generates the q-space grid for a given ihkl interval, applying masking.

        Args:
            ihkl (dict): HKL interval information.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_parameters (dict): Parameters for mask generation.
            mask_strategy (MaskStrategy): Instance of mask strategy.

        Returns:
            np.ndarray: Masked q-space grid coordinates.
        """
        # Generate h, k, l ranges
        h_start, h_end = ihkl['h_start'] , ihkl['h_end']
        k_start, k_end = ihkl['k_start'] , ihkl['k_end']
        l_start, l_end = ihkl['l_start'] , ihkl['l_end']
        step = 1/supercell

        # Create ranges
        h_vals = np.arange(h_start, h_end + step[0], step[0]) if h_end > h_start else np.array([h_start])
        k_vals = np.arange(k_start, k_end + step[1], step[1]) if k_end > k_start else np.array([k_start])
        l_vals = np.arange(l_start, l_end + step[2], step[2]) if l_end > l_start else np.array([l_start])

        # Create meshgrid
        mesh = np.meshgrid(h_vals, k_vals, l_vals, indexing='ij')
        q_points = np.stack([m.flatten() for m in mesh], axis=1)  # Shape: (M, 3)

        # Convert to q-space using vectors
       
        spetial_points = np.array((0,0,0))
        # Apply mask using MaskStrategy instance
        mask = mask_strategy.apply(q_points, spetial_points)
        q_points_masked = q_points[mask]
        q_space_masked = 2 * np.pi * np.dot(q_points_masked, B_)  # Shape: (M, D)
        
        return q_space_masked

    #@delayed
    def generate_rifft_grid(
        chunk_data: list,
        supercell: np.ndarray
    ) -> np.ndarray:
        """
        Generates the RIFFT grid based on chunk data and supercell information.

        Args:
            chunk_data (list): List of point_data dictionaries associated with the chunk_id.
            supercell (dict): Supercell information (e.g., lattice parameters).

        Returns:
            np.ndarray: RIFFT grid coordinates (M, D).
        """
        # Example implementation: Adjust based on actual requirements
        # Assuming supercell contains necessary info to generate r_space_grid
        # Here, we simply extract coordinates for demonstration
        coordinates = np.array([pd['coordinates'] for pd in chunk_data])
        dist_from_atom_center = np.array([pd['dist_from_atom_center'] for pd in chunk_data])
        step_in_frac = np.array([pd['step_in_frac'] for pd in chunk_data])

        # Combine to generate r_space_grid; customize as needed
        r_space_grid = coordinates - dist_from_atom_center  # Placeholder operation
        return r_space_grid

    def initialize_rifft_amplitudes(
        db_manager: DatabaseManager,
        rifft_saver: RIFFTInDataSaver,
        chunk_ids: list
    ) -> dict:
        """
        Synchronous wrapper to initialize rifft_amplitudes.

        Args:
            db_manager (DatabaseManager): Instance to manage database operations.
            rifft_saver (RIFFTInDataSaver): Instance to handle data saving/loading.
            chunk_ids (list): List of chunk_id integers.

        Returns:
            dict: Dictionary mapping chunk_id to rifft_amplitudes (numpy arrays).
        """
        return compute(initialize_rifft_amplitudes(db_manager, rifft_saver, chunk_ids))[0]

    def generate_q_space_grid_sync(
        ihkl: dict,
        B_: np.ndarray,
        mask_parameters: dict,
        mask_strategy: any,
        supercell:np.ndarray
    ) -> np.ndarray:
        """
        Synchronous wrapper to generate q-space grid.

        Args:
            ihkl (dict): HKL interval information.
            vectors (np.ndarray): Reciprocal space matrix B_.
            mask_parameters (dict): Parameters for mask generation.
            mask_strategy (MaskStrategy): Instance of mask strategy.

        Returns:
            np.ndarray: Masked q-space grid coordinates.
        """
        return compute(generate_q_space_grid(ihkl, B_, mask_parameters, mask_strategy, supercell))[0]

    def generate_rifft_grid_sync(
        chunk_data: list,
        supercell: np.ndarray
    ) -> np.ndarray:
        """
        Synchronous wrapper to generate rifft grid.

        Args:
            chunk_data (list): List of point_data dictionaries associated with the chunk_id.
            supercell (np.ndarray): Supercell information (e.g., lattice parameters).

        Returns:
            np.ndarray: RIFFT grid coordinates (M, D).
        """
        return compute(generate_rifft_grid(chunk_data, supercell))[0]

    # Collect all delayed tasks and execute
    ihkl_element_tasks = []
    for ihkl in hkl_intervals:
        for element in unique_elements:
            task = process_ihkl_element(
                ihkl=ihkl,
                element=element,
                B_=B_,
                mask_strategy=MaskStrategy,  # <-- Pass the MaskStrategy instance
                mask_parameters=MaskStrategyParameters
            )
            ihkl_element_tasks.append(task)

        # Create chunk_id tasks
        chunk_id_tasks = []
        for task in ihkl_element_tasks:
            if task is not None:
                ihkl_result = compute(task)  # <-- Correct usage: compute returns a tuple
                if ihkl_result is None:
                    continue  # Skip if ihkl_result is None
                ihkl_id, element, q_space_grid, q_amplitudes, q_amplitudes_av = ihkl_result[0]
                for chunk_id in chunk_ids:
                    chunk_task = process_chunk_id(
                        chunk_id=chunk_id,
                        ihkl_id=ihkl_id,
                        q_space_grid=q_space_grid,
                        q_amplitudes=q_amplitudes,
                        q_amplitudes_av=q_amplitudes_av,
                        rifft_saver=rifft_saver,
                        point_data_list=point_data_list
                    )
                    chunk_id_tasks.append(chunk_task)
    
        # Execute all chunk_id tasks collectively for better parallelism
        if chunk_id_tasks:
            compute(*chunk_id_tasks)

    logger.info("Completed compute_amplitudes_delta")
