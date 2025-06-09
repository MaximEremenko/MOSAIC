# processors/amplitude_delta_calculator.py

import numpy as np
from dask import delayed, compute
import logging
import os
import time
from utilities.nufft_wrapper import execute_nufft, execute_inverse_nufft
from utilities.cunufft_wrapper import execute_cunufft, execute_inverse_cunufft
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from managers.database_manager import DatabaseManager

from processors.rifft_grid_generator import GridGenerator1D, GridGenerator2D, GridGenerator3D
from processors.point_data_processor import PointDataProcessor

logger = logging.getLogger(__name__)


def compute_amplitudes_delta(
    parameters: dict,
    FormFactorFactoryProducer,
    MaskStrategy,  # <-- MaskStrategy is now an instance, not a class
    MaskStrategyParameters: dict,
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor: PointDataProcessor
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
            dict: Dictionary mapping chunk_id to amplitudes (numpy arrays).
        """
        rifft_amplitudes = {}
        for chunk_id in chunk_ids:
            filename = rifft_saver.generate_filename(chunk_id, suffix='_amplitudes')
            file_path = os.path.join(rifft_saver.output_dir, filename)
            if os.path.exists(file_path):
                try:
                    data = rifft_saver.load_data(filename)
                    rifft_amplitudes[chunk_id] = data.get('amplitudes', np.array([]))
                    logger.debug(f"Loaded amplitudes for chunk_id: {chunk_id} from file: {filename}")
                except Exception as e:
                    logger.error(f"Error loading rifft_amplitudes for chunk_id: {chunk_id} from file: {filename}: {e}")
                    rifft_amplitudes[chunk_id] = np.array([])  # Initialize empty if load fails
            else:
                rifft_amplitudes[chunk_id] = np.array([])  # Initialize empty if file does not exist
                logger.debug(f"No existing _amplitudes for chunk_id: {chunk_id}. Initialized empty array.")
        return rifft_amplitudes

    """
    Computes delta amplitudes using NUFFT with nested parallelism.

    Args:
        parameters (dict): Dictionary containing required inputs such as reciprocal_space_intervals, vectors, supercell, etc.
        FormFactorFactoryProducer: Factory to create form factor calculators.
        MaskStrategy (MaskStrategy): Instance implementing the MaskStrategy interface.
        MaskStrategyParameters (dict): Parameters for the masking strategy.
        db_manager (DatabaseManager): Instance of DatabaseManager for database interactions.
        output_dir (str): Directory where rifft data will be saved.
    """

    # Extract inputs from parameters
    reciprocal_space_intervals_all = parameters["reciprocal_space_intervals_all"]
    reciprocal_space_intervals = parameters["reciprocal_space_intervals"]
    point_data_list = parameters["point_data_list"]
    original_coords = parameters["original_coords"]  # Shape: (N, D)
    cells_origin = parameters["cells_origin"]    # Shape: (N, D)
    elements = parameters["elements"]                # Shape: (N,)
    vectors = parameters["vectors"]                  # Reciprocal space matrix B_ (D, D)
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0)
    B_ = np.linalg.inv(vectors/supercell)
    # Initialize complex coefficients
    M = original_coords.shape[0]
    c = np.ones((M)) + 1J * np.zeros((M))

    # Initialize RIFFTInDataSaver
    rifft_saver = RIFFTInDataSaver(output_dir=output_dir, file_extension='hdf5')

    # Retrieve pending parts and chunk IDs from the database
    chunk_ids = db_manager.get_pending_chunk_ids()  # List of chunk_id integers
    # Identify unique elements
    unique_elements = np.unique(elements)

    #@delayed
    def process_ireciprocal_space_element(
        ireciprocal_space: dict,
        q_space_grid : np.ndarray,
        element: str,
        B_: np.ndarray,
        mask_strategy: any,
        mask_parameters: dict
    ):
        """
        Processes a single (ireciprocal_space, element) pair.

        Args:
            ireciprocal_space (dict): HKL interval information.
            element (str): Chemical element symbol.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_strategy (MaskStrategy): Instance of mask strategy.
            mask_parameters (dict): Parameters for the mask strategy.

        Returns:
            tuple: (ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
        """
        logger.debug(f"Processing ireciprocal_space: {ireciprocal_space}, element: {element}")

        # Generate q-space grid
        if q_space_grid.size == 0:
            logger.warning(f"No q-space points after masking for ireciprocal_space: {ireciprocal_space}, element: {element}")
            return None  # Skip if no points
        # Calculate form factors
        ff = FormFactorFactoryProducer.calculate(q_space_grid, element, charge=charge)

        # Extract mask for elements
        mask_elements = (elements == element)
        if not np.any(mask_elements):
            logger.warning(f"No points found for element: {element}")
            return None
        # Perform NUFFT calculations
        q_amplitudes = ff * execute_nufft(original_coords[mask_elements], c[mask_elements]*0.0+1.0, q_space_grid, eps=1e-12)
        q_amplitudes_av = execute_nufft(cells_origin, c*0.0+1.0, q_space_grid, eps=1e-12)
        q_amplitudes_delta = execute_nufft(original_coords[mask_elements] - cells_origin[mask_elements], c[mask_elements]*0.0+1.0, q_space_grid, eps=1e-12)
        # Final computation
        q_amplitudes_av_final = ff * q_amplitudes_av * q_amplitudes_delta / c.size
 
        logger.info(f"Completed NUFFT computations for ireciprocal_space: {ireciprocal_space}, element: {element}")
        print((q_amplitudes - q_amplitudes_av_final)[39])
        return (ireciprocal_space['id'], element, q_space_grid, q_amplitudes, q_amplitudes_av_final)

    def process_ireciprocal_space_coeff(
        ireciprocal_space: dict,
        q_space_grid: np.ndarray,
        coeff: np.ndarray,
        B_: np.ndarray,
        mask_strategy: any,
        mask_parameters: dict
    ):
        """
        Processes a single (ireciprocal_space, element) pair.
    
        Args:
            ireciprocal_space (dict): HKL interval information.
            coeff (np.ndarray): Chemical element coefficient.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_strategy (MaskStrategy): Instance of mask strategy.
            mask_parameters (dict): Parameters for the mask strategy.
    
        Returns:
            tuple: (ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
        """
        logger.debug(f"Processing ireciprocal_space: {ireciprocal_space}")
    
        # Generate q-space grid
        #q_space_grid = generate_q_space_grid_sync(ireciprocal_space, B_, mask_parameters, mask_strategy, supercell)
        if q_space_grid.size == 0:
            logger.warning(f"No q-space points after masking for ireciprocal_space: {ireciprocal_space}")
            return None  # Skip if no points
        M = c.size
        c_ = coeff*(np.ones((M)) + 1J * np.zeros((M)))
        # Extract mask for elements
    
        # Perform NUFFT calculations
        q_amplitudes = execute_nufft(original_coords, c_, q_space_grid, eps=1e-12)
        q_amplitudes_av = execute_nufft(cells_origin, c_*0.0+1.0, q_space_grid, eps=1e-12)
        q_amplitudes_delta = execute_nufft((original_coords - cells_origin), c_, q_space_grid, eps=1e-12)
    
        # Final computation
        q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / c.size
    
        logger.info(f"Completed NUFFT computations for ireciprocal_space: {ireciprocal_space}")
        print((q_amplitudes - q_amplitudes_av_final)[39])
        return (ireciprocal_space['id'], q_space_grid, q_amplitudes, q_amplitudes_av_final)



    #@delayed
    def process_chunk_id(
        chunk_id: int,
        ireciprocal_space_id: int,
        q_space_grid: np.ndarray,
        q_amplitudes: np.ndarray,
        q_amplitudes_av: np.ndarray,
        rifft_saver: RIFFTInDataSaver,
        point_data_list: list,  # Now includes 'id's
        rifft_amplitudes_chunk_n: np.ndarray,
        total_reciprocal_points : np.ndarray,
        #nmasked_reciprocal_space_points : np.ndarray
    ):
        """
        Processes a single chunk_id associated with a (ireciprocal_space, element) pair.
    
        Args:
            chunk_id (int): Chunk identifier.
            ireciprocal_space_id (int): HKL interval identifier.
            q_space_grid (np.ndarray): Array of q-space coordinates.
            q_amplitudes (np.ndarray): Array of q_amplitudes.
            q_amplitudes_av (np.ndarray): Array of q_amplitudes_av.
            rifft_saver (RIFFTInDataSaver): Instance to handle data saving/loading.
            point_data_list (list): List of point_data dictionaries with 'id's.
    
        Returns:
            None
        """
        logger.info(f"Processing chunk_id: {chunk_id} for ireciprocal_space_id: {ireciprocal_space_id}")
    
        # Retrieve all point_data associated with this chunk_id
        chunk_data = [pd for pd in point_data_list if pd["chunk_id"] == chunk_id]
        if not chunk_data:
            logger.warning(f"No point data found for chunk_id: {chunk_id}")
            return
    
        # Generate RIFFT grid
        rifft_space_grid, grid_shapeNd = generate_rifft_grid_sync(chunk_data, supercell)
        if rifft_space_grid.size == 0:
            logger.warning(f"No rifft_space_grid points generated for chunk_id: {chunk_id}")
            return
    
        # Perform inverse NUFFT
        r_amplitudes_partial = execute_inverse_nufft(
            q_coords=q_space_grid,
            c=q_amplitudes - q_amplitudes_av,
            real_coords=rifft_space_grid,
            eps=1e-12
        )
        masked_reciprocal_space_points = q_space_grid.shape[0]
        # Generate filename based on chunk_id
        filename = point_data_processor.data_saver.generate_filename(chunk_id, suffix='_amplitudes')
        filename_shapeNd = point_data_processor.data_saver.generate_filename(chunk_id, suffix='_shapeNd')
        try:
            existing_data = point_data_processor.data_saver.load_data(filename_shapeNd)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data({'shapeNd': grid_shapeNd}, filename_shapeNd)
            
            
        print(f"total_reciprocal_points = {total_reciprocal_points}")
        total_reciprocal_points_filename =  point_data_processor.data_saver.generate_filename(chunk_id, suffix='_amplitudes_ntotal_reciprocal_space_points')
        existing_data_tpp = point_data_processor.data_saver.load_data(total_reciprocal_points_filename) 
        ntotal_reciprocal_points = existing_data_tpp.get('ntotal_reciprocal_points', np.zeros([1], dtype = int))
        if (ntotal_reciprocal_points == 0):
            point_data_processor.data_saver.save_data({'ntotal_reciprocal_points': total_reciprocal_points}, total_reciprocal_points_filename)        
            print(f"ntotal_reciprocal_points = {ntotal_reciprocal_points}")
        
        
        
        nreciprocal_space_points_filename = point_data_processor.data_saver.generate_filename(chunk_id, suffix='_amplitudes_nreciprocal_space_points')
        
        # Load existing rifft amplitudes if file exists
        try:
            existing_data = point_data_processor.data_saver.load_data(filename)
            rifft_amplitudes_chunk = existing_data.get('amplitudes', np.array([]))
            existing_data_npp = point_data_processor.data_saver.load_data(nreciprocal_space_points_filename) 
            nreciprocal_space_points = existing_data_npp.get('nreciprocal_space_points', np.zeros([1], dtype = int))
            if(rifft_amplitudes_chunk_n.shape[0]<1):
                rifft_amplitudes_chunk_n = rifft_amplitudes_chunk*0.0
            if (q_space_grid.shape[1] > 2):
                il = np.round(np.array(q_space_grid[:,2], dtype = np.float64, order="C" ), 8)
                if np.all(il!=0):
                    rifft_amplitudes_chunk[:,1] = rifft_amplitudes_chunk[:,1] +  r_amplitudes_partial + np.conj(r_amplitudes_partial)
                    nreciprocal_space_points += masked_reciprocal_space_points + masked_reciprocal_space_points
                else:
                    rifft_amplitudes_chunk[:,1] = rifft_amplitudes_chunk[:,1] +  r_amplitudes_partial
                    nreciprocal_space_points += masked_reciprocal_space_points
            else:
                rifft_amplitudes_chunk[:,1] = rifft_amplitudes_chunk[:, 1] +  r_amplitudes_partial 
                nreciprocal_space_points += masked_reciprocal_space_points
        except FileNotFoundError:
            logger.info(f"No amplitudes file for chunk_id: {chunk_id} to file: {filename}")
            rifft_amplitudes_chunk[:,1] = r_amplitudes_partial  # Initialize if file does not exist
        print(rifft_amplitudes_chunk[0:15, 1])
        print(nreciprocal_space_points)
        # Save updated rifft amplitudes
        point_data_processor._save_chunk_data(chunk_id, None, rifft_amplitudes_chunk, nreciprocal_space_points)
        logger.info(f"Saved rifft amplitudes for chunk_id: {chunk_id} to file: {filename}")
    
        # Update database to mark this (point, reciprocal_space) association as saved
        # Assuming all points in chunk_data are associated with ireciprocal_space_id
        db_manager.update_saved_status_for_chunk_or_point(ireciprocal_space_id, None, chunk_id, 1)
        logger.info(f"Updated saved status for chunk_id: {chunk_id} and ireciprocal_space_id: {ireciprocal_space_id}")
        return(rifft_amplitudes_chunk)
    #@delayed
    def reciprocal_space_points_counter(
        ireciprocal_space: dict,
        supercell: np.ndarray
    ) -> np.ndarray:
        """
        Generates the q-space grid for a given ireciprocal_space interval, applying masking.

        Args:
            ireciprocal_space (dict): HKL interval information.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_parameters (dict): Parameters for mask generation.
            mask_strategy (MaskStrategy): Instance of mask strategy.

        Returns:
            np.ndarray: Masked q-space grid coordinates.
        """
        # Generate h, k, l ranges
        
        supercell = np.array(supercell)
        dim = supercell.shape[0]
        step = 1/supercell        
        
        h_start, h_end = ireciprocal_space['h_start'] , ireciprocal_space['h_end']
        if dim > 1:
            k_start, k_end = ireciprocal_space['k_start'] , ireciprocal_space['k_end']
        if dim > 2:
            l_start, l_end = ireciprocal_space['l_start'] , ireciprocal_space['l_end']


        # Create ranges
        
        h_vals = np.arange(h_start, h_end + step[0], step[0]) if h_end > h_start else np.array([h_start])
        if dim > 1:
            k_vals = np.arange(k_start, k_end + step[1], step[1]) if k_end > k_start else np.array([k_start])
        if dim > 2:
            l_vals = np.arange(l_start, l_end + step[2], step[2]) if l_end > l_start else np.array([l_start])
        
        if(supercell.shape[0]>2):
          if (ireciprocal_space['l_start'] == 0 and  ireciprocal_space['l_end'] == 0):
                
                total_points =  h_vals.shape[0]*k_vals.shape[0]*l_vals.shape[0]
          else:
                total_points =  2*h_vals.shape[0]*k_vals.shape[0]*l_vals.shape[0]
        elif(supercell.shape[0]>1):
            total_points =  h_vals.shape[0]*k_vals.shape[0]
        else:
            total_points =  h_vals.shape[0]
        
        return total_points
    
    def generate_q_space_grid(
        ireciprocal_space: dict,
        B_: np.ndarray,
        mask_parameters: dict,
        mask_strategy: any,
        supercell: np.ndarray
    ) -> np.ndarray:
        """
        Generates the q-space grid for a given ireciprocal_space interval, applying masking.

        Args:
            ireciprocal_space (dict): HKL interval information.
            B_ (np.ndarray): Reciprocal space matrix B_.
            mask_parameters (dict): Parameters for mask generation.
            mask_strategy (MaskStrategy): Instance of mask strategy.

        Returns:
            np.ndarray: Masked q-space grid coordinates.
        """
        # Generate h, k, l ranges
        h_start, h_end = ireciprocal_space['h_start'] , ireciprocal_space['h_end']
        k_start, k_end = ireciprocal_space['k_start'] , ireciprocal_space['k_end']
        l_start, l_end = ireciprocal_space['l_start'] , ireciprocal_space['l_end']
        supercell = np.array(supercell)
        step = 1/supercell

        # Create ranges
        h_vals = np.arange(h_start, h_end + step[0], step[0]) if h_end > h_start else np.array([h_start])
        k_vals = np.arange(k_start, k_end + step[1], step[1]) if k_end > k_start else np.array([k_start])
        l_vals = np.arange(l_start, l_end + step[2], step[2]) if l_end > l_start else np.array([l_start])

        # Create meshgrid
        mesh = np.meshgrid(h_vals, k_vals, l_vals, indexing='ij')
        q_points = np.stack([m.flatten() for m in mesh], axis=1)  # Shape: (M, 3)
        
        # Convert to q-space using vectors
        #spetial_points = np.array((0,0,0))
        # Apply mask using MaskStrategy instance
        mask = mask_strategy.generate_mask(q_points)
        q_points_masked = q_points[mask]
        q_space_masked = 2 * np.pi * np.dot(q_points_masked[:,0:supercell.shape[0]], B_)  # Shape: (M, D)
        
        return (q_space_masked)

    #@delayed
    def grid_generator_factory(dimensionality, step_in_frac):
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
            logger.error(f"Unsupported dimensionality: {dimensionality}")
            raise ValueError(f"Unsupported dimensionality: {dimensionality}")
    
    def _generate_grid(chunk_id: int, dimensionality, step_sizes, central_point, dist_from_atom_center, central_point_id):
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
        # Generate each dimension independently
        epsilon = 1e-8
        grids = []
        for i in range(dimensionality):
            dist = dist_from_atom_center[i]
            step = step_sizes[i]
            
            # If step is zero or the distance is too small to form more than one step:
            # Just produce a single point in that dimension
            if step <= 0 or dist <= step:
                logger.debug(f"Dimension {i}: step={step}, dist={dist}, generating single-point dimension.")
                grid = np.array([0.0])
            else:
                # Produce a range in this dimension
                start = -dist
                stop = dist + step - epsilon
                grid = np.arange(start, stop, step)
                # If no points generated due to floating point issues, fallback to single-point
                if grid.size == 0:
                    logger.debug(f"Dimension {i}: Could not form a range, fallback to single point.")
                    grid = np.array([0.0])
    
            grids.append(grid)
    
        # Now form the meshgrid from the possibly mixed-dimensional grids
        mesh = np.meshgrid(*grids, indexing='ij')
        grid_points = np.vstack([m.flatten() for m in mesh]).T + central_point
        grid_shapeNd = np.array(mesh[0].shape)
        return grid_points, grid_shapeNd
    
    
    def _process_chunk(chunk_data):
        """
        Processes a single chunk of points.

        Args:
            chunk_id (int): The ID of the chunk.
            mask (np.ndarray): Boolean array indicating which points in the chunk are uninitialized.
        """
       
        coordinates = np.array([pd['coordinates'] for pd in chunk_data])
        dist_from_atom_center = np.array([pd['dist_from_atom_center'] for pd in chunk_data])
        step_in_frac = np.array([pd['step_in_frac'] for pd in chunk_data])
        ids = np.array([pd['id'] for pd in chunk_data])
        dimensionality = coordinates.shape[1]

        all_grid_data = []
        all_grid_shapeNd = []

        for i in range(coordinates.shape[0]):
            central_point = coordinates[i]
            dist = dist_from_atom_center[i]
            step = step_in_frac[i]
            central_point_id = ids[i]
          
            grid_points,  grid_shapeNd = _generate_grid(chunk_id, dimensionality, step, central_point, dist, central_point_id)
            #amplitude_data = _generate_amplitude(chunk_id, central_point_id, i)

            # Collect data for this chunk
            all_grid_data.append(grid_points)
            all_grid_shapeNd.append(grid_shapeNd)
           # all_amplitude_data.append(amplitude_data)

        # Merge all grid_points and amplitudes for this chunk
        merged_grid_points = np.vstack(all_grid_data)
        merged_grid_shapeNd = np.vstack(all_grid_shapeNd)
        
        #merged_amplitude_data = np.vstack(all_amplitude_data)

        # Mark all points in this chunk as initialized
        #self.point_data.grid_amplitude_initialized[mask] = True
        #self.logger.debug(f"Chunk {chunk_id}: All uninitialized points marked as initialized.")
        return merged_grid_points, merged_grid_shapeNd
        
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
        # Combine to generate r_space_grid; customize as needed
        r_space_grid, grid_shapeNd = _process_chunk(chunk_data) #coordinates - dist_from_atom_center  # Placeholder operation
        return r_space_grid, grid_shapeNd
    
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
        ireciprocal_space: dict,
        B_: np.ndarray,
        mask_parameters: dict,
        mask_strategy: any,
        supercell:np.ndarray
    ) -> np.ndarray:
        """
        Synchronous wrapper to generate q-space grid.

        Args:
            ireciprocal_space (dict): HKL interval information.
            vectors (np.ndarray): Reciprocal space matrix B_.
            mask_parameters (dict): Parameters for mask generation.
            mask_strategy (MaskStrategy): Instance of mask strategy.

        Returns:
            np.ndarray: Masked q-space grid coordinates.
        """
        return generate_q_space_grid(ireciprocal_space, B_, mask_parameters, mask_strategy, supercell)

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
    total_reciprocal_points = np.zeros(1, dtype = np.int64)
    for ireciprocal_space in reciprocal_space_intervals_all:
        reciprocal_space_interval = {}
        # Dynamically handle present keys
        if 'h_range' in ireciprocal_space:
            reciprocal_space_interval['h_start'] = ireciprocal_space['h_range'][0]
            reciprocal_space_interval['h_end'] = ireciprocal_space['h_range'][1]
        if 'k_range' in ireciprocal_space:
            reciprocal_space_interval['k_start'] = ireciprocal_space['k_range'][0]
            reciprocal_space_interval['k_end'] = ireciprocal_space['k_range'][1]
        if 'l_range' in ireciprocal_space:
            reciprocal_space_interval['l_start'] = ireciprocal_space['l_range'][0]
            reciprocal_space_interval['l_end'] = ireciprocal_space['l_range'][1]
        total_reciprocal_points = total_reciprocal_points +  reciprocal_space_points_counter(reciprocal_space_interval, supercell) 
        print(f"total_reciprocal_points {total_reciprocal_points}")
    
    # Collect all delayed tasks and execute
    rifft_amplitudes_chunk_n = np.array([])
    for ireciprocal_space in reciprocal_space_intervals:
        ireciprocal_space_element_tasks = []
        q_space_grid = generate_q_space_grid_sync(ireciprocal_space, B_, MaskStrategyParameters, MaskStrategy, supercell)
        if "coeff" in parameters:
            #for element in unique_elements:
            task = process_ireciprocal_space_coeff(
                ireciprocal_space=ireciprocal_space,
                q_space_grid = q_space_grid,
                coeff=parameters["coeff"],
                B_=B_,
                mask_strategy=MaskStrategy,  # <-- Pass the MaskStrategy instance
                mask_parameters=MaskStrategyParameters
            )
            ireciprocal_space_element_tasks.append(task)  
        else:
            for element in unique_elements:
                task = process_ireciprocal_space_element(
                    ireciprocal_space=ireciprocal_space,
                    q_space_grid = q_space_grid,
                    element=element,
                    B_=B_,
                    mask_strategy=MaskStrategy,  # <-- Pass the MaskStrategy instance
                    mask_parameters=MaskStrategyParameters
                )
                ireciprocal_space_element_tasks.append(task)
        
        if ireciprocal_space_element_tasks[0] is None:
            # Handle the case where ireciprocal_space_element_tasks is None or empty
            reciprocal_space_intervals_task = None
        else:
            if "coeff" in parameters :
                if parameters["coeff"] is not None:
                    # Proceed only if we have tasks
                    ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]
                    element = "All"
                    q_space_grid = ireciprocal_space_element_tasks[0][1]
                
                    ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]     # from the first element
                    element = "All"     # from the first element
                    q_space_grid = ireciprocal_space_element_tasks[0][1] # from the first element
                    
                    # Sum q_amplitudes across all tuples in the list
                    q_amplitudes = np.sum([x[2] for x in ireciprocal_space_element_tasks], axis=0)
                    
                    # Sum q_amplitudes_av_final across all tuples in the list
                    q_amplitudes_av_final = np.sum([x[3] for x in ireciprocal_space_element_tasks], axis=0)
                    reciprocal_space_intervals_task = (ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
                else:
                    print(parameters["coeff"])
                    # Proceed only if we have tasks
                    ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]
                    element = "All"
                    q_space_grid = ireciprocal_space_element_tasks[0][2]
                
                    ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]     # from the first element
                    element = "All"     # from the first element
                    q_space_grid = ireciprocal_space_element_tasks[0][2] # from the first element
                    
                    # Sum q_amplitudes across all tuples in the list
                    q_amplitudes = np.sum([x[3] for x in ireciprocal_space_element_tasks], axis=0)
                    
                    # Sum q_amplitudes_av_final across all tuples in the list
                    q_amplitudes_av_final = np.sum([x[4] for x in ireciprocal_space_element_tasks], axis=0)
                    reciprocal_space_intervals_task = (ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
            else:
                   # Proceed only if we have tasks
                   ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]
                   element = "All"
                   q_space_grid = ireciprocal_space_element_tasks[0][2]
               
                   ireciprocal_space_id = ireciprocal_space_element_tasks[0][0]     # from the first element
                   element = "All"     # from the first element
                   q_space_grid = ireciprocal_space_element_tasks[0][2] # from the first element
                   
                   # Sum q_amplitudes across all tuples in the list
                   q_amplitudes = np.sum([x[3] for x in ireciprocal_space_element_tasks], axis=0)
                   
                   # Sum q_amplitudes_av_final across all tuples in the list
                   q_amplitudes_av_final = np.sum([x[4] for x in ireciprocal_space_element_tasks], axis=0)
                   reciprocal_space_intervals_task = (ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av_final)
                   
        # Create chunk_id tasks
        chunk_id_tasks = []

        if reciprocal_space_intervals_task is not None:
            #ireciprocal_space_result = compute(task)  # <-- Correct usage: compute returns a tuple
            #if ireciprocal_space_result is None:
            #    continue  # Skip if ireciprocal_space_result is None
            ireciprocal_space_id, element, q_space_grid, q_amplitudes, q_amplitudes_av = reciprocal_space_intervals_task
            for chunk_id in chunk_ids:
                
                
                
                
                rifft_amplitudes_chunk_n = process_chunk_id(
                    chunk_id=chunk_id,
                    ireciprocal_space_id=ireciprocal_space_id,
                    q_space_grid=q_space_grid,
                    q_amplitudes=q_amplitudes,
                    q_amplitudes_av=q_amplitudes_av,
                    rifft_saver=rifft_saver,
                    point_data_list=point_data_list,
                    rifft_amplitudes_chunk_n = rifft_amplitudes_chunk_n,
                    total_reciprocal_points = total_reciprocal_points,
                    #nmasked_reciprocal_space_points = nmasked_reciprocal_space_points
                )
                #chunk_id_tasks.append(chunk_task)
    
        # Execute all chunk_id tasks collectively for better parallelism
        #if chunk_id_tasks:
        #    compute(*chunk_id_tasks)

    logger.info("Completed compute_amplitudes_delta")
