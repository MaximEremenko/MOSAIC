# main.py

import os
import numpy as np
import logging
from utilities.logger_config import setup_logging

from factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from factories.parameters_processor_factory import ParametersProcessorFactoryProvider
from factories.point_processor_factory import PointProcessorFactory
from data_structures.point_data import PointData
from processors.point_data_processor import PointDataProcessor
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from processors.point_data_hkl_manager import HKLIntervalManager
from managers.database_manager import DatabaseManager  # Import the DatabaseManager

import h5py  # For reading HDF5 files
import json  # For secure parsing


def main():
    setup_logging()
    logger = logging.getLogger('app')

    # Configuration file processing
    config_file_path = '../tests/config/pmn_300k_RN.rmc6f'
    processor_type = 'calculate'  # 'read' or 'calculate'
    average_file_path = '../tests/config/external_average_file.rmc6f'  # Only needed when processor_type is 'read'

    # Define the corresponding HDF5 file path for configuration data
    config_hdf5_file_path = os.path.splitext(config_file_path)[0] + '_config.hdf5'

    try:
        if os.path.exists(config_hdf5_file_path):
            # If HDF5 file exists, use it
            file_type = 'hdf5'
            logger.info(f"HDF5 file found: {config_hdf5_file_path}. Loading data from HDF5.")
        else:
            # Else, determine the file type based on the configuration file
            from utilities.utils import determine_configuration_file_type
            file_type = determine_configuration_file_type(config_file_path)
            logger.info(f"No HDF5 file found. Processing configuration file: {config_file_path}")

        # Use the factory provider to get the appropriate factory
        factory = ConfigurationProcessorFactoryProvider.get_factory(file_type)

        if file_type == 'hdf5':
            # For HDF5, processor_type and average_file_path are irrelevant
            processor = factory.create_processor(config_hdf5_file_path)
        else:
            # For configuration files, pass processor_type and average_file_path
            processor = factory.create_processor(config_file_path, processor_type, average_file_path)

        processor.process()

        # Access the required outputs
        original_coords = processor.get_coordinates()
        average_coords = processor.get_average_coordinates()
        elements = processor.get_elements()
        refnumbers = processor.get_refnumbers()
        vectors = processor.get_vectors()
        metric = processor.get_metric()
        supercell = processor.get_supercell()
        
        logger.info("Original Coordinates:\n%s", original_coords.head())
        logger.info("Average Coordinates:\n%s", average_coords.head())
        logger.info("Supercell Dimensions:\n%s", supercell)
        logger.info("Elements:\n%s", elements.head())
        logger.info("Vectors (Base Vectors):\n%s", vectors)
        logger.info("Metric:\n%s", metric)

        # Parameters file processing
        parameters_file_path = '../tests/config/input_parameters.json'
        parameters_hdf5_path = '../tests/config/parameters.hdf5'
        parameters_source_type = 'file'  # 'file' or 'api'

        try:
            parameters_factory_provider = ParametersProcessorFactoryProvider()
            parameters_factory = parameters_factory_provider.get_factory()

            if os.path.exists(parameters_hdf5_path):
                logger.info(f"Parameters HDF5 file found: {parameters_hdf5_path}. Loading parameters from HDF5.")
                processor = parameters_factory.create_processor(
                    source=parameters_hdf5_path,
                    source_type='hdf5',
                    hdf5_file_path=parameters_hdf5_path
                )
            else:
                logger.info(f"No parameters HDF5 file found. Processing parameters from source: {parameters_file_path}")
                processor = parameters_factory.create_processor(
                    source=parameters_file_path,
                    source_type=parameters_source_type,
                    hdf5_file_path=parameters_hdf5_path
                )
            processor.process()

            parameters = processor.get_parameters()
            logger.info("Parameters Loaded:")
            logger.info(parameters)
        except Exception as e:
            logger.error(f"An error occurred while processing parameters: {e}")

        # Point data processing
        try:
            # Determine the method
            rspace_info = parameters.get('rspace_info', {})
            method = rspace_info.get('method')
            point_data_hdf5_path = '../tests/config/processed_point_data/point_data.hdf5'  # Path to save/load point data
            # Set HDF5 file path in parameters
            parameters['hdf5_file_path'] = point_data_hdf5_path

            # Specify the number of chunks
            num_chunks = rspace_info.get('num_chunks', 10)  # Default to 10 if not specified
            logger.info(f"Number of chunks to create: {num_chunks}")

            # Create the appropriate point processor with num_chunks
            point_processor = PointProcessorFactory.create_processor(method, parameters, average_structure={
                'average_coords': average_coords,
                'elements': elements,
                'refnumbers': refnumbers,
                'vectors': vectors,
                'metric': metric,
                'supercell': supercell  
            })

            if point_processor is None:
                logger.error(f"Unknown method '{method}' specified.")
                return

            # Process parameters to prepare point data
            point_processor.process_parameters()
            
            point_data = point_processor.get_point_data()
            
            logger.info("Point data prepared for calculation.")
            
            # Initialize DatabaseManager
            db_path = '../tests/config/processed_point_data/point_hkl_associations.db'
            db_manager = DatabaseManager(db_path)

            # Convert PointData to list of dictionaries for batch insertion
            point_data_list = []
            for i in range(point_data.central_point_ids.size):
                pd_dict = {
                    'central_point_id': int(point_data.central_point_ids[i]),
                    'coordinates': point_data.coordinates[i].tolist(),
                    'dist_from_atom_center': point_data.dist_from_atom_center[i].tolist(),
                    'step_in_frac': point_data.step_in_frac[i].tolist(),
                    'chunk_id': int(point_data.chunk_ids[i]),
                    'grid_amplitude_initialized': int(point_data.grid_amplitude_initialized[i])
                }
                point_data_list.append(pd_dict)

            # Insert PointData in batch
            point_ids = db_manager.insert_point_data_batch(point_data_list)
            logger.info(f"Inserted or found {len(point_ids)} PointData entries in the database.")

            # HKL Interval processing
            hkl_data_hdf5_path = '../tests/config/processed_point_data/point_hkl_data.hdf5'
            point_data_hkl_manager = HKLIntervalManager(
                hdf5_file_path=hkl_data_hdf5_path,
                parameters=parameters,
                supercell=supercell
            )
            point_data_hkl_manager.process_hkl_intervals()

            # Extract HKLIntervals as list of dictionaries
            hkl_intervals = []
            for d in point_data_hkl_manager.hkl_intervals:
                hkl_interval = {
                    'h_range': d['h_range'],
                    'k_range': d['k_range'],
                    'l_range': d['l_range']
                }
                hkl_intervals.append(hkl_interval)

            logger.info(f"Loaded {len(hkl_intervals)} hkl_intervals from HKLIntervalManager.")

            # Insert HKLIntervals in batch
            hkl_ids = db_manager.insert_hkl_interval_batch(hkl_intervals)
            logger.info(f"Inserted or found {len(hkl_ids)} HKLInterval entries in the database.")

            # Create associations between PointData and HKLIntervals in bulk
            # Generate all possible (point_id, hkl_id) pairs
            associations = []
            for point_id in point_ids:
                for hkl_id in hkl_ids:
                    associations.append((point_id, hkl_id))

            logger.info(f"Creating {len(associations)} associations between PointData and HKLIntervals.")

            # Insert associations in batch
            db_manager.associate_point_hkl_batch(associations)
            logger.info("All PointData and HKLInterval associations have been created or already exist.")

            # Optional: Process unsaved associations
            unsaved_associations = db_manager.get_unsaved_associations()
            logger.info(f"Processing {len(unsaved_associations)} unsaved associations.")

            # Example processing: Placeholder for actual data saving logic
            # Replace this loop with actual processing code
            updates = []
            for point_id, hkl_id in unsaved_associations:
                # TODO: Implement your data processing and saving logic here
                # For demonstration, we'll assume processing is successful and mark as saved
                # e.g., process_data(point_id, hkl_id)
                
                # After successful processing, prepare to update the saved status
                updates.append((1, point_id, hkl_id))  # 1 represents True

            if updates:
                db_manager.update_saved_status_batch(updates)
                logger.info(f"Updated saved status for {len(updates)} associations.")

            # Close the database connection
            db_manager.close()

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
if __name__ == '__main__':
    main()
