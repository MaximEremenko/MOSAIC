# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:52:11 2024

@author: Maksim Eremenko
"""

# main.py

import os
import logging
from utilities.logger_config import setup_logging

from factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from factories.parameters_processor_factory import ParametersProcessorFactoryProvider
from factories.point_processor_factory import PointProcessorFactory
from data_structures.point_data import PointData
from functions.split_point_data import split_point_data
from processors.point_data_processor import PointDataProcessor
from data_storage.rifft_in_data_saver import RIFFTInDataSaver


def main():
    # Configuration file processing
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

        # If processed using the configuration file, save to HDF5
        if file_type != 'hdf5':
            from data_storage.hdf5_data_storage import HDF5ConfigDataSaver
            data_saver = HDF5ConfigDataSaver(config_hdf5_file_path)
            data = {
                'original_coords': original_coords,
                'average_coords': average_coords,
                'elements': elements,
                'refnumbers': refnumbers,
                'vectors': vectors,
                'metric': metric,
                'supercell': supercell
            }
            data_saver.save_data(data)
            logger.info(f"Processed data saved to HDF5 file: {config_hdf5_file_path}")

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
            point_data_hdf5_path = '../tests/config/point_data.hdf5'  # Path to save/load point data
            # Set HDF5 file path in parameters
            parameters['hdf5_file_path'] = point_data_hdf5_path
            # Create the appropriate point processor
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
            
            # Access data from point_data
            coordinates = point_data.coordinates               # NumPy array of shape (N, D)
            dist_from_atom_center = point_data.dist_from_atom_center  # NumPy array
            step_in_frac = point_data.step_in_frac             # NumPy array
            central_point_ids = point_data.central_point_ids   # NumPy array
            
            # Example: Output some data
            logger.debug(f"Coordinates shape: {coordinates.shape}")
            logger.debug(f"First few coordinates:\n{coordinates[:5]}")
            logger.debug(f"Central Point IDs:\n{central_point_ids[:5]}")
            
            point_data_chunks = split_point_data(point_data, 10)  # Split into chunks of 10 points each
            
            # Initialize DataSaver
            data_saver = RIFFTInDataSaver(output_dir='../tests/config/processed_point_data', file_extension='hdf5')

            # Process each point_data chunk
            point_data_file_paths = []
            for chunk_id, point_data_chunk in enumerate(point_data_chunks):
                # Each chunk should include 'central_point_ids'
                chunk_dict = {
                    'coordinates': point_data_chunk.coordinates,
                    'dist_from_atom_center': point_data_chunk.dist_from_atom_center,
                    'step_in_frac': point_data_chunk.step_in_frac,
                    'central_point_ids': point_data_chunk.central_point_ids
                }

                # Initialize PointDataProcessor
                point_data_processor = PointDataProcessor(data_saver=data_saver, save_rifft_coordinates=True)
                saved_files = point_data_processor.process_point_data_chunk(chunk_id, chunk_dict)
                if saved_files:
                    if 'grid_points' in saved_files:
                        logger.info(f"Processed point_data chunk {chunk_id}. Grid points saved to {saved_files['grid_points']}")
                    if 'amplitudes' in saved_files:
                        logger.info(f"Processed point_data chunk {chunk_id}. Amplitudes saved to {saved_files['amplitudes']}")
                    point_data_file_paths.append(saved_files)
                else:
                    logger.info(f"No data generated for point_data chunk {chunk_id}.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
