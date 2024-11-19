# main.py

import os
import logging
from utilities.logger_config import setup_logging

from factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from factories.parameters_processor_factory import ParametersProcessorFactoryProvider
from factories.point_processor_factory import PointProcessorFactory
from data_structures.point_data import PointData
from processors.point_data_processor import PointDataProcessor
from data_storage.rifft_in_data_saver import RIFFTInDataSaver

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
            
            # Initialize DataSaver
            data_saver = RIFFTInDataSaver(output_dir='../tests/config/processed_point_data', file_extension='hdf5')

            # Initialize PointDataProcessor
            point_data_processor = PointDataProcessor(data_saver=data_saver, save_rifft_coordinates=rspace_info.get('save_rifft_coordinates', False))

            # Process point data
            point_data_processor.process_point_data(point_data)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
if __name__ == '__main__':
    main()
