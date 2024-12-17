# main.py

import os
import numpy as np
import logging
from utilities.logger_config import setup_logging
from processors.rifft_grid_generator import GridGenerator1D, GridGenerator2D, GridGenerator3D
from factories.configuration_processor_factory import ConfigurationProcessorFactoryProvider
from factories.parameters_processor_factory import ParametersProcessorFactoryProvider
from factories.point_processor_factory import PointProcessorFactory
from data_structures.point_data import PointData
from processors.point_data_processor import PointDataProcessor
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
from processors.point_data_hkl_manager import HKLIntervalManager
from managers.database_manager import DatabaseManager  # Import the DatabaseManager
from functions.split_point_data import split_point_data

import h5py  # For reading HDF5 files
import json  # For secure parsing

#from form_factors.default_form_factor_calculator import DefaultFormFactorCalculator
from strategies.shape_strategies import SphereShapeStrategy
from processors.amplitude_delta_calculator import compute_amplitudes_delta

from form_factors.form_factor_factory_producer import FormFactorFactoryProducer

def main():
    setup_logging()
    logger = logging.getLogger('app')

    # Configuration file processing
    config_file_path = '../tests/config/pmn30pt_533kAVERAGE.rmc6f'
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
            
        # Databaser data processing
        try:     


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
            
            # Initialize DatabaseManager
            db_path = '../tests/config/processed_point_data/point_hkl_associations.db'
            db_manager = DatabaseManager(db_path)            
            # Insert PointData in batch
            point_ids = db_manager.insert_point_data_batch(point_data_list)
            logger.info(f"Inserted or found {len(point_ids)} PointData entries in the database.")
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

            # ADDD YOUR CODE HERE
            if unsaved_associations:
                logger.info("Starting processing of unsaved associations.")
                
                # Step 1: Extract unique point_ids and hkl_ids from unsaved_associations
                point_ids = list(set([assoc[0] for assoc in unsaved_associations]))
                hkl_ids = list(set([assoc[1] for assoc in unsaved_associations]))
                
                logger.info(f"Unique Point IDs to process: {len(point_ids)}")
                logger.info(f"Unique HKL Interval IDs to process: {len(hkl_ids)}")
                
                # Step 2: Retrieve point data for these point_ids
                all_point_data = db_manager.get_point_data_for_point_ids(point_ids)
                
                if not all_point_data:
                    logger.warning("No point data found for the unsaved associations.")
                else:
                    logger.info(f"Retrieved data for {len(all_point_data)} points.")
                
                # Step 3: Retrieve HKL interval data for these hkl_ids
                hkl_intervals = []
                if hkl_ids:
                    placeholders = ','.join(['?'] * len(hkl_ids))
                    query = f"SELECT * FROM HKLInterval WHERE id IN ({placeholders})"
                    #try:
                    db_manager.cursor.execute(query, hkl_ids)
                    rows = db_manager.cursor.fetchall()
                    for row in rows:
                        hkl_interval = {
                            'id': row[0],
                            'h_start': row[1],
                            'h_end': row[2],
                            'k_start': row[3],
                            'k_end': row[4],
                            'l_start': row[5],
                            'l_end': row[6]
                        }
                        hkl_intervals.append(hkl_interval)
                    logger.info(f"Retrieved data for {len(hkl_intervals)} HKL intervals.")
                    # except sqlite3.Error as e:
                        # logger.error(f"Failed to retrieve HKL intervals: {e}")
                
                # Step 4: Prepare compute_params for amplitude delta computation
                compute_params = {
                    "hkl_intervals": hkl_intervals,
                    "point_data_list": [
                        {
                            'central_point_id': pd['central_point_id'],
                            'coordinates': pd['coordinates'],
                            'dist_from_atom_center': pd['dist_from_atom_center'],
                            'step_in_frac': pd['step_in_frac'],
                            'chunk_id': pd['chunk_id'],
                            'grid_amplitude_initialized': pd['grid_amplitude_initialized'],
                            'id': pd['central_point_id']  # Assuming 'central_point_id' is unique and serves as 'id'
                        }
                        for pd in all_point_data
                    ],
                    "original_coords": original_coords.to_numpy(),
                    "average_coords": average_coords.to_numpy(),
                    "elements": elements.to_numpy(),
                    "vectors": vectors,
                    "supercell": supercell
                }
                
                # Step 5: Initialize MaskStrategy instance
                mask_parameters = parameters["peakInfo"]
                mask_strategy = SphereShapeStrategy(mask_parameters)
                
                # Step 6: Initialize FormFactorFactoryProducer
                form_factor_factory_producer = FormFactorFactoryProducer.get_factory('neutron')
                form_factor_factory_calculator = form_factor_factory_producer.create_calculator(method='default')
                # Step 7: Compute amplitude deltas
                try:
                    amplitude_data_chunks = compute_amplitudes_delta(
                        parameters=compute_params,
                        FormFactorFactoryProducer=form_factor_factory_calculator,
                        MaskStrategy=mask_strategy,
                        MaskStrategyParameters=mask_parameters,
                        db_manager=db_manager,
                        output_dir='../tests/config/processed_point_data',
                        point_data_processor = point_data_processor
                    )
                    logger.info("Amplitude delta computation completed successfully.")
                except Exception as e:
                    logger.error(f"Failed to compute amplitude deltas: {e}")
                    amplitude_data_chunks = {}
                    db_manager.close()

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
if __name__ == '__main__':
    main()
