# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:05:08 2024

@author: Maksim Eremenko
"""

from dask import delayed, compute
from utilities.nufft_wrapper import execute_nufft, execute_inverse_nufft
from data_storage.rifft_in_data_saver import RIFFTInDataSaver
import numpy as np


def compute_amplitudes_delta(
    parameters,
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters,
    db_manager,
):
    """
    Computes delta amplitudes using NUFFT and nested parallelism.
    
    Args:
        parameters (dict): A dictionary containing required inputs such as hkl_intervals, point data, coordinates, etc.
        FormFactorFactoryProducer: Factory to create form factor calculators.
        MaskStrategy: Strategy for masking data points.
        MaskStrategyParameters: Parameters for the masking strategy.
        db_manager: Database manager to track progress.
    """
    # Extract inputs from parameters
    hkl_intervals = parameters["hkl_intervals"]
    point_data_list = parameters["point_data_list"]
    original_coords = parameters["original_coords"]
    average_coords = parameters["average_coords"]
    elements = parameters["elements"]
    un_elements = np.unique(elements)
    c = np.ones(original_coords.shape[0], dtype=complex)

    rifft_amplitudes = initialize_rifft_amplitudes(db_manager)
    pending_parts = db_manager.get_pending_parts()
    chunk_ids = db_manager.get_pending_chunk_ids()

    def process_reciprocal_space_coordinates(reciprocal_space_coordinates_parts):
        """
        Process a set of reciprocal space coordinates.
        """
        for un_element in un_elements:
            mask_elements = elements == un_element
            reciprocal_space_coordinates_nd = generate_meshgrid(reciprocal_space_coordinates_parts)
            reciprocal_space_coordinates_unrolled = reciprocal_space_coordinates_nd.reshape(-1, reciprocal_space_coordinates_nd.shape[-1])
            
            mask_ = MaskStrategy(reciprocal_space_coordinates_unrolled, MaskStrategyParameters)
            reciprocal_space_coordinates_masked = reciprocal_space_coordinates_unrolled[mask_]
            q_space_grid = 2 * np.pi * np.dot(reciprocal_space_coordinates_masked, parameters["vectors"].T)

            ff_calculator = FormFactorFactoryProducer.get_factory('default').create_calculator(method='default')
            ff = ff_calculator.calculate(q_space_grid, un_element)

            q_amplitudes = ff * execute_nufft(original_coords[mask_elements], c, q_space_grid, eps=1e-5)
            q_amplitudes_av = execute_nufft(average_coords[mask_elements], c, q_space_grid, eps=1e-5)
            q_amplitudes_delta = execute_nufft(original_coords[mask_elements] - average_coords[mask_elements], c, q_space_grid, eps=1e-5)

            q_amplitudes_av *= ff * q_amplitudes_av * q_amplitudes_delta / c.size
            return q_space_grid, q_amplitudes, q_amplitudes_av

    def process_chunk_id(chunk_id, q_space_grid, q_amplitudes, q_amplitudes_av):
        """
        Process a single chunk ID.
        """
        chunk_data = [pd for pd in point_data_list if pd["chunk_id"] == chunk_id]
        rifft_space_grid = generate_rifft_grid(chunk_data)

        r_amplitudes_partial = execute_inverse_nufft(
            q_space_grid, q_amplitudes - q_amplitudes_av, rifft_space_grid
        )

        filename = RIFFTInDataSaver.generate_filename(chunk_id)
        rifft_amplitudes[chunk_id] = RIFFTInDataSaver.load_data(filename)
        rifft_amplitudes[chunk_id] += r_amplitudes_partial
        RIFFTInDataSaver.save_data(rifft_amplitudes[chunk_id], filename)
        db_manager.update_saved_status_for_chunk_or_point(None, chunk_id=chunk_id, status=1)

    # Nested parallelism with Dask
    tasks = []
    for reciprocal_space_coordinates_parts in pending_parts:
        q_space_grid, q_amplitudes, q_amplitudes_av = process_reciprocal_space_coordinates(reciprocal_space_coordinates_parts)

        for chunk_id in chunk_ids:
            tasks.append(delayed(process_chunk_id)(chunk_id, q_space_grid, q_amplitudes, q_amplitudes_av))

    compute(*tasks)


def initialize_rifft_amplitudes(db_manager):
    """
    Initializes rifft_amplitudes from the database or creates a new structure.
    """
    try:
        return db_manager.load_rifft_amplitudes()
    except FileNotFoundError:
        return {}
    

def generate_meshgrid(reciprocal_space_coordinates_parts):
    """
    Generates a meshgrid for reciprocal space coordinates.
    """
    return np.meshgrid(*[np.linspace(start, end, num) for start, end, num in reciprocal_space_coordinates_parts])


def generate_rifft_grid(chunk_data):
    """
    Generates the RIFFT grid based on chunk data.
    """
    return np.array([
        [pd["coordinates"], pd["dist_from_atom_center"], pd["step_in_frac"]]
        for pd in chunk_data
    ])
