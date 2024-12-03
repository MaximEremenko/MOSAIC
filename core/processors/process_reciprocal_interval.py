# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:20:41 2024

@author: Maksim Eremenko
"""

# processors/process_reciprocal_interval.py

import numpy as np
from mask_generator import MaskGenerator
from nufft_calculators import NUFFTCalculator  # We'll define this later
from form_factors_generator import FormFactorsGenerator  # We'll define this later

class ProcessReciprocalInterval:
    def __init__(self, mask_generator, nufft_calculator, form_factors_generator, parameters):
        """
        Initializes the ProcessReciprocalInterval processor.

        Args:
            mask_generator (MaskGenerator): An instance of MaskGenerator.
            nufft_calculator (NUFFTCalculator): An instance of NUFFTCalculator.
            form_factors_generator (FormFactorsGenerator): An instance of FormFactorsGenerator.
            parameters (dict): Additional parameters required for processing.
        """
        self.mask_generator = mask_generator
        self.nufft_calculator = nufft_calculator
        self.form_factors_generator = form_factors_generator
        self.parameters = parameters

    def process(self, hkl_interval):
        """
        Processes the given hkl interval.

        Args:
            hkl_interval (dict): Dictionary containing hkl interval ranges.

        Returns:
            dict: Dictionary containing q_amplitudes and q_amplitudes_av.
        """
        # Convert hkl_interval to hkl_grid
        hkl_grid = self._convert_hkl_interval_to_grid(hkl_interval)
        # Generate mask
        mask = self.mask_generator.generate_mask(hkl_grid)
        # Apply mask
        hkl_grid_masked = self.mask_generator.apply_mask(mask, hkl_grid)
        if hkl_grid_masked.size == 0:
            return None  # No data to process
        # Perform NUFFT computations
        q_amplitudes, q_amplitudes_av = self._perform_nufft(hkl_grid_masked)
        return {
            'q_amplitudes': q_amplitudes,
            'q_amplitudes_av': q_amplitudes_av,
            'hkl_grid_masked': hkl_grid_masked
        }

    def _convert_hkl_interval_to_grid(self, hkl_interval):
        """
        Converts hkl_interval to hkl_grid.

        Args:
            hkl_interval (dict): Dictionary containing hkl interval ranges.

        Returns:
            np.ndarray: An array of hkl points.
        """
        h_range = np.arange(hkl_interval['h_range'][0], hkl_interval['h_range'][1] + self.parameters['hkl_step'], self.parameters['hkl_step'])
        k_range = np.arange(hkl_interval['k_range'][0], hkl_interval['k_range'][1] + self.parameters['hkl_step'], self.parameters['hkl_step'])
        l_range = np.arange(hkl_interval['l_range'][0], hkl_interval['l_range'][1] + self.parameters['hkl_step'], self.parameters['hkl_step'])
        hkl_mesh = np.array(np.meshgrid(h_range, k_range, l_range)).T.reshape(-1, 3)
        return hkl_mesh

    def _perform_nufft(self, hkl_grid_masked):
        """
        Performs NUFFT computations.

        Args:
            hkl_grid_masked (np.ndarray): Masked hkl grid points.

        Returns:
            tuple: (q_amplitudes, q_amplitudes_av)
        """
        # Generate form factors
        form_factors = self.form_factors_generator.generate_form_factors(hkl_grid_masked)
        # Compute s, t, u for NUFFT
        s, t, u = self._compute_stu(hkl_grid_masked)
        # Perform NUFFT
        q_amplitudes = self.nufft_calculator.compute_nufft1(s, t, u, form_factors)
        q_amplitudes_av = self.nufft_calculator.compute_nufft1_av(s, t, u)
        return q_amplitudes, q_amplitudes_av

    def _compute_stu(self, hkl_grid_masked):
        """
        Computes s, t, u from hkl_grid_masked.

        Args:
            hkl_grid_masked (np.ndarray): Masked hkl grid points.

        Returns:
            tuple: (s, t, u)
        """
        # Assuming B_ matrix is provided in parameters
        B_ = self.parameters['B_']
        q_space_grid = 2 * np.pi * hkl_grid_masked @ B_.T
        s = q_space_grid[:, 0]
        t = q_space_grid[:, 1]
        u = q_space_grid[:, 2]
        return s, t, u
