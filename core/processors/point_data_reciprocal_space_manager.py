# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:43:52 2024

@author: Maksim Eremenko
"""

# processors/point_data_reciprocal_space_manager.py
# managers/reciprocal_space_interval_manager.py

import numpy as np
import h5py
import os
import logging
from processors.reciprocal_space_interval_generator import ReciprocalSpaceIntervalGenerator

import numpy as np
import h5py
import os
import logging

# Make sure to import the dimension-aware ReciprocalSpaceIntervalGenerator:
from processors.reciprocal_space_interval_generator import ReciprocalSpaceIntervalGenerator

class ReciprocalSpaceIntervalManager:
    def __init__(self, hdf5_file_path: str, parameters: dict, supercell):
        """
        Initializes the ReciprocalSpaceIntervalManager.

        Args:
            hdf5_file_path (str): Path to the HDF5 file for saving/loading data.
            parameters (dict): The parameters dictionary containing peakInfo and rspace_info.
            supercell (tuple/list): The supercell dimensions (1D, 2D, or 3D).
        """
        self.hdf5_file_path = hdf5_file_path
        self.parameters = parameters
        self.supercell = supercell
        self.dim = len(self.supercell)
        self.reciprocal_space_intervals = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fresh_start = self.parameters.get('rspace_info', {}).get('fresh_start', True)
        self.previous_max_limit = None

    def process_reciprocal_space_intervals(self):
        """
        Handles the entire process of managing reciprocal_space intervals.
        """
        if not self.fresh_start and os.path.exists(self.hdf5_file_path):
            # Load existing reciprocal_space data
            if self.load_from_hdf5():
                # Get previous max limits
                self.previous_max_limit = self.get_previous_max_limit()
                self.logger.info(f"Previous max reciprocal_space limits: {self.previous_max_limit}")
            else:
                self.logger.warning("Failed to load existing reciprocal_space data. Starting fresh.")
                self.fresh_start = True
        else:
            self.fresh_start = True

        # Initialize ReciprocalSpaceIntervalGenerator
        reciprocal_space_generator = ReciprocalSpaceIntervalGenerator(
            self.supercell, 
            previous_max_limit=None if self.fresh_start else self.previous_max_limit
        )

        peak_info = self.parameters.get('peakInfo', {})
        reciprocal_space_limits_list = peak_info.get('reciprocal_space_limits', [])

        # Dimension-based default values if not provided
        default_limit = [10.0]*self.dim
        default_subvolume_step = [5.0]*self.dim

        all_reciprocal_space_intervals = []
        for reciprocal_space_limit_info in reciprocal_space_limits_list:
            limit = reciprocal_space_limit_info.get('limit', default_limit)
            subvolume_step = reciprocal_space_limit_info.get('subvolume_step', default_subvolume_step)

            # Ensure these are truncated or extended to match dimension (just in case)
            limit = limit[:self.dim]
            subvolume_step = subvolume_step[:self.dim]

            reciprocal_space_intervals = reciprocal_space_generator.generate_intervals(limit, subvolume_step)
            
            # Convert np.float64 to Python floats
            reciprocal_space_intervals = self._convert_intervals_to_python_floats(reciprocal_space_intervals)

            all_reciprocal_space_intervals.extend(reciprocal_space_intervals)

        if self.fresh_start:
            # Set new reciprocal_space intervals
            self.reciprocal_space_intervals = all_reciprocal_space_intervals
        else:
            # Update with new intervals
            self.update_intervals(all_reciprocal_space_intervals)

        # Save the reciprocal_space data
        self.save_to_hdf5()

    def update_intervals(self, new_reciprocal_space_intervals):
        """
        Updates the reciprocal_space intervals by appending new intervals.

        Args:
            new_reciprocal_space_intervals (list): New reciprocal_space intervals to add.
        """
        existing_intervals_set = set(self._interval_to_str(interval) for interval in self.reciprocal_space_intervals)
        new_intervals_set = set(self._interval_to_str(interval) for interval in new_reciprocal_space_intervals)
        intervals_to_add = [interval for interval in new_reciprocal_space_intervals if self._interval_to_str(interval) not in existing_intervals_set]
        self.new_intervals_set = new_intervals_set
        self.reciprocal_space_intervals.extend(intervals_to_add)
        self.logger.info(f"Added {len(intervals_to_add)} new reciprocal_space intervals. Total intervals: {len(self.reciprocal_space_intervals)}")

    def save_to_hdf5(self):
        """
        Saves the reciprocal_space intervals to an HDF5 file.
        """
        try:
            with h5py.File(self.hdf5_file_path, 'w') as h5file:
                # Save reciprocal_space_intervals
                reciprocal_space_intervals_grp = h5file.create_group('reciprocal_space_intervals')
                for idx, interval in enumerate(self.reciprocal_space_intervals):
                    interval_str = self._interval_to_str(interval)
                    reciprocal_space_intervals_grp.attrs[str(idx)] = interval_str
            self.logger.info(f"reciprocal_space intervals saved to {self.hdf5_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save reciprocal_space data to HDF5: {e}")

    def load_from_hdf5(self):
        """
        Loads reciprocal_space intervals from an HDF5 file.
        """
        if not os.path.exists(self.hdf5_file_path):
            self.logger.error(f"HDF5 file {self.hdf5_file_path} does not exist.")
            return False
        try:
            with h5py.File(self.hdf5_file_path, 'r') as h5file:
                # Load reciprocal_space_intervals
                reciprocal_space_intervals_grp = h5file['reciprocal_space_intervals']
                self.reciprocal_space_intervals = []
                for idx in sorted(reciprocal_space_intervals_grp.attrs.keys(), key=int):
                    interval_str = reciprocal_space_intervals_grp.attrs[idx]
                    interval = eval(interval_str)
                    self.reciprocal_space_intervals.append(interval)
            self.logger.info(f"reciprocal_space intervals loaded from {self.hdf5_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load reciprocal_space data from HDF5: {e}")
            return False

    def get_previous_max_limit(self):
        """
        Determines the previous maximum reciprocal_space limits from the existing intervals.

        Returns:
            tuple: Maximum limits of the form (h_max) for 1D, (h_max, k_max) for 2D,
                   or (h_max, k_max, l_max) for 3D.
        """
        if not self.reciprocal_space_intervals:
            return None

        # Extract max for h
        h_max = max(interval['h_range'][1] for interval in self.reciprocal_space_intervals)

        if self.dim == 1:
            return (h_max,)

        # Extract max for k
        k_max = max(interval['k_range'][1] for interval in self.reciprocal_space_intervals)

        if self.dim == 2:
            return (h_max, k_max)

        # Extract max for l (consider only non-zero intervals)
        l_values = [interval['l_range'][1] for interval in self.reciprocal_space_intervals if interval['l_range'][1] != 0.0]
        l_max = max(l_values) if l_values else 0.0

        return (h_max, k_max, l_max)

    def _interval_to_str(self, interval):
        """
        Converts an interval dictionary to a string representation.

        Args:
            interval (dict): The interval dictionary.

        Returns:
            str: String representation of the interval.
        """
        return str(interval)

    def _convert_intervals_to_python_floats(self, intervals):
        """
        Converts np.float64 values in the intervals to Python floats.

        Args:
            intervals (list): List of interval dictionaries.

        Returns:
            list: Intervals with Python floats instead of np.float64.
        """
        converted_intervals = []
        for interval in intervals:
            converted_interval = {}
            for key, val in interval.items():
                # val is typically a tuple (start, end)
                start, end = val
                converted_interval[key] = (float(start), float(end))
            converted_intervals.append(converted_interval)
        return converted_intervals


