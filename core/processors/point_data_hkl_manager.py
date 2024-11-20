# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:43:52 2024

@author: Maksim Eremenko
"""

# processors/point_data_hkl_manager.py
# managers/hkl_interval_manager.py

import numpy as np
import h5py
import os
import logging
from processors.hkl_interval_generator import HKLIntervalGenerator

class HKLIntervalManager:
    def __init__(self, hdf5_file_path: str, parameters: dict, supercell):
        """
        Initializes the HKLIntervalManager.

        Args:
            hdf5_file_path (str): Path to the HDF5 file for saving/loading data.
            parameters (dict): The parameters dictionary containing peakInfo and rspace_info.
            supercell (tuple/list): The supercell dimensions.
        """
        self.hdf5_file_path = hdf5_file_path
        self.parameters = parameters
        self.supercell = supercell
        self.hkl_intervals = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fresh_start = self.parameters.get('rspace_info', {}).get('fresh_start', True)
        self.previous_max_limit = None

    def process_hkl_intervals(self):
        """
        Handles the entire process of managing hkl intervals.
        """
        if not self.fresh_start and os.path.exists(self.hdf5_file_path):
            # Load existing hkl data
            if self.load_from_hdf5():
                # Get previous max limits
                self.previous_max_limit = self.get_previous_max_limit()
                self.logger.info(f"Previous max hkl limits: {self.previous_max_limit}")
            else:
                self.logger.warning("Failed to load existing hkl data. Starting fresh.")
                self.fresh_start = True
        else:
            self.fresh_start = True

        # Initialize HKLIntervalGenerator
        hkl_generator = HKLIntervalGenerator(self.supercell, previous_max_limit=None if self.fresh_start else self.previous_max_limit)

        # Generate hkl intervals
        peak_info = self.parameters.get('peakInfo', {})
        hkl_limits_list = peak_info.get('hkl_limits', [])
        all_hkl_intervals = []
        for hkl_limit_info in hkl_limits_list:
            limit = hkl_limit_info.get('limit', [10.0, 10.0, 10.0])
            subvolume_step = hkl_limit_info.get('subvolume_step', [5.0, 5.0, 5.0])
            hkl_intervals = hkl_generator.generate_intervals(limit, subvolume_step)
            all_hkl_intervals.extend(hkl_intervals)

        if self.fresh_start:
            # Set new hkl intervals
            self.hkl_intervals = all_hkl_intervals
        else:
            # Update with new intervals
            self.update_intervals(all_hkl_intervals)

        # Save the hkl data
        self.save_to_hdf5()

    def update_intervals(self, new_hkl_intervals):
        """
        Updates the hkl intervals by appending new intervals.

        Args:
            new_hkl_intervals (list): New hkl intervals to add.
        """
        existing_intervals_set = set(self._interval_to_str(interval) for interval in self.hkl_intervals)
        new_intervals_set = set(self._interval_to_str(interval) for interval in new_hkl_intervals)
        intervals_to_add = [interval for interval in new_hkl_intervals if self._interval_to_str(interval) not in existing_intervals_set]

        self.hkl_intervals.extend(intervals_to_add)
        self.logger.info(f"Added {len(intervals_to_add)} new hkl intervals. Total intervals: {len(self.hkl_intervals)}")

    def save_to_hdf5(self):
        """
        Saves the hkl intervals to an HDF5 file.
        """
        try:
            with h5py.File(self.hdf5_file_path, 'w') as h5file:
                # Save hkl_intervals
                hkl_intervals_grp = h5file.create_group('hkl_intervals')
                for idx, interval in enumerate(self.hkl_intervals):
                    interval_str = self._interval_to_str(interval)
                    hkl_intervals_grp.attrs[str(idx)] = interval_str
            self.logger.info(f"HKL intervals saved to {self.hdf5_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save hkl data to HDF5: {e}")

    def load_from_hdf5(self):
        """
        Loads hkl intervals from an HDF5 file.
        """
        if not os.path.exists(self.hdf5_file_path):
            self.logger.error(f"HDF5 file {self.hdf5_file_path} does not exist.")
            return False
        try:
            with h5py.File(self.hdf5_file_path, 'r') as h5file:
                # Load hkl_intervals
                hkl_intervals_grp = h5file['hkl_intervals']
                self.hkl_intervals = []
                for idx in sorted(hkl_intervals_grp.attrs.keys(), key=int):
                    interval_str = hkl_intervals_grp.attrs[idx]
                    interval = eval(interval_str)
                    self.hkl_intervals.append(interval)
            self.logger.info(f"HKL intervals loaded from {self.hdf5_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load hkl data from HDF5: {e}")
            return False

    def get_previous_max_limit(self):
        """
        Determines the previous maximum hkl limits from the existing intervals.

        Returns:
            tuple: (h_max, k_max, l_max) of the previous maximum limits.
        """
        if not self.hkl_intervals:
            return None

        h_max = max(interval['h_range'][1] for interval in self.hkl_intervals)
        k_max = max(interval['k_range'][1] for interval in self.hkl_intervals)
        l_max = max(interval['l_range'][1] for interval in self.hkl_intervals if interval['l_range'][1] != 0.0)
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

