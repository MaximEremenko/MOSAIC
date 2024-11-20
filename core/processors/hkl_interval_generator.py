import numpy as np
import logging
from math import ceil

class HKLIntervalGenerator:
    def __init__(self, supercell, previous_max_limit=None):
        """
        Initializes the HKLIntervalGenerator.

        Args:
            supercell (tuple/list): The supercell dimensions (e.g., (40, 40, 40)).
            previous_max_limit (tuple/list, optional): The previous maximum hkl limits 
                as (h_max, k_max, l_max). Defaults to None.
        """
        self.supercell = np.array(supercell)
        self.hkl_grid_step = 1.0 / self.supercell  # e.g., 0.025 for supercell=40
        self.logger = logging.getLogger(self.__class__.__name__)
        self.previous_max_limit = previous_max_limit

    def generate_intervals(self, hkl_limits, subvolume_step):
        """
        Generates hkl intervals based on limits and subvolume steps.

        Args:
            hkl_limits (list): The maximum h, k, l values (e.g., [10.0, 10.0, 10.0]).
            subvolume_step (list): The steps for subvolumes (e.g., [5.0, 5.0, 5.0]).

        Returns:
            list of dicts: Each dict represents an hkl interval with 'h_range', 'k_range', 'l_range'.
        """
        h_max, k_max, l_max = hkl_limits
        h_sub, k_sub, l_sub = subvolume_step

        # Extract previous max limits or set to 0.0 if not provided
        if self.previous_max_limit:
            previous_h_max, previous_k_max, previous_l_max = self.previous_max_limit
        else:
            previous_h_max, previous_k_max, previous_l_max = (0.0, 0.0, 0.0)

        intervals = []

        # Generate all h and k ranges up to current h_max and k_max
        h_ranges_all = self._generate_symmetric_intervals(axis='h', max_value=h_max, sub_step=h_sub)
        k_ranges_all = self._generate_symmetric_intervals(axis='k', max_value=k_max, sub_step=k_sub)

        # Generate all l ranges up to current l_max
        l_ranges_all = self._generate_l_intervals(l_max, l_sub, previous_l_max)

        self.logger.debug(f"All h_ranges: {h_ranges_all}")
        self.logger.debug(f"All k_ranges: {k_ranges_all}")
        self.logger.debug(f"All l_ranges: {l_ranges_all}")

        # Identify new ranges beyond previous_max_limit
        new_h_ranges = [h for h in h_ranges_all if self._is_new_range(h, previous_h_max)]
        new_k_ranges = [k for k in k_ranges_all if self._is_new_range(k, previous_k_max)]
        new_l_ranges = [l for l in l_ranges_all if self._is_new_range(l, previous_l_max)]

        self.logger.debug(f"New h_ranges: {new_h_ranges}")
        self.logger.debug(f"New k_ranges: {new_k_ranges}")
        self.logger.debug(f"New l_ranges: {new_l_ranges}")

        # Generate shell intervals where at least one of h, k, l is in new ranges
        for l_range in l_ranges_all:
            for h_range in h_ranges_all:
                for k_range in k_ranges_all:
                    if (h_range in new_h_ranges or
                        k_range in new_k_ranges or
                        l_range in new_l_ranges):
                        interval = {
                            'h_range': h_range,
                            'k_range': k_range,
                            'l_range': l_range
                        }
                        intervals.append(interval)
                        self.logger.debug(f"Added shell interval: {interval}")

        # Separate handling for l=0.0 intervals
        # Only add l=0.0 intervals for new h and/or k ranges to prevent duplicates
        l_zero_interval = (0.0, 0.0)
        for h_range in new_h_ranges:
            for k_range in k_ranges_all:
                interval = {
                    'h_range': h_range,
                    'k_range': k_range,
                    'l_range': l_zero_interval
                }
                intervals.append(interval)
                self.logger.debug(f"Added l=0.0 interval for new h_range: {interval}")

        for k_range in new_k_ranges:
            for h_range in h_ranges_all:
                # To avoid adding the same interval twice when both h and k are new
                if h_range not in new_h_ranges:
                    interval = {
                        'h_range': h_range,
                        'k_range': k_range,
                        'l_range': l_zero_interval
                    }
                    intervals.append(interval)
                    self.logger.debug(f"Added l=0.0 interval for new k_range: {interval}")

        self.logger.info(f"Generated {len(intervals)} hkl intervals.")
        return intervals

    def _generate_symmetric_intervals(self, axis, max_value, sub_step):
        """
        Generates symmetric intervals (negative and positive) for a given axis.

        Args:
            axis (str): Axis label ('h' or 'k').
            max_value (float): Maximum value for the axis.
            sub_step (float): Step size for the subvolume.

        Returns:
            list of tuples: Each tuple represents an interval (start, end).
        """
        intervals = []
        n_intervals = ceil(max_value / sub_step)

        # Generate negative intervals
        negative_intervals = self._generate_negative_intervals(axis, max_value, sub_step, n_intervals)
        intervals.extend(negative_intervals)

        # Generate positive intervals
        positive_intervals = self._generate_positive_intervals(axis, max_value, sub_step, n_intervals)
        intervals.extend(positive_intervals)

        return intervals

    def _generate_negative_intervals(self, axis, max_value, sub_step, n_intervals):
        """
        Generates negative intervals for a given axis beyond previous_max.

        Args:
            axis (str): Axis label ('h' or 'k').
            max_value (float): Maximum value for the axis.
            sub_step (float): Step size for the subvolume.
            n_intervals (int): Number of subvolumes.

        Returns:
            list of tuples: Each tuple represents a negative interval (start, end).
        """
        intervals = []
        grid_step = self.hkl_grid_step[self._axis_index(axis)]

        for i in range(n_intervals):
            # Calculate start and end dynamically
            end = -(i * sub_step + grid_step)
            start = -((i + 1) * sub_step)
            # Ensure that we do not go beyond -max_value
            if start < -max_value:
                start = -max_value
            intervals.append((start, end))
            self.logger.debug(f"{axis}-negative-interval: ({start}, {end})")

        return intervals

    def _generate_positive_intervals(self, axis, max_value, sub_step, n_intervals):
        """
        Generates positive intervals for a given axis beyond previous_max.

        Args:
            axis (str): Axis label ('h' or 'k').
            max_value (float): Maximum value for the axis.
            sub_step (float): Step size for the subvolume.
            n_intervals (int): Number of subvolumes.

        Returns:
            list of tuples: Each tuple represents a positive interval (start, end).
        """
        intervals = []
        grid_step = self.hkl_grid_step[self._axis_index(axis)]

        for i in range(n_intervals):
            if i == 0:
                start = 0.0
                end = sub_step
            else:
                start = i * sub_step + grid_step
                end = (i + 1) * sub_step + grid_step
            # Ensure that we do not exceed max_value
            if end > max_value:
                end = max_value
            intervals.append((start, end))
            self.logger.debug(f"{axis}-positive-interval: ({start}, {end})")

        return intervals

    def _generate_l_intervals(self, l_max, l_sub, previous_max):
        """
        Generates l intervals excluding l=0 beyond previous_max.

        Args:
            l_max (float): Maximum l value.
            l_sub (float): Subvolume step for l.
            previous_max (float): Previous maximum l value.

        Returns:
            list of tuples: Each tuple represents an l interval (start, end).
        """
        intervals = []
        grid_step = self.hkl_grid_step[2]  # l-axis index is 2
        n_intervals = ceil(l_max / l_sub)

        for i in range(n_intervals):
            if i == 0:
                start = grid_step  # 0.025
                end = l_sub
            else:
                start = i * l_sub + grid_step
                end = (i + 1) * l_sub + grid_step
            # Ensure that we do not exceed l_max
            if end > l_max:
                end = l_max
            intervals.append((start, end))
            self.logger.debug(f"l-positive-interval: ({start}, {end})")

        return intervals

    def _is_new_range(self, range_tuple, previous_max):
        """
        Determines if a given range is new based on previous_max.

        Args:
            range_tuple (tuple): The range as (start, end).
            previous_max (float): The previous maximum value for the axis.

        Returns:
            bool: True if the range is new, False otherwise.
        """
        return abs(range_tuple[1]) > previous_max

    def _axis_index(self, axis):
        """
        Returns the index of the axis.

        Args:
            axis (str): Axis label ('h', 'k', or 'l').

        Returns:
            int: Index corresponding to the axis.
        """
        return {'h': 0, 'k': 1, 'l': 2}[axis]
