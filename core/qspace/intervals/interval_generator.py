import numpy as np
import logging
from math import ceil

class ReciprocalSpaceIntervalGenerator:
    def __init__(self, supercell, previous_max_limit=None):
        """
        Initializes the ReciprocalSpaceIntervalGenerator.

        Args:
            supercell (tuple/list): The supercell dimensions (e.g., (40, 40, 40) for 3D).
            previous_max_limit (tuple/list, optional): The previous maximum reciprocal_space limits 
                as (h_max, k_max, l_max) or a subset for lower dimensions.
                Defaults to None.
        """
        self.supercell = np.array(supercell)
        self.dim = len(self.supercell)
        self.hkl_grid_step = 1.0 / self.supercell
        self.logger = logging.getLogger(self.__class__.__name__)
        self.previous_max_limit = previous_max_limit

        # For convenience, define axis labels depending on dimension
        self.axes = ['h', 'k', 'l'][:self.dim]

    def generate_intervals(self, hkl_limits, subvolume_step):
        """
        Generates reciprocal space intervals based on limits and subvolume steps.

        Args:
            hkl_limits (list): The maximum values for each axis (e.g., [10.0], [10.0, 10.0], or [10.0, 10.0, 10.0]).
            subvolume_step (list): The steps for subvolumes for each axis (e.g., [5.0], [5.0,5.0], or [5.0,5.0,5.0]).

        Returns:
            list of dicts: Each dict represents an interval with keys like 'h_range', 'k_range', 'l_range' depending on dimension.
        """
        # Extract previous max limits or set to zeros if not provided
        if self.previous_max_limit is not None:
            prev_limits = list(self.previous_max_limit) + [0.0]*(self.dim - len(self.previous_max_limit))
        else:
            prev_limits = [0.0] * self.dim

        intervals = []

        # Generate intervals for each axis
        # For axis h and k (if present), we generate symmetric intervals
        # For axis l (if present, i.e. in 3D), we generate only positive intervals (excluding zero)
        
        # Generate intervals for h (always present)
        h_ranges_all = self._generate_symmetric_intervals('h', hkl_limits[0], subvolume_step[0])
        prev_h_max = prev_limits[0]

        # For 1D, we have only h. For 2D, we have h and k. For 3D, we have h, k, and l.
        if self.dim > 1:
            k_ranges_all = self._generate_symmetric_intervals('k', hkl_limits[1], subvolume_step[1])
            prev_k_max = prev_limits[1]
        else:
            k_ranges_all = [(0.0,0.0)]  # Dummy to simplify logic

        if self.dim > 2:
            l_ranges_all = self._generate_l_intervals(hkl_limits[2], subvolume_step[2], prev_limits[2])
            prev_l_max = prev_limits[2]
        else:
            l_ranges_all = [(0.0,0.0)]  # Dummy to simplify logic

        self.logger.debug(f"All h_ranges: {h_ranges_all}")
        if self.dim > 1:
            self.logger.debug(f"All k_ranges: {k_ranges_all}")
        if self.dim > 2:
            self.logger.debug(f"All l_ranges: {l_ranges_all}")

        # Identify new ranges beyond previous_max_limit
        new_h_ranges = [h for h in h_ranges_all if self._is_new_range(h, prev_h_max)]
        if self.dim > 1:
            new_k_ranges = [k for k in k_ranges_all if self._is_new_range(k, prev_k_max)]
        else:
            new_k_ranges = []
        if self.dim > 2:
            new_l_ranges = [l for l in l_ranges_all if self._is_new_range(l, prev_l_max)]
        else:
            new_l_ranges = []

        self.logger.debug(f"New h_ranges: {new_h_ranges}")
        if self.dim > 1:
            self.logger.debug(f"New k_ranges: {new_k_ranges}")
        if self.dim > 2:
            self.logger.debug(f"New l_ranges: {new_l_ranges}")

        # For dimension-specific interval combination:
        if self.dim == 1:
            # Only h-axis
            # All intervals that are new beyond previous max
            for h_range in new_h_ranges:
                intervals.append({'h_range': h_range})
        elif self.dim == 2:
            # h and k axes
            # Generate shell intervals where at least one of h or k is new
            for h_range in h_ranges_all:
                for k_range in k_ranges_all:
                    if (h_range in new_h_ranges or k_range in new_k_ranges):
                        intervals.append({'h_range': h_range, 'k_range': k_range})
        else:
            # 3D case
            # Original logic for shell intervals
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

        self.logger.info(f"Generated {len(intervals)} reciprocal_space intervals.")
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
        Generates negative intervals for a given axis.

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
            end = -(i * sub_step + grid_step)
            start = -((i + 1) * sub_step)
            if start < -max_value:
                start = -max_value
            intervals.append((np.float64(start), np.float64(end)))
            self.logger.debug(f"{axis}-negative-interval: ({start}, {end})")

        return intervals

    def _generate_positive_intervals(self, axis, max_value, sub_step, n_intervals):
        """
        Generates positive intervals for a given axis.

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
                end = (i + 1) * sub_step 
            if end > max_value:
                end = max_value
            intervals.append((np.float64(start), np.float64(end)))
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
        grid_step = self.hkl_grid_step[2]  # l-axis index is always 2 for 3D
        n_intervals = ceil(l_max / l_sub)

        for i in range(n_intervals):
            if i == 0:
                start = grid_step
                end = l_sub
            else:
                start = i * l_sub + grid_step
                end = (i + 1) * l_sub #+ grid_step
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
            axis (str): Axis label ('h', 'k', 'l').

        Returns:
            int: Index corresponding to the axis.
        """
        return {'h': 0, 'k': 1, 'l': 2}[axis]
