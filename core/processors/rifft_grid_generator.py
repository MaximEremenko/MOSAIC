# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:56:54 2024

@author: Maksim Eremenko
"""

# processors/rifft_grid_generator.py

from abc import ABC, abstractmethod
import numpy as np
import logging

class RIFFTGridGenerator(ABC):
    """
    Abstract base class for grid generators.
    """

    def __init__(self, step_in_frac):
        self.step_in_frac = step_in_frac  # Can be a scalar or an array

    @abstractmethod
    def generate_grid_around_point(self, central_point, dist_from_atom_center):
        """
        Generates a grid around the central point.

        Args:
            central_point (np.ndarray): Coordinates of the central point (shape: (D,))
            dist_from_atom_center (float): Distance from the atom center

        Returns:
            np.ndarray: Array of grid points (shape: (N, D))
        """
        pass

class GridGenerator1D(RIFFTGridGenerator):
    def generate_grid_around_point(self, central_point, dist_from_atom_center):
        if dist_from_atom_center == 0:
            return np.array([central_point])  # Only the central point

        num_steps = int(np.ceil(dist_from_atom_center / self.step_in_frac))
        grid_range = np.arange(-num_steps, num_steps + 1) * self.step_in_frac
        grid_points = grid_range + central_point
        return grid_points.reshape(-1, 1)  # Shape: (N, 1)

class GridGenerator2D(RIFFTGridGenerator):
    def generate_grid_around_point(self, central_point, dist_from_atom_center):
        if dist_from_atom_center == 0:
            return np.array([central_point])

        num_steps = int(np.ceil(dist_from_atom_center / self.step_in_frac))
        grid_range = np.arange(-num_steps, num_steps + 1) * self.step_in_frac
        mesh_x, mesh_y = np.meshgrid(grid_range, grid_range, indexing='ij')
        grid_points = np.vstack([mesh_x.flatten(), mesh_y.flatten()]).T + central_point
        return grid_points  # Shape: (N, 2)

# class GridGenerator3D(RIFFTGridGenerator):
#     def generate_grid_around_point(self, central_point, dist_from_atom_center):
#         if dist_from_atom_center == 0:
#             return np.array([central_point])

#         # Ensure self.step_in_frac is an array of shape (3,)
#         if np.isscalar(self.step_in_frac):
#             step_sizes = np.array([self.step_in_frac] * 3)
#         else:
#             step_sizes = self.step_in_frac

#         num_steps = np.ceil(dist_from_atom_center / step_sizes).astype(int)

#         # Generate grid ranges for each dimension
#         ranges = [np.arange(-n, n + 1) * s for n, s in zip(num_steps, step_sizes)]
#         mesh = np.meshgrid(*ranges, indexing='ij')
#         grid_points = np.vstack([m.flatten() for m in mesh]).T + central_point
#         return grid_points

class GridGenerator3D(RIFFTGridGenerator):
    def __init__(self, step_in_frac):
        """
        Initializes the GridGenerator3D with step sizes in angstroms for each dimension.

        Args:
            step_in_frac (float or array-like): Step sizes for x, y, z axes. 
                                                    If float, same step size is used for all axes.
                                                    If array-like, must have three elements.
        """
        super().__init__(step_in_frac)  # Initialize the base class
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_grid_around_point(self, central_point, dist_from_atom_center):
        """
        Generates a 3D grid around the central point based on distances and step sizes.

        Args:
            central_point (array-like): Coordinates of the central point (x, y, z).
            dist_from_atom_center (array-like): Distances from the central point along each axis (dx, dy, dz).

        Returns:
            np.ndarray: Array of grid points with shape (N, 3).
        """
        self.logger.debug(f"Generating 3D grid around point {central_point} with distances {dist_from_atom_center}")

        # Ensure dist_from_atom_center is an array-like with three elements
        if not isinstance(dist_from_atom_center, (np.ndarray, list, tuple)):
            self.logger.error(f"dist_from_atom_center must be an array-like of three elements, got {type(dist_from_atom_center)}")
            raise ValueError("dist_from_atom_center must be an array-like of three elements")

        dist_from_atom_center = np.array(dist_from_atom_center, dtype=float)
        if dist_from_atom_center.shape != (3,):
            self.logger.error(f"dist_from_atom_center must have shape (3,), got {dist_from_atom_center.shape}")
            raise ValueError("dist_from_atom_center must have shape (3,)")

        # Check if all distances are zero
        if np.all(dist_from_atom_center == 0):
            self.logger.debug("All distances are zero, returning central point only.")
            return np.array([central_point], dtype=float)

        # Ensure self.step_in_angstroms is an array of shape (3,)
        if np.isscalar(self.step_in_frac):
            step_sizes = np.array([self.step_in_frac] * 3, dtype=float)
            self.logger.debug(f"Using uniform step sizes: {step_sizes}")
        else:
            step_sizes = np.array(self.step_in_frac, dtype=float)
            if step_sizes.shape != (3,):
                self.logger.error(f"step_in_frac must be a scalar or array of shape (3,), got shape {step_sizes.shape}")
                raise ValueError("step_in_frac must be a scalar or an array of shape (3,)")
            self.logger.debug(f"Using per-dimension step sizes: {step_sizes}")

        # Initialize lists to store grid sizes
        x_space_grid_size = []
        y_space_grid_size = []
        z_space_grid_size = []

        # Generate grid ranges for each dimension using the provided approach
        # Define a small epsilon to prevent floating point errors
        epsilon = 1e-12

        # Generate grid ranges for each dimension using the provided approach with epsilon adjustment
        x_space_grid = np.arange(-dist_from_atom_center[0],
                                 dist_from_atom_center[0] + step_sizes[0] - epsilon,
                                 step_sizes[0])
        y_space_grid = np.arange(-dist_from_atom_center[1],
                                 dist_from_atom_center[1] + step_sizes[1] - epsilon,
                                 step_sizes[1])
        z_space_grid = np.arange(-dist_from_atom_center[2],
                                 dist_from_atom_center[2] + step_sizes[2] - epsilon,
                                 step_sizes[2])

        # Append grid sizes
        x_space_grid_size.append(x_space_grid.size)
        y_space_grid_size.append(y_space_grid.size)
        z_space_grid_size.append(z_space_grid.size)

        self.logger.debug(f"x_space_grid: {x_space_grid}")
        self.logger.debug(f"y_space_grid: {y_space_grid}")
        self.logger.debug(f"z_space_grid: {z_space_grid}")
        self.logger.debug(f"x_space_grid_size: {x_space_grid_size}")
        self.logger.debug(f"y_space_grid_size: {y_space_grid_size}")
        self.logger.debug(f"z_space_grid_size: {z_space_grid_size}")

        # Create meshgrid
        mesh = np.meshgrid(x_space_grid, y_space_grid, z_space_grid, indexing='ij')
        grid_points = np.vstack([m.flatten() for m in mesh]).T + central_point

        self.logger.debug(f"Generated grid points shape: {grid_points.shape}")
        self.logger.debug(f"First few grid points:\n{grid_points[:5]}")  # Show first 5 grid points for verification

        return grid_points  # Shape: (N, 3)