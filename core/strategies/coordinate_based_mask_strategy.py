# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:12:32 2024

@author: Maksim Eremenko
"""
# strategies/coordinate_based_mask_strategy.py
from interfaces.shape_strategy import ShapeStrategy
from interfaces.mask_strategy import MaskStrategy
import numpy as np
from symmetry.symmetry_applier import SymmetryApplier
from staretegies.shape_strategies import SphereShapeStrategy, EllipsoidShapeStrategy

class CoordinateBasedMaskStrategy(MaskStrategy):
    def generate_mask(self, data_points: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Generates a boolean mask based on multiple shapes and optional symmetry.

        Args:
            data_points (np.ndarray): An array of data points.
            parameters (dict): Includes 'shapes', 'negative'.

        Returns:
            np.ndarray: Boolean mask array.
        """
        shapes = parameters.get('shapes', [])
        negative = parameters.get('negative', False)
        mask = np.zeros(len(data_points), dtype=bool)

        # Group shapes by shape type and parameters for efficiency
        shape_groups = self._group_shapes(shapes)

        for shape_key, group_shapes in shape_groups.items():
            shape_type, shape_params = shape_key
            central_points = np.array([shape['coordinate'] for shape in group_shapes])

            # Get the appropriate shape strategy
            shape_strategy = self._get_shape_strategy(shape_params)

            # Generate mask for the group
            shape_mask = shape_strategy.apply(data_points, central_points)

            # Apply symmetry if specified (assuming same symmetry for group)
            symmetry_params = group_shapes[0].get('symmetry_parameters', None)
            if symmetry_params:
                symmetry_applier = SymmetryApplier(symmetry_params)
                symmetry_mask = symmetry_applier.apply(data_points)
                shape_mask &= symmetry_mask

            mask |= shape_mask

        if negative:
            mask = ~mask

        return mask

    def _group_shapes(self, shapes):
        """
        Groups shapes by type and parameters.

        Args:
            shapes (list): List of shape dictionaries.

        Returns:
            dict: Grouped shapes.
        """
        shape_groups = {}
        for shape in shapes:
            shape_params = shape.get('shape_parameters', {})
            shape_type = shape_params.get('shape', 'sphere')
            # Create a hashable key from shape type and sorted parameters
            shape_key = (shape_type, tuple(sorted(shape_params.items())))
            if shape_key not in shape_groups:
                shape_groups[shape_key] = []
            shape_groups[shape_key].append(shape)
        return shape_groups

    def _get_shape_strategy(self, shape_params: dict) -> ShapeStrategy:
        """
        Determines the appropriate shape strategy based on parameters.

        Args:
            shape_params (dict): Shape parameters.

        Returns:
            ShapeStrategy: An instance of a shape strategy.
        """
        shape_type = shape_params.get('shape', 'sphere')
        if shape_type == 'sphere':
            radius = shape_params.get('radius', 1.0)
            return SphereShapeStrategy(radius)
        elif shape_type == 'ellipsoid':
            axes = np.array(shape_params.get('axes', [1.0, 1.0, 1.0]))
            theta = shape_params.get('theta', 0.0)
            phi = shape_params.get('phi', 0.0)
            return EllipsoidShapeStrategy(axes, theta, phi)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")
