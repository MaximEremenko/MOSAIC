# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:50:03 2024

@author: Maksim Eremenko
"""

# strategies/mask_strategies.py

import numpy as np
from interfaces.mask_strategy import IMaskStrategy
import sympy as sp
from utilities.logic_parser import parse_logic, preprocess, allowed_locals, symbol_map
import pandas as pd
#from interfaces.shape_strategy import ShapeStrategy

class DefaultMaskStrategy(IMaskStrategy):
    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Returns a mask of all True values (no masking).

        Args:
            hkl_mesh (np.ndarray): An array of hkl points.

        Returns:
            np.ndarray: A boolean array of all True values.
        """
        return np.ones(hkl_mesh.shape[0], dtype=bool)
    

class CustomReciprocalSpacePointsStrategy(IMaskStrategy):
    def __init__(self, file_path: str, ih: np.ndarray, ik: np.ndarray, il: np.ndarray):
        """
        Initializes the strategy with the file containing custom reciprocal space points.

        Args:
            file_path (str): Path to the file with custom reciprocal space points.
            ih (np.ndarray): h indices array.
            ik (np.ndarray): k indices array.
            il (np.ndarray): l indices array.
        """
        self.file_path = file_path
        self.ih = ih
        self.ik = ik
        self.il = il

    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Generates a mask based on custom reciprocal space points from a file.

        Args:
            hkl_mesh (np.ndarray): An array of hkl points.

        Returns:
            np.ndarray: A boolean array representing the mask.
        """
        # Read hkl reflections from file
        df_hkl_reflections = pd.read_table(
            self.file_path, skiprows=0, sep='\\s+', engine='python'
        )
        hkl_reflections = df_hkl_reflections.values

        # Filter reflections within ih, ik, il ranges
        h_min, h_max = self.ih.min() - 0.5, self.ih.max() + 0.5
        k_min, k_max = self.ik.min() - 0.5, self.ik.max() + 0.5
        l_min, l_max = self.il.min() - 0.5, self.il.max() + 0.5

        mask = (
            (hkl_reflections[:, 0] >= h_min) & (hkl_reflections[:, 0] <= h_max) &
            (hkl_reflections[:, 1] >= k_min) & (hkl_reflections[:, 1] <= k_max) &
            (hkl_reflections[:, 2] >= l_min) & (hkl_reflections[:, 2] <= l_max)
        )
        hkl_reflections_in_lim = hkl_reflections[mask]

        # Create mask for hkl_mesh
        mask = self._create_mask_from_reflections(hkl_mesh, hkl_reflections_in_lim)
        return mask

    def _create_mask_from_reflections(self, hkl_mesh: np.ndarray, reflections: np.ndarray) -> np.ndarray:
        # Use structured arrays for comparison
        hkl_dtype = np.dtype([('h', hkl_mesh.dtype), ('k', hkl_mesh.dtype), ('l', hkl_mesh.dtype)])
        hkl_mesh_structured = hkl_mesh.view(hkl_dtype).reshape(-1)
        reflections_structured = reflections.view(hkl_dtype).reshape(-1)
        mask = np.in1d(hkl_mesh_structured, reflections_structured)
        return mask
    
# from mantid.geometry import SpaceGroupFactory, PointGroupFactory
# class SpaceGroupSymmetryStrategy(IMaskStrategy):
#     def __init__(self, space_group_symmetry: str, ih: np.ndarray, ik: np.ndarray, il: np.ndarray):
#         """
#         Initializes the strategy with space group symmetry information.

#         Args:
#             space_group_symmetry (str): Space group symbol.
#             ih (np.ndarray): h indices array.
#             ik (np.ndarray): k indices array.
#             il (np.ndarray): l indices array.
#         """
#         self.space_group_symmetry = space_group_symmetry
#         self.ih = ih
#         self.ik = ik
#         self.il = il

#     def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
#         """
#         Generates a mask based on space group symmetry.

#         Args:
#             hkl_mesh (np.ndarray): An array of hkl points.

#         Returns:
#             np.ndarray: A boolean array representing the mask.
#         """
#         # Generate central points (assuming origin)
#         central_point = np.array([0, 0, 0])
#         acpx = self._find_central_points(central_point[0], self.ih)
#         acpy = self._find_central_points(central_point[1], self.ik)
#         acpz = self._find_central_points(central_point[2], self.il)

#         # Generate all combinations of central points
#         hkl_reflections = self._generate_hkl_reflections(acpx, acpy, acpz)

#         # Filter reflections using space group symmetry
#         allowed_reflections = self._filter_allowed_reflections(hkl_reflections)

#         # Create mask for hkl_mesh
#         mask = self._create_mask_from_reflections(hkl_mesh, allowed_reflections)
#         return mask

#     def _find_central_points(self, coordinate: float, indices: np.ndarray) -> np.ndarray:
#         min_index, max_index = indices.min() - 0.5, indices.max() + 0.5
#         # Implement logic to find central points within the indices range
#         return np.array([coordinate])  # Placeholder

#     def _generate_hkl_reflections(self, acpx: np.ndarray, acpy: np.ndarray, acpz: np.ndarray) -> np.ndarray:
#         hkl_reflections = np.array(np.meshgrid(acpx, acpy, acpz)).T.reshape(-1, 3)
#         return hkl_reflections

#     def _filter_allowed_reflections(self, hkl_reflections: np.ndarray) -> np.ndarray:
#         sg_object = SpaceGroupFactory.createSpaceGroup(self.space_group_symmetry)
#         is_allowed = [sg_object.isAllowedReflection(hkl) for hkl in hkl_reflections]
#         allowed_reflections = hkl_reflections[is_allowed]
#         return allowed_reflections

#     def _create_mask_from_reflections(self, hkl_mesh: np.ndarray, reflections: np.ndarray) -> np.ndarray:
#         hkl_dtype = np.dtype([('h', hkl_mesh.dtype), ('k', hkl_mesh.dtype), ('l', hkl_mesh.dtype)])
#         hkl_mesh_structured = hkl_mesh.view(hkl_dtype).reshape(-1)
#         reflections_structured = reflections.view(hkl_dtype).reshape(-1)
#         mask = np.in1d(hkl_mesh_structured, reflections_structured)
#         return mask



class EqBasedStrategy(IMaskStrategy):
    """
    Strategy that masks hkl points according to an arbitrary logical equation.
    
    Example:
        condition = (
            "(cos(pi*h) + cos(pi*k) + cos(pi*l) > -0.5 and "
            "cos(pi*h) + cos(pi*k) + cos(pi*l) < 0.5) and "
            "sqrt((acos(-cos(2*pi*h))/(2*pi))**2 + … ) >= 0.2"
        )
        strat = EqBasedStrategy(condition)
        mask = strat.generate_mask(hkl_mesh)
    """
    def __init__(self, condition: str):
        """
        Args:
            condition: A Sympy‐friendly logical string in terms of h, k, l.
        """
        # 1) Preprocess the raw string (π → pi, insert *, convert ^ to **, etc.)
        cond_str = preprocess(condition)
        
        # 2) Parse into a single Sympy Boolean expression
        #    symbol_map is {'h': h, 'k': k, 'l': l}
        expr: sp.Boolean = parse_logic(cond_str, symbol_map, allowed_locals)
        
        # 3) Lambdify to get a fast NumPy‐vectorized function f(h,k,l)->bool
        h, k, l = symbol_map['h'], symbol_map['k'], symbol_map['l']
        self._f = sp.lambdify((h, k, l), expr, modules="numpy")
    
    def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
        """
        Evaluate the logical condition at each point in hkl_mesh.

        Args:
            hkl_mesh: np.ndarray of shape (N, 3), columns = h, k, l.

        Returns:
            mask: np.ndarray of shape (N,), dtype=bool.
        """
        h_vals = hkl_mesh[:, 0]
        k_vals = hkl_mesh[:, 1]
        l_vals = hkl_mesh[:, 2]

        # evaluate and ensure boolean dtype
        mask = self._f(h_vals, k_vals, l_vals)
        return mask.astype(bool)



# class CoordinateBasedStrategy(IMaskStrategy):
#     def __init__(
#         self,
#         coordinate: np.ndarray,
#         ih: np.ndarray,
#         ik: np.ndarray,
#         il: np.ndarray,
#         shape_strategy: ShapeStrategy,
#         symmetry_applier=None  # Optional symmetry applier
#     ):
#         """
#         Initializes the strategy with coordinates and shape strategy.

#         Args:
#             coordinate (np.ndarray): Central coordinate.
#             ih (np.ndarray): h indices array.
#             ik (np.ndarray): k indices array.
#             il (np.ndarray): l indices array.
#             shape_strategy (ShapeStrategy): Shape strategy for masking.
#             symmetry_applier (optional): Symmetry applier for point group symmetry.
#         """
#         self.coordinate = coordinate
#         self.ih = ih
#         self.ik = ik
#         self.il = il
#         self.shape_strategy = shape_strategy
#         self.symmetry_applier = symmetry_applier

#     def generate_mask(self, hkl_mesh: np.ndarray) -> np.ndarray:
#         """
#         Generates a mask based on coordinates and shape.

#         Args:
#             hkl_mesh (np.ndarray): An array of hkl points.

#         Returns:
#             np.ndarray: A boolean array representing the mask.
#         """
#         # Generate central points
#         acpx = self._find_central_points(self.coordinate[0], self.ih)
#         acpy = self._find_central_points(self.coordinate[1], self.ik)
#         acpz = self._find_central_points(self.coordinate[2], self.il)

#         central_points = self._generate_central_points(acpx, acpy, acpz)

#         # Apply shape mask
#         mask = self.shape_strategy.apply(hkl_mesh, central_points)

#         # Apply symmetry if specified
#         if self.symmetry_applier:
#             symmetry_mask = self.symmetry_applier.apply(hkl_mesh)
#             mask &= symmetry_mask

#         return mask

#     def _find_central_points(self, coordinate: float, indices: np.ndarray) -> np.ndarray:
#         min_index, max_index = indices.min() - 0.5, indices.max() + 0.5
#         # Implement logic to find central points within the indices range
#         return np.array([coordinate])  # Placeholder

#     def _generate_central_points(self, acpx: np.ndarray, acpy: np.ndarray, acpz: np.ndarray) -> np.ndarray:
#         central_points = np.array(np.meshgrid(acpx, acpy, acpz)).T.reshape(-1, 3)
#         return central_points
    
