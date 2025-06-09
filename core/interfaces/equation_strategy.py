# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:53:12 2025

@author: Maksim Eremenko
"""

# # interfaces/equation_mask_strategy.py
import re
import numpy as np
import sympy as sp
from interfaces.mask_strategy import MaskStrategy


class EquationMaskStrategy(MaskStrategy):
    """
    A MaskStrategy implementation that generates a boolean mask based on
    an equation provided as a string. The equation is processed in the
    constructor (via __init__) and converted into a numerical function.
    
    For example, a valid equation string might be:
        "(1 + cos(2*π*h))*(1 + cos(2*π*k))*(1 + cos(2*π*l)) < 0.2"
    """
    
    def __init__(self, parameters: dict):
        # Retrieve the equation string from parameters.
        eq_str = parameters.get("equation_str")
        if eq_str is None:
            raise ValueError("The 'equation_str' parameter is required.")
        
        # Replace Unicode π with "pi".
        eq_str = eq_str.replace("π", "pi")
        # Insert multiplication symbol between "pi" and any letter (e.g., "pi*h").
        eq_str = re.sub(r"pi([a-zA-Z])", r"pi*\1", eq_str)
        print("Processed equation:", eq_str)
        
        # Build an allowed locals dictionary with common functions and constants.
        allowed_locals = {
            'pi': sp.pi,
            'E': sp.E,
            'I': sp.I,
            'oo': sp.oo,
            'infty': sp.oo,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'atan2': sp.atan2,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'asinh': sp.asinh,
            'acosh': sp.acosh,
            'atanh': sp.atanh,
            'exp': sp.exp,
            'log': sp.log,
            'ln': sp.log,
            'sqrt': sp.sqrt,
            'Abs': sp.Abs,
            'abs': sp.Abs,
            'floor': sp.floor,
            'ceiling': sp.ceiling,
            'min': min,
            'max': max,
            'Rational': sp.Rational,
            'factorial': sp.factorial,
            'gamma': sp.gamma,
            'erf': sp.erf,
            'erfc': sp.erfc,
            'Ei': sp.Ei,
        }
        
        # Define the symbolic variables.
        h, k, l = sp.symbols('h k l')
        
        # Parse the equation string into a sympy expression.
        expr = sp.sympify(eq_str, locals=allowed_locals)
        # Convert the sympy expression into a numerical function.
        self.f = sp.lambdify((h, k, l), expr, modules="numpy")
    
    def generate_mask(self, data_points: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Evaluates the precompiled numerical function on the given data points.
        Assumes data_points is an (n, 3) array with columns corresponding to h, k, and l.
        """
        # Extract h, k, and l values.
        h_vals = data_points[:, 0]
        k_vals = data_points[:, 1]
        l_vals = data_points[:, 2]
        # Evaluate the function; it should return a boolean array.
        mask = self.f(h_vals, k_vals, l_vals)
        return np.array(mask, dtype=bool)
    
    def apply_mask(self, mask: np.ndarray, data_array: np.ndarray, parameters: dict) -> np.ndarray:
        """
        Applies the boolean mask to the data array.
        """
        return data_array[mask]