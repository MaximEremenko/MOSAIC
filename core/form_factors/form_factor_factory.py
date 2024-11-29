# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:23:02 2024

@author: Maksim Eremenko
"""

# form_factors/form_factor_factory.py
from form_factors.form_factor_calculator import  FormFactorCalculator

class FormFactorFactory:
    def create_calculator(self, method: str, **kwargs) -> FormFactorCalculator:
        pass
