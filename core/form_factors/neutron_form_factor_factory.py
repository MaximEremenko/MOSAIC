# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:23:38 2024

@author: Maksim Eremenko
"""

# form_factors/neutron_form_factor_factory.py
from form_factors.form_factor_factory import  FormFactorFactory
from form_factors.form_factor_factory import FormFactorFactory
from form_factors.json_parameter_provider import JSONParameterProvider
from form_factors.neutron_form_factor_calculator import NeutronFormFactorCalculator
from form_factors.default_form_factor_calculator import DefaultFormFactorCalculator
from form_factors.parameter_provider import ParameterProvider
from form_factors.form_factor_calculator import FormFactorCalculator
from utilities.rmc_neutron_scl import rmc_neutron_scl_


# Default Parameter Provider Using Hardcoded Values
class NeutronDefaultParameterProvider(ParameterProvider):
    def get_parameters(self, element: str, charge: int = 0) -> dict:
        fca, _ = rmc_neutron_scl_(element)
        return fca


class NeutronFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = 'default', **kwargs) -> FormFactorCalculator:
        if method == 'default':
            parameter_provider = self._default_parameter_provider()
        elif method == 'custom':
            # Use the custom JSON parameter provider
            filename = kwargs.get('filename', 'parameters.json')
            parameter_provider = JSONParameterProvider(filename)
        else:
            raise ValueError(f"Unknown method: {method}")
        return NeutronFormFactorCalculator(parameter_provider)


    def _default_parameter_provider(self) -> ParameterProvider:
        return NeutronDefaultParameterProvider()
