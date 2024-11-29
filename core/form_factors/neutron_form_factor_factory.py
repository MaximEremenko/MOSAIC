# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:23:38 2024

@author: Maksim Eremenko
"""

# form_factors/neutron_form_factor_factory.py
from form_factors.form_factor_factory import  FormFactorFactory


class NeutronFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = 'default', **kwargs) -> FormFactorCalculator:
        if method == 'default':
            return DefaultFormFactorCalculator()
        else:
            parameter_provider = kwargs.get('parameter_provider')
            if parameter_provider is None:
                parameter_provider = self._default_parameter_provider()
            return NeutronFormFactorCalculator(parameter_provider)

    def _default_parameter_provider(self) -> ParameterProvider:
        return None
