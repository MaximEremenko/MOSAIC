# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:24:49 2024

@author: Maksim Eremenko
"""

# form_factors/electron_form_factor_factory.py

from form_factors.form_factor_factory import FormFactorFactory
from form_factors.electron_form_factor_calculator import ElectronFormFactorCalculator
from form_factors.default_form_factor_calculator import DefaultFormFactorCalculator
from form_factors.parameter_provider import ParameterProvider
from form_factors.equation_strategy import EquationStrategy
from form_factors.electron_lobato_equation import ElectronLobatoEquation
from form_factors.electron_peng_equation import ElectronPengEquation
from form_factors.form_factor_calculator import FormFactorCalculator
from form_factors.json_parameter_provider import JSONParameterProvider

class ElectronFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = 'default', **kwargs) -> FormFactorCalculator:
        if method == 'default':
            return DefaultFormFactorCalculator()
        else:
            equation_strategy = self._get_equation_strategy(method)
            parameter_provider = kwargs.get('parameter_provider')
            if parameter_provider is None:
                parameter_provider = self._default_parameter_provider(method)
            return ElectronFormFactorCalculator(equation_strategy, parameter_provider)

    def _get_equation_strategy(self, method: str) -> EquationStrategy:
        if method == 'peng':
            return ElectronPengEquation()
        elif method == 'lobato':
            return ElectronLobatoEquation()
        else:
            raise ValueError(f"Unsupported electron method '{method}'.")

    def _default_parameter_provider(self, method: str) -> ParameterProvider:
        filename = f'electron_params_{method}.json'
        return JSONParameterProvider(filename)

