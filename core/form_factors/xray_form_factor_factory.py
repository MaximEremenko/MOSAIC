# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:24:43 2024

@author: Maksim Eremenko
"""

# form_factors/xray_form_factor_factory.py

from form_factors.form_factor_factory import FormFactorFactory
from form_factors.xray_form_factor_calculator import XRayFormFactorCalculator
from form_factors.default_form_factor_calculator import DefaultFormFactorCalculator
from form_factors.parameter_provider import ParameterProvider
from form_factors.equation_strategy import EquationStrategy
from form_factors.xray_doyle_turner_equation import XRayDoyleTurnerEquation
from form_factors.xray_lobato_equation import XRayLobatoEquation
from form_factors.form_factor_calculator import FormFactorCalculator
from form_factors.json_parameter_provider import JSONParameterProvider

class XRayFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = 'default', **kwargs) -> FormFactorCalculator:
        if method == 'default':
            return DefaultFormFactorCalculator()
        else:
            equation_strategy = self._get_equation_strategy(method)
            parameter_provider = kwargs.get('parameter_provider')
            if parameter_provider is None:
                parameter_provider = self._default_parameter_provider(method)
            return XRayFormFactorCalculator(equation_strategy, parameter_provider)

    def _get_equation_strategy(self, method: str) -> EquationStrategy:
        if method == 'doyle_turner':
            return XRayDoyleTurnerEquation()
        elif method == 'lobato':
            return XRayLobatoEquation()
        else:
            raise ValueError(f"Unsupported X-ray method '{method}'.")

    def _default_parameter_provider(self, method: str) -> ParameterProvider:
        filename = f'xray_params_{method}.json'
        return JSONParameterProvider(filename)
