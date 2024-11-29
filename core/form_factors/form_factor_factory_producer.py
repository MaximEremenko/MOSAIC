# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:24:56 2024

@author: Maksim Eremenko
"""

# form_factors/form_factor_factory_producer.py

from form_factors.form_factor_factory import FormFactorFactory
from form_factors.neutron_form_factor_factory import NeutronFormFactorFactory
from form_factors.xray_form_factor_factory import XRayFormFactorFactory
from form_factors.electron_form_factor_factory import ElectronFormFactorFactory

class FormFactorFactoryProducer:
    @staticmethod
    def get_factory(experiment_type: str) -> FormFactorFactory:
        """
        Returns the appropriate factory based on the experiment type.

        Args:
            experiment_type (str): The type of experiment ('neutron', 'xray', 'electron').

        Returns:
            FormFactorFactory: The corresponding factory instance.
        """
        if experiment_type.lower() == 'neutron':
            return NeutronFormFactorFactory()
        elif experiment_type.lower() == 'xray':
            return XRayFormFactorFactory()
        elif experiment_type.lower() == 'electron':
            return ElectronFormFactorFactory()
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

