# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:24:56 2024

@author: Maksim Eremenko
"""

# form_factors/form_factor_factory_producer.py

from core.scattering.form_factors.atomic_number_form_factor_factory import (
    AtomicNumberFormFactorFactory,
)
from core.scattering.form_factors.electron_form_factor_factory import (
    ElectronFormFactorFactory,
)
from core.scattering.form_factors.form_factor_factory import FormFactorFactory
from core.scattering.form_factors.neutron_form_factor_factory import (
    NeutronFormFactorFactory,
)
from core.scattering.form_factors.ones_form_factor_factory import (
    OnesFormFactorFactory,
)
from core.scattering.form_factors.xray_form_factor_factory import XRayFormFactorFactory

class FormFactorFactoryProducer:
    _factories = {
        "ones": OnesFormFactorFactory,
        "atomic_number": AtomicNumberFormFactorFactory,
        "neutron": NeutronFormFactorFactory,
        "xray": XRayFormFactorFactory,
        "electron": ElectronFormFactorFactory,
    }

    @staticmethod
    def register_factory(experiment_type: str, factory_cls: type[FormFactorFactory]) -> None:
        FormFactorFactoryProducer._factories[experiment_type.lower()] = factory_cls

    @staticmethod
    def get_factory(experiment_type: str) -> FormFactorFactory:
        """
        Returns the appropriate factory based on the experiment type.

        Args:
            experiment_type (str): The scattering-weight kind
                ('ones', 'atomic_number', 'neutron', 'xray', 'electron').

        Returns:
            FormFactorFactory: The corresponding factory instance.
        """
        try:
            factory_cls = FormFactorFactoryProducer._factories[experiment_type.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported experiment type: {experiment_type}") from exc
        return factory_cls()

