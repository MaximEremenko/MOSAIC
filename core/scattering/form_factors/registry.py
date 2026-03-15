from __future__ import annotations

from core.scattering.form_factors.form_factor_factory_producer import (
    FormFactorFactoryProducer,
)
from .contracts import FormFactorSelection


class FormFactorRegistry:
    def create_calculator(self, selection: FormFactorSelection):
        return FormFactorFactoryProducer.get_factory(selection.family).create_calculator(
            selection.calculator
        )
