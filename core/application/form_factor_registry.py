from __future__ import annotations

from core.domain.models import FormFactorSelection
from core.form_factors.form_factor_factory_producer import FormFactorFactoryProducer


class FormFactorRegistry:
    def create_calculator(self, selection: FormFactorSelection):
        return FormFactorFactoryProducer.get_factory(selection.family).create_calculator(
            selection.calculator
        )
