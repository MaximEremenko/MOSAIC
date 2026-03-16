from __future__ import annotations

from core.scattering.form_factors.form_factor_factory_producer import (
    FormFactorFactoryProducer,
)
from .contracts import ScatteringWeightSelection


class ScatteringWeightRegistry:
    def create_calculator(self, selection: ScatteringWeightSelection):
        return FormFactorFactoryProducer.get_factory(selection.kind).create_calculator(
            selection.calculator
        )

__all__ = ["ScatteringWeightRegistry"]
