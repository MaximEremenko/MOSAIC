# -*- coding: utf-8 -*-
"""Unity/ones scattering-weight factory."""

from core.scattering.form_factors.default_form_factor_calculator import (
    DefaultFormFactorCalculator,
)
from core.scattering.form_factors.form_factor_calculator import FormFactorCalculator
from core.scattering.form_factors.form_factor_factory import FormFactorFactory


class OnesFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = "default", **kwargs) -> FormFactorCalculator:
        if str(method).strip().lower() != "default":
            raise ValueError(
                "The 'ones' scattering-weight kind only supports calculator='default'."
            )
        return DefaultFormFactorCalculator()
