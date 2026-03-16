# -*- coding: utf-8 -*-
"""Atomic-number scattering-weight factory."""

from __future__ import annotations

import numpy as np

from core.scattering.form_factors.form_factor_calculator import FormFactorCalculator
from core.scattering.form_factors.form_factor_factory import FormFactorFactory

_ATOMIC_SYMBOLS = (
    "h", "he",
    "li", "be", "b", "c", "n", "o", "f", "ne",
    "na", "mg", "al", "si", "p", "s", "cl", "ar",
    "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
    "ga", "ge", "as", "se", "br", "kr",
    "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd",
    "in", "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy",
    "ho", "er", "tm", "yb", "lu",
    "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi",
    "po", "at", "rn",
    "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf",
    "es", "fm", "md", "no", "lr",
    "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn", "nh", "fl", "mc",
    "lv", "ts", "og",
)
_ATOMIC_NUMBER_BY_SYMBOL = {symbol: index + 1 for index, symbol in enumerate(_ATOMIC_SYMBOLS)}
_ATOMIC_NUMBER_BY_SYMBOL.update({"d": 1, "7l": 3, "va": 0})


def _normalize_element_symbol(element: str) -> str:
    return "".join(ch for ch in str(element).strip().lower() if ch.isalnum())


class AtomicNumberFormFactorCalculator(FormFactorCalculator):
    def calculate(
        self,
        reciprocal_space_coordinates: np.ndarray,
        element: str,
        charge: int = 0,
    ) -> np.ndarray:
        symbol = _normalize_element_symbol(element)
        if symbol not in _ATOMIC_NUMBER_BY_SYMBOL:
            raise ValueError(
                f"Unknown element symbol {element!r} for scattering-weight kind 'atomic_number'."
            )
        n_coordinates = reciprocal_space_coordinates.shape[0]
        return np.full(n_coordinates, float(_ATOMIC_NUMBER_BY_SYMBOL[symbol]))


class AtomicNumberFormFactorFactory(FormFactorFactory):
    def create_calculator(self, method: str = "default", **kwargs) -> FormFactorCalculator:
        if str(method).strip().lower() != "default":
            raise ValueError(
                "The 'atomic_number' scattering-weight kind only supports calculator='default'."
            )
        return AtomicNumberFormFactorCalculator()
