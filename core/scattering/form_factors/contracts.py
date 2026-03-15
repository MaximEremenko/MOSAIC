from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormFactorSelection:
    family: str
    calculator: str


__all__ = ["FormFactorSelection"]
