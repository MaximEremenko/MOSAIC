from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScatteringWeightSelection:
    kind: str
    calculator: str


__all__ = ["ScatteringWeightSelection"]
