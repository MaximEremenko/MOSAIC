from __future__ import annotations

__all__ = ["ScatteringStage"]


def __getattr__(name: str):
    if name == "ScatteringStage":
        from .stage import ScatteringStage

        return ScatteringStage
    raise AttributeError(name)
