from __future__ import annotations

__all__ = ["DecodingStage"]


def __getattr__(name: str):
    if name == "DecodingStage":
        from .stage import DecodingStage

        return DecodingStage
    raise AttributeError(name)
