from __future__ import annotations

from .loader import (
    load_chunk_residual_field_and_grid,
    load_residual_field_and_generate_grid,
    resolve_output_dir,
)

__all__ = [
    "ResidualFieldStage",
    "load_chunk_residual_field_and_grid",
    "load_residual_field_and_generate_grid",
    "resolve_output_dir",
]


def __getattr__(name: str):
    if name == "ResidualFieldStage":
        from .stage import ResidualFieldStage

        return ResidualFieldStage
    raise AttributeError(name)
