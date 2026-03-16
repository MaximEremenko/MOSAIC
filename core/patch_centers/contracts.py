from __future__ import annotations

from dataclasses import dataclass

from core.models import StructureData, WorkflowParameters


@dataclass(frozen=True)
class PointSelectionRequest:
    method: str
    parameters: WorkflowParameters
    structure: StructureData
    hdf5_file_path: str


__all__ = ["PointSelectionRequest"]
