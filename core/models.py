from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeSettings:
    worker_dashboard: bool = False
    backend: str = "local"
    max_workers: int = 2
    threads_per_worker: int = 16
    processes: bool = False
    wait_timeout: str = "120s"


@dataclass(frozen=True)
class RunSettings:
    run_parameters_path: Path
    input_parameters_path: Path
    config_root: Path
    working_path: Path
    runtime: RuntimeSettings
    root_path: Path | None = None


@dataclass(frozen=True)
class WorkflowParameters:
    schema_version: int
    struct_info: dict[str, Any]
    peak_info: dict[str, Any]
    rspace_info: dict[str, Any]
    runtime_info: dict[str, Any] = field(default_factory=dict)
    input_parameters_path: str | None = None
    config_root: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkflowParameters":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            struct_info=dict(payload.get("structInfo", {})),
            peak_info=dict(payload.get("peakInfo", {})),
            rspace_info=dict(payload.get("rspace_info", {})),
            runtime_info=dict(payload.get("runtime_info", {})),
            input_parameters_path=payload.get("input_parameters_path"),
            config_root=payload.get("config_root"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "structInfo": dict(self.struct_info),
            "peakInfo": dict(self.peak_info),
            "rspace_info": dict(self.rspace_info),
            "runtime_info": dict(self.runtime_info),
        }
        if self.input_parameters_path is not None:
            payload["input_parameters_path"] = self.input_parameters_path
        if self.config_root is not None:
            payload["config_root"] = self.config_root
        return payload


@dataclass(frozen=True)
class MaskSpec:
    parameters: dict[str, Any]
    postprocessing_mode: str = "displacement"


@dataclass(frozen=True)
class ReciprocalInterval:
    h_range: tuple[float, float]
    k_range: tuple[float, float] = (0.0, 0.0)
    l_range: tuple[float, float] = (0.0, 0.0)
    interval_id: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ReciprocalInterval":
        return cls(
            h_range=tuple(payload.get("h_range", (0.0, 0.0))),
            k_range=tuple(payload.get("k_range", (0.0, 0.0))),
            l_range=tuple(payload.get("l_range", (0.0, 0.0))),
            interval_id=payload.get("id"),
        )

    def to_mapping(self, dim: int) -> dict[str, Any]:
        payload: dict[str, Any] = {"h_range": tuple(self.h_range)}
        if dim >= 2:
            payload["k_range"] = tuple(self.k_range)
        if dim == 3:
            payload["l_range"] = tuple(self.l_range)
        if self.interval_id is not None:
            payload["id"] = self.interval_id
        return payload


@dataclass
class PointData:
    coordinates: np.ndarray
    dist_from_atom_center: np.ndarray
    step_in_frac: np.ndarray
    central_point_ids: np.ndarray
    chunk_ids: np.ndarray
    grid_amplitude_initialized: np.ndarray

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates)
        self.dist_from_atom_center = np.asarray(self.dist_from_atom_center)
        self.step_in_frac = np.asarray(self.step_in_frac)
        self.central_point_ids = np.asarray(self.central_point_ids)
        self.chunk_ids = np.asarray(self.chunk_ids)
        self.grid_amplitude_initialized = np.asarray(self.grid_amplitude_initialized)

        if self.chunk_ids.size == 0:
            self.chunk_ids = np.zeros(self.coordinates.shape[0], dtype=int)
        if self.grid_amplitude_initialized.size == 0:
            self.grid_amplitude_initialized = np.zeros(
                self.coordinates.shape[0], dtype=bool
            )


@dataclass(frozen=True)
class StructureData:
    vectors: Any
    metric: Any
    supercell: Any
    original_coords: Any
    average_coords: Any
    elements: Any
    refnumbers: Any
    cells_origin: Any
    cell_ids: Any = None
    coeff: Any = None

    def average_structure_payload(self) -> dict[str, Any]:
        return {
            "average_coords": self.average_coords,
            "elements": self.elements,
            "refnumbers": self.refnumbers,
            "vectors": self.vectors,
            "metric": self.metric,
            "supercell": self.supercell,
            "cell_ids": self.cell_ids,
        }


@dataclass(frozen=True)
class PointSelectionRequest:
    method: str
    parameters: WorkflowParameters
    structure: StructureData
    hdf5_file_path: str


@dataclass(frozen=True)
class PostprocessingRequest:
    output_dir: str
    parameters: dict[str, Any]
    client: Any


@dataclass(frozen=True)
class ReciprocalSpaceArtifacts:
    output_dir: str
    saver: Any
    point_data_processor: Any
    db_manager: Any
    compact_intervals: list[dict[str, Any]]
    padded_intervals: list[dict[str, Any]]

    def close(self) -> None:
        close = getattr(self.db_manager, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class FormFactorSelection:
    family: str
    calculator: str


@dataclass(frozen=True)
class AmplitudeExecutionContext:
    workflow_parameters: WorkflowParameters
    structure: StructureData
    artifacts: ReciprocalSpaceArtifacts
    dimension: int
    postprocessing_mode: str
    unsaved_interval_chunks: list[tuple[int, int]]
    point_rows: list[dict[str, Any]]
    intervals: list[dict[str, Any]]
    chemical_filtered: bool
    use_coeff: bool
    centered_coefficients: np.ndarray
    mask_strategy: Any
    form_factor_selection: FormFactorSelection


@dataclass(frozen=True)
class PostprocessingContext:
    workflow_parameters: WorkflowParameters
    structure: StructureData
    artifacts: ReciprocalSpaceArtifacts
    postprocessing_mode: str
