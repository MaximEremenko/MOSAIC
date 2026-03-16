from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _coerce_optional_tuple(value) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


class _MappingView:
    def get(self, key: str, default: object = None):
        return self.to_mapping().get(key, default)

    def __getitem__(self, key: str):
        return self.to_mapping()[key]


@dataclass
class StructureInfo(_MappingView):
    dimension: int
    working_directory: str = ""
    filename: str = ""
    config_root: str | None = None
    filename_av: str | None = None
    coeff_scheme: str | None = None
    coeff_file: str | None = None
    cells_limits_min: tuple[Any, ...] | None = None
    cells_limits_max: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        self.cells_limits_min = _coerce_optional_tuple(self.cells_limits_min)
        self.cells_limits_max = _coerce_optional_tuple(self.cells_limits_max)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "StructureInfo":
        mapping = dict(payload or {})
        return cls(
            dimension=int(mapping.get("dimension", 1)),
            working_directory=str(mapping.get("working_directory", "")),
            filename=str(mapping.get("filename", "")),
            config_root=mapping.get("config_root"),
            filename_av=mapping.get("filename_av"),
            coeff_scheme=mapping.get("coeff_scheme")
            if mapping.get("coeff_scheme") is not None
            else mapping.get("coeff_source")
            if mapping.get("coeff_source") is not None
            else mapping.get("coefficient_scheme"),
            coeff_file=mapping.get("coeff_file"),
            cells_limits_min=mapping.get("cells_limits_min"),
            cells_limits_max=mapping.get("cells_limits_max"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dimension": self.dimension,
            "working_directory": self.working_directory,
            "filename": self.filename,
        }
        if self.config_root is not None:
            payload["config_root"] = self.config_root
        if self.filename_av is not None:
            payload["filename_av"] = self.filename_av
        if self.coeff_scheme is not None:
            payload["coeff_scheme"] = self.coeff_scheme
        if self.coeff_file is not None:
            payload["coeff_file"] = self.coeff_file
        if self.cells_limits_min is not None:
            payload["cells_limits_min"] = list(self.cells_limits_min)
        if self.cells_limits_max is not None:
            payload["cells_limits_max"] = list(self.cells_limits_max)
        return payload


@dataclass
class ReciprocalSpaceLimit(_MappingView):
    limit: tuple[Any, ...]
    subvolume_step: tuple[Any, ...]

    def __post_init__(self) -> None:
        self.limit = tuple(self.limit)
        self.subvolume_step = tuple(self.subvolume_step)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ReciprocalSpaceLimit":
        mapping = dict(payload or {})
        return cls(
            limit=tuple(mapping.get("limit", ())),
            subvolume_step=tuple(mapping.get("subvolume_step", ())),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "limit": list(self.limit),
            "subvolume_step": list(self.subvolume_step),
        }


@dataclass
class SpecialPoint(_MappingView):
    coordinate: tuple[Any, ...] | None = None
    radius: float | None = None
    space_group_symmetry: Any = None
    shape: str | None = None

    def __post_init__(self) -> None:
        self.coordinate = _coerce_optional_tuple(self.coordinate)
        if self.radius is not None:
            self.radius = float(self.radius)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SpecialPoint":
        mapping = dict(payload or {})
        return cls(
            coordinate=mapping.get("coordinate"),
            radius=mapping.get("radius"),
            space_group_symmetry=mapping.get("spaceGroupSymmetry")
            if "spaceGroupSymmetry" in mapping
            else mapping.get("space_group_symmetry"),
            shape=mapping.get("shape"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.coordinate is not None:
            payload["coordinate"] = list(self.coordinate)
        if self.radius is not None:
            payload["radius"] = self.radius
        if self.space_group_symmetry is not None:
            payload["spaceGroupSymmetry"] = self.space_group_symmetry
        if self.shape is not None:
            payload["shape"] = self.shape
        return payload


@dataclass
class PeakInfo(_MappingView):
    reciprocal_space_limits: tuple[ReciprocalSpaceLimit, ...] = ()
    mask_equation: str | None = None
    special_points: tuple[SpecialPoint, ...] = ()
    r1: float | None = None
    r2: float | None = None

    def __post_init__(self) -> None:
        self.reciprocal_space_limits = tuple(self.reciprocal_space_limits)
        self.special_points = tuple(self.special_points)
        if self.r1 is not None:
            self.r1 = float(self.r1)
        if self.r2 is not None:
            self.r2 = float(self.r2)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PeakInfo":
        mapping = dict(payload or {})
        return cls(
            reciprocal_space_limits=tuple(
                ReciprocalSpaceLimit.from_mapping(item)
                for item in mapping.get("reciprocal_space_limits", [])
            ),
            mask_equation=mapping.get("mask_equation"),
            special_points=tuple(
                SpecialPoint.from_mapping(item)
                for item in mapping.get("specialPoints", [])
            ),
            r1=mapping.get("r1"),
            r2=mapping.get("r2"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reciprocal_space_limits": [
                interval.to_mapping() for interval in self.reciprocal_space_limits
            ]
        }
        if self.mask_equation is not None:
            payload["mask_equation"] = self.mask_equation
        if self.special_points:
            payload["specialPoints"] = [
                point.to_mapping() for point in self.special_points
            ]
        if self.r1 is not None:
            payload["r1"] = self.r1
        if self.r2 is not None:
            payload["r2"] = self.r2
        return payload


@dataclass
class ProcessingPoint(_MappingView):
    filename: str | None = None
    element_symbol: str | None = None
    reference_number: int | None = None
    dist_from_atom_center: tuple[Any, ...] | None = None
    step_in_angstrom: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        self.dist_from_atom_center = _coerce_optional_tuple(self.dist_from_atom_center)
        self.step_in_angstrom = _coerce_optional_tuple(self.step_in_angstrom)
        if self.reference_number is not None:
            self.reference_number = int(self.reference_number)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ProcessingPoint":
        mapping = dict(payload or {})
        return cls(
            filename=mapping.get("filename"),
            element_symbol=mapping.get("elementSymbol"),
            reference_number=mapping.get("referenceNumber"),
            dist_from_atom_center=mapping.get("distFromAtomCenter"),
            step_in_angstrom=mapping.get("stepInAngstrom"),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.filename is not None:
            payload["filename"] = self.filename
        if self.element_symbol is not None:
            payload["elementSymbol"] = self.element_symbol
        if self.reference_number is not None:
            payload["referenceNumber"] = self.reference_number
        if self.dist_from_atom_center is not None:
            payload["distFromAtomCenter"] = list(self.dist_from_atom_center)
        if self.step_in_angstrom is not None:
            payload["stepInAngstrom"] = list(self.step_in_angstrom)
        return payload


@dataclass
class RSpaceInfo(_MappingView):
    num_chunks: int = 1
    fresh_start: bool = False
    method: str = "from_average"
    filter_type: str = "Chebyshev"
    save_rifft_coordinates: bool = False
    smooth_intensities: bool = True
    print_intensities: bool = False
    rspace_parallel_processing: bool = False
    points: tuple[ProcessingPoint, ...] = ()
    mode: str = "displacement"
    run_postprocessing: bool = True
    cells_limits_min: tuple[Any, ...] | None = None
    cells_limits_max: tuple[Any, ...] | None = None
    chemical_filtered_ordering: bool | None = None
    coeff_center_by: str | None = None
    use_coeff: bool | None = None
    decoder: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.num_chunks = int(self.num_chunks)
        self.points = tuple(self.points)
        self.cells_limits_min = _coerce_optional_tuple(self.cells_limits_min)
        self.cells_limits_max = _coerce_optional_tuple(self.cells_limits_max)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RSpaceInfo":
        mapping = dict(payload or {})
        return cls(
            num_chunks=int(mapping.get("num_chunks", 1)),
            fresh_start=bool(mapping.get("fresh_start", False)),
            method=str(mapping.get("method", "from_average")),
            filter_type=str(mapping.get("filter_type", "Chebyshev")),
            save_rifft_coordinates=bool(mapping.get("save_rifft_coordinates", False)),
            smooth_intensities=bool(mapping.get("smooth_intensities", True)),
            print_intensities=bool(mapping.get("print_intensities", False)),
            rspace_parallel_processing=bool(
                mapping.get("rspace_parallel_processing", False)
            ),
            points=tuple(
                ProcessingPoint.from_mapping(item) for item in mapping.get("points", [])
            ),
            mode=str(mapping.get("mode", "displacement")),
            run_postprocessing=bool(mapping.get("run_postprocessing", True)),
            cells_limits_min=mapping.get("cells_limits_min"),
            cells_limits_max=mapping.get("cells_limits_max"),
            chemical_filtered_ordering=mapping.get("chemical_filtered_ordering"),
            coeff_center_by=mapping.get("coeff_center_by"),
            use_coeff=mapping.get("use_coeff"),
            decoder=dict(mapping.get("decoder")) if isinstance(mapping.get("decoder"), Mapping) else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_chunks": self.num_chunks,
            "fresh_start": self.fresh_start,
            "method": self.method,
            "filter_type": self.filter_type,
            "save_rifft_coordinates": self.save_rifft_coordinates,
            "smooth_intensities": self.smooth_intensities,
            "print_intensities": self.print_intensities,
            "rspace_parallel_processing": self.rspace_parallel_processing,
            "points": [point.to_mapping() for point in self.points],
            "mode": self.mode,
            "run_postprocessing": self.run_postprocessing,
        }
        if self.cells_limits_min is not None:
            payload["cells_limits_min"] = list(self.cells_limits_min)
        if self.cells_limits_max is not None:
            payload["cells_limits_max"] = list(self.cells_limits_max)
        if self.chemical_filtered_ordering is not None:
            payload["chemical_filtered_ordering"] = self.chemical_filtered_ordering
        if self.coeff_center_by is not None:
            payload["coeff_center_by"] = self.coeff_center_by
        if self.use_coeff is not None:
            payload["use_coeff"] = self.use_coeff
        if self.decoder is not None:
            payload["decoder"] = dict(self.decoder)
        return payload


@dataclass
class ScatteringWeightConfig(_MappingView):
    kind: str = "ones"
    calculator: str = "default"

    @staticmethod
    def _normalize_kind(value: Any) -> str:
        normalized = str(value or "ones").strip().lower()
        if normalized in {
            "unity",
            "atomicnumber",
            "z",
            "neutron_scattering_length",
            "neutron_scattering_lengths",
        }:
            raise ValueError(
                "Legacy scattering_weights.kind values are no longer supported. "
                "Use one of: ones, atomic_number, neutron, xray, electron."
            )
        return normalized

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ScatteringWeightConfig":
        mapping = dict(payload or {})
        legacy_keys = {
            "family",
            "source",
            "factory",
            "type",
            "method",
            "name",
            "form_factor_family",
            "form_factor_calculator",
        }
        present_legacy_keys = sorted(key for key in legacy_keys if key in mapping)
        if present_legacy_keys:
            raise ValueError(
                "Legacy scattering-weight keys are no longer supported: "
                f"{', '.join(present_legacy_keys)}. "
                "Use runtime.scattering_weights.kind and runtime.scattering_weights.calculator."
            )
        unsupported_keys = sorted(
            key for key in mapping if key not in {"kind", "calculator"}
        )
        if unsupported_keys:
            raise ValueError(
                "Unsupported scattering-weight keys: "
                f"{', '.join(unsupported_keys)}. "
                "Use runtime.scattering_weights.kind and runtime.scattering_weights.calculator."
            )
        return cls(
            kind=cls._normalize_kind(mapping.get("kind") or "ones"),
            calculator=str(mapping.get("calculator") or "default").strip(),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "calculator": self.calculator,
        }


@dataclass
class WorkflowRuntimeInfo(_MappingView):
    scattering_weights: ScatteringWeightConfig = field(default_factory=ScatteringWeightConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "WorkflowRuntimeInfo":
        mapping = dict(payload or {})
        present_legacy_keys = sorted(
            key for key in ("scatteringWeights", "form_factor", "formFactor") if key in mapping
        )
        if present_legacy_keys:
            raise ValueError(
                "Legacy runtime scattering-weight keys are no longer supported: "
                f"{', '.join(present_legacy_keys)}. "
                "Use runtime.scattering_weights."
            )
        scattering_weights = ScatteringWeightConfig.from_mapping(
            mapping.get("scattering_weights")
        )
        extra = {
            key: value
            for key, value in mapping.items()
            if key != "scattering_weights"
        }
        return cls(scattering_weights=scattering_weights, extra=extra)

    def to_mapping(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["scattering_weights"] = self.scattering_weights.to_mapping()
        return payload


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
    struct_info: StructureInfo
    peak_info: PeakInfo
    rspace_info: RSpaceInfo
    runtime_info: WorkflowRuntimeInfo = field(default_factory=WorkflowRuntimeInfo)
    input_parameters_path: str | None = None
    config_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "struct_info",
            self.struct_info
            if isinstance(self.struct_info, StructureInfo)
            else StructureInfo.from_mapping(self.struct_info),
        )
        object.__setattr__(
            self,
            "peak_info",
            self.peak_info
            if isinstance(self.peak_info, PeakInfo)
            else PeakInfo.from_mapping(self.peak_info),
        )
        object.__setattr__(
            self,
            "rspace_info",
            self.rspace_info
            if isinstance(self.rspace_info, RSpaceInfo)
            else RSpaceInfo.from_mapping(self.rspace_info),
        )
        object.__setattr__(
            self,
            "runtime_info",
            self.runtime_info
            if isinstance(self.runtime_info, WorkflowRuntimeInfo)
            else WorkflowRuntimeInfo.from_mapping(self.runtime_info),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "WorkflowParameters":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            struct_info=StructureInfo.from_mapping(payload.get("structInfo", {})),
            peak_info=PeakInfo.from_mapping(payload.get("peakInfo", {})),
            rspace_info=RSpaceInfo.from_mapping(payload.get("rspace_info", {})),
            runtime_info=WorkflowRuntimeInfo.from_mapping(
                payload.get("runtime_info", {})
            ),
            input_parameters_path=payload.get("input_parameters_path"),
            config_root=payload.get("config_root"),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "structInfo": self.struct_info.to_mapping(),
            "peakInfo": self.peak_info.to_mapping(),
            "rspace_info": self.rspace_info.to_mapping(),
            "runtime_info": self.runtime_info.to_mapping(),
        }
        if self.input_parameters_path is not None:
            payload["input_parameters_path"] = self.input_parameters_path
        if self.config_root is not None:
            payload["config_root"] = self.config_root
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


@dataclass(frozen=True)
class StructureData:
    vectors: np.ndarray
    metric: dict[str, np.ndarray | float] | np.ndarray
    supercell: np.ndarray
    original_coords: object
    average_coords: object
    elements: object
    refnumbers: object
    cells_origin: object
    cell_ids: np.ndarray | None = None
    coeff: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vectors", np.asarray(self.vectors, dtype=float))
        object.__setattr__(self, "supercell", np.asarray(self.supercell))
        if self.cell_ids is not None:
            object.__setattr__(self, "cell_ids", np.asarray(self.cell_ids))
        if self.coeff is not None:
            object.__setattr__(self, "coeff", np.asarray(self.coeff, dtype=float))
        if isinstance(self.metric, Mapping):
            metric = {
                str(key): (
                    np.asarray(value, dtype=float)
                    if isinstance(value, (list, tuple, np.ndarray))
                    else float(value)
                )
                for key, value in self.metric.items()
            }
            object.__setattr__(self, "metric", metric)
        else:
            object.__setattr__(self, "metric", np.asarray(self.metric, dtype=float))

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
class ReciprocalSpaceArtifacts:
    output_dir: str
    saver: object
    point_data_processor: object
    db_manager: object
    compact_intervals: list[dict[str, Any]]
    padded_intervals: list[dict[str, Any]]
    transient_interval_payloads: dict[int, Any] = field(default_factory=dict)

    def close(self) -> None:
        close = getattr(self.db_manager, "close", None)
        if callable(close):
            close()


__all__ = [
    "ScatteringWeightConfig",
    "PeakInfo",
    "PointData",
    "RSpaceInfo",
    "ReciprocalInterval",
    "ReciprocalSpaceArtifacts",
    "ReciprocalSpaceLimit",
    "RunSettings",
    "RuntimeSettings",
    "SpecialPoint",
    "StructureData",
    "StructureInfo",
    "WorkflowParameters",
    "WorkflowRuntimeInfo",
]
