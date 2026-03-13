from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from core.application.common import first_present
from core.application.configuration_registry import DefaultConfigurationProcessorRegistry
from core.domain.models import StructureData, WorkflowParameters
from core.utilities.rmc_neutron_scl import rmc_neutron_scl_
from core.utilities.utils import determine_configuration_file_type


class StructureLoadingService:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = DefaultConfigurationProcessorRegistry()

    def load(
        self, workflow_parameters: WorkflowParameters, working_path: str
    ) -> StructureData:
        parameters = workflow_parameters.to_payload()
        struct = parameters["structInfo"]
        cfg_path = os.path.join(working_path, struct["filename"])
        cfg_type = determine_configuration_file_type(struct["filename"])
        cfg_proc = self.registry.get_factory(cfg_type).create_processor(
            cfg_path, "calculate"
        )
        cfg_proc.process()

        vectors = cfg_proc.get_vectors()
        metric = cfg_proc.get_metric()
        supercell = cfg_proc.get_supercell()
        original_coords = cfg_proc.get_coordinates()
        average_coords = cfg_proc.get_average_coordinates()
        elements = cfg_proc.get_elements()
        refnumbers = cfg_proc.get_refnumbers()
        cells_origin = cfg_proc.get_cells_origin()
        cell_ids = cfg_proc.get_cell_ids() if hasattr(cfg_proc, "get_cell_ids") else None

        coeff = elements.apply(lambda element: rmc_neutron_scl_(element)[0])
        coeff_from_cfg = cfg_proc.get_coeff() if hasattr(cfg_proc, "get_coeff") else None
        coeff_source = first_present(struct, ("coeff_source", "coeffSource")) or "auto"
        coeff_source = str(coeff_source).strip().lower()
        if coeff_source not in {"auto", "config", "file"}:
            raise ValueError(
                f"Unsupported coeff_source={coeff_source!r} (use 'auto'|'config'|'file')."
            )
        if coeff_source in {"auto", "config"} and coeff_from_cfg is not None:
            coeff = coeff_from_cfg

        coeff_file = first_present(
            struct,
            (
                "coeff_file",
                "coeff_filename",
                "coeff_path",
                "coefficients_file",
                "intensity_coeff_file",
                "intensity_coeff_filename",
            ),
        )
        if coeff_file is not None and coeff_source in {"auto", "file"} and (
            coeff_from_cfg is None or coeff_source == "file"
        ):
            coeff_path = Path(coeff_file)
            if not coeff_path.is_absolute():
                coeff_path = Path(working_path) / coeff_path
            coeff = self._load_coefficients_from_file(
                coeff_path=str(coeff_path),
                dim=int(struct["dimension"]),
                n_atoms=int(len(elements)),
                supercell=np.asarray(supercell, int),
                vectors=np.asarray(vectors, float),
                cells_origin=np.asarray(cells_origin.to_numpy(), float),
            )

        return StructureData(
            vectors=vectors,
            metric=metric,
            supercell=supercell,
            original_coords=original_coords,
            average_coords=average_coords,
            elements=elements,
            refnumbers=refnumbers,
            cells_origin=cells_origin,
            cell_ids=cell_ids,
            coeff=coeff,
        )

    def _load_coefficients_from_file(
        self,
        *,
        coeff_path: str,
        dim: int,
        n_atoms: int,
        supercell: np.ndarray,
        vectors: np.ndarray,
        cells_origin: np.ndarray,
    ) -> np.ndarray:
        arr = np.asarray(np.loadtxt(coeff_path, dtype=float), dtype=float)

        if arr.ndim == 0:
            return np.full((n_atoms,), float(arr))
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim == 1:
            if arr.size != n_atoms:
                raise ValueError(
                    f"Coefficient file '{coeff_path}' has {arr.size} values, "
                    f"but configuration has {n_atoms} atoms."
                )
            return arr
        if arr.ndim != 2:
            raise ValueError(
                f"Coefficient file '{coeff_path}' must be 1D or 2D, got shape {arr.shape}."
            )
        if dim != 2:
            raise ValueError(
                f"2D coefficient-matrix mapping is only supported for dim=2; got dim={dim}."
            )

        nx, ny = int(supercell[0]), int(supercell[1])
        if arr.shape not in {(ny, nx), (nx, ny)}:
            raise ValueError(
                f"Coefficient matrix shape {arr.shape} does not match supercell "
                f"(ny,nx)=({ny},{nx}) or (nx,ny)=({nx},{ny})."
            )

        inv_vectors = np.linalg.inv(np.asarray(vectors, float))
        frac = np.asarray(cells_origin, float) @ inv_vectors
        fx = np.asarray(frac[:, 0], float)
        fy = np.asarray(frac[:, 1], float)

        ix = np.clip(np.round(fx * nx).astype(int), 0, nx - 1)
        iy = np.clip(np.round(fy * ny).astype(int), 0, ny - 1)

        if arr.shape == (ny, nx):
            coeff = arr[iy, ix]
        else:
            coeff = arr[ix, iy]

        self.logger.info(
            "Loaded coefficient matrix '%s' (shape %s) mapped onto %d atoms.",
            coeff_path,
            tuple(arr.shape),
            n_atoms,
        )
        return np.asarray(coeff, float)
