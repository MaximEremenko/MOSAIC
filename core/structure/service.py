from __future__ import annotations

import logging
import os

from core.config.registry import (
    DefaultConfigurationProcessorRegistry,
)
from core.models import StructureData, WorkflowParameters
from core.config.file_type import determine_configuration_file_type

from .coefficients import resolve_structure_coefficients


class StructureLoadingService:
    def __init__(
        self,
        *,
        registry: DefaultConfigurationProcessorRegistry | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = registry or DefaultConfigurationProcessorRegistry()

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

        coeff = resolve_structure_coefficients(
            struct=struct,
            cfg_proc=cfg_proc,
            working_path=working_path,
            elements=elements,
            supercell=supercell,
            vectors=vectors,
            cells_origin=cells_origin,
            logger=self.logger,
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
