from __future__ import annotations

from pathlib import Path

from core.models import PointData
from core.patch_centers.contracts import PointSelectionRequest
from core.patch_centers.factory import PointProcessorFactory


class PointSelectionService:
    def __init__(self, *, processor_factory=None) -> None:
        self.processor_factory = processor_factory or PointProcessorFactory.create_processor

    def select(self, request: PointSelectionRequest) -> PointData:
        parameters = request.parameters.to_payload()
        parameters["hdf5_file_path"] = request.hdf5_file_path
        Path(request.hdf5_file_path).parent.mkdir(parents=True, exist_ok=True)
        processor = self.processor_factory(
            request.method,
            parameters,
            average_structure=request.structure.average_structure_payload(),
        )
        processor.process_parameters()
        return processor.get_point_data()
