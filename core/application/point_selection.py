from __future__ import annotations

from pathlib import Path

from core.domain.models import PointData, PointSelectionRequest
from core.factories.point_processor_factory import PointProcessorFactory


class PointSelectionService:
    def select(self, request: PointSelectionRequest) -> PointData:
        parameters = request.parameters.to_payload()
        parameters["hdf5_file_path"] = request.hdf5_file_path
        Path(request.hdf5_file_path).parent.mkdir(parents=True, exist_ok=True)
        processor = PointProcessorFactory.create_processor(
            request.method,
            parameters,
            average_structure=request.structure.average_structure_payload(),
        )
        processor.process_parameters()
        return processor.get_point_data()
