from __future__ import annotations

from pathlib import Path

import numpy as np

from core.application.point_selection.point_data import PointDataProcessor
from core.application.reciprocal_space.interval_mapping import pad_interval
from core.application.reciprocal_space.manager import (
    ReciprocalSpaceIntervalManager,
)
from core.domain.models import PointData, ReciprocalSpaceArtifacts, WorkflowParameters
from core.infrastructure.persistence.database_manager import DatabaseManager
from core.infrastructure.storage import RIFFTInDataSaver


class ReciprocalSpacePreparationService:
    def prepare(
        self,
        workflow_parameters: WorkflowParameters,
        point_data: PointData,
        supercell,
        output_dir: str,
    ) -> ReciprocalSpaceArtifacts:
        parameters = workflow_parameters.to_payload()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        saver = RIFFTInDataSaver(output_dir, "hdf5")
        point_data_processor = PointDataProcessor(
            data_saver=saver,
            save_rifft_coordinates=parameters["rspace_info"].get(
                "save_rifft_coordinates", False
            ),
        )
        point_data_processor.process_point_data(point_data)

        dim = int(parameters["structInfo"]["dimension"])
        db_manager = DatabaseManager(
            str(Path(output_dir) / "point_reciprocal_space_associations.db"),
            dim,
        )
        db_manager.insert_point_data_batch(
            [
                {
                    "central_point_id": int(point_data.central_point_ids[index]),
                    "coordinates": point_data.coordinates[index].tolist(),
                    "dist_from_atom_center": point_data.dist_from_atom_center[
                        index
                    ].tolist(),
                    "step_in_frac": point_data.step_in_frac[index].tolist(),
                    "chunk_id": int(point_data.chunk_ids[index]),
                    "grid_amplitude_initialized": int(
                        point_data.grid_amplitude_initialized[index]
                    ),
                }
                for index in range(point_data.central_point_ids.size)
            ]
        )

        recip_h5 = str(Path(output_dir) / "point_reciprocal_space_data.hdf5")
        reciprocal_manager = ReciprocalSpaceIntervalManager(
            recip_h5, parameters, supercell
        )
        reciprocal_manager.process_reciprocal_space_intervals()

        compact_intervals: list[dict] = []
        for interval in reciprocal_manager.reciprocal_space_intervals:
            entry = {"h_range": interval["h_range"]}
            if dim >= 2:
                entry["k_range"] = interval.get("k_range", (0, 0))
            if dim == 3:
                entry["l_range"] = interval.get("l_range", (0, 0))
            compact_intervals.append(entry)

        interval_ids = db_manager.insert_reciprocal_space_interval_batch(compact_intervals)
        unique_chunks = np.unique(point_data.chunk_ids)
        db_manager.insert_interval_chunk_status_batch(
            [
                (interval_id, int(chunk_id), 0)
                for interval_id in interval_ids
                for chunk_id in unique_chunks
            ]
        )
        padded_intervals = [pad_interval(interval, dim) for interval in compact_intervals]
        return ReciprocalSpaceArtifacts(
            output_dir=output_dir,
            saver=saver,
            point_data_processor=point_data_processor,
            db_manager=db_manager,
            compact_intervals=compact_intervals,
            padded_intervals=padded_intervals,
        )
