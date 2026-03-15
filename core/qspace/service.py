from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from core.patch_centers.point_data import PointDataProcessor
from core.qspace.intervals.interval_mapping import pad_interval
from core.qspace.intervals.manager import (
    ReciprocalSpaceIntervalManager,
)
from core.models import PointData, ReciprocalSpaceArtifacts, WorkflowParameters
from core.storage.database_manager import DatabaseManager
from core.storage.rifft_in_data_saver import RIFFTInDataSaver


class _ReciprocalSpaceArtifactBundle:
    def __init__(
        self,
        *,
        parameters: dict,
        dimension: int,
        saver: RIFFTInDataSaver,
        point_data_processor: PointDataProcessor,
        db_manager: DatabaseManager,
        reciprocal_manager: ReciprocalSpaceIntervalManager,
    ) -> None:
        self.parameters = parameters
        self.dimension = dimension
        self.saver = saver
        self.point_data_processor = point_data_processor
        self.db_manager = db_manager
        self.reciprocal_manager = reciprocal_manager


class _DefaultReciprocalSpaceArtifactBuilder:
    def __init__(
        self,
        *,
        saver_factory: Callable[[str, str], RIFFTInDataSaver] = RIFFTInDataSaver,
        db_manager_factory: Callable[[str, int], DatabaseManager] = DatabaseManager,
    ) -> None:
        self.saver_factory = saver_factory
        self.db_manager_factory = db_manager_factory

    def create(
        self,
        *,
        workflow_parameters: WorkflowParameters,
        output_dir: str,
        supercell,
    ) -> _ReciprocalSpaceArtifactBundle:
        parameters = workflow_parameters.to_payload()
        dimension = workflow_parameters.struct_info.dimension
        saver = self.saver_factory(output_dir, "hdf5")
        point_data_processor = PointDataProcessor(
            data_saver=saver,
            save_rifft_coordinates=workflow_parameters.rspace_info.save_rifft_coordinates,
        )
        db_manager = self.db_manager_factory(
            str(Path(output_dir) / "point_reciprocal_space_associations.db"),
            dimension,
        )
        reciprocal_manager = ReciprocalSpaceIntervalManager(
            str(Path(output_dir) / "point_reciprocal_space_data.hdf5"),
            parameters,
            supercell,
        )
        return _ReciprocalSpaceArtifactBundle(
            parameters=parameters,
            dimension=dimension,
            saver=saver,
            point_data_processor=point_data_processor,
            db_manager=db_manager,
            reciprocal_manager=reciprocal_manager,
        )


class ReciprocalSpacePreparationService:
    def __init__(
        self,
        *,
        artifact_builder: _DefaultReciprocalSpaceArtifactBuilder | None = None,
    ) -> None:
        self.artifact_builder = artifact_builder or _DefaultReciprocalSpaceArtifactBuilder()

    def prepare(
        self,
        workflow_parameters: WorkflowParameters,
        point_data: PointData,
        supercell,
        output_dir: str,
    ) -> ReciprocalSpaceArtifacts:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        artifact_bundle = self.artifact_builder.create(
            workflow_parameters=workflow_parameters,
            output_dir=output_dir,
            supercell=supercell,
        )
        artifact_bundle.point_data_processor.process_point_data(point_data)
        artifact_bundle.db_manager.insert_point_data_batch(
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
        artifact_bundle.reciprocal_manager.process_reciprocal_space_intervals()

        compact_intervals: list[dict] = []
        for interval in artifact_bundle.reciprocal_manager.reciprocal_space_intervals:
            entry = {"h_range": interval["h_range"]}
            if artifact_bundle.dimension >= 2:
                entry["k_range"] = interval.get("k_range", (0, 0))
            if artifact_bundle.dimension == 3:
                entry["l_range"] = interval.get("l_range", (0, 0))
            compact_intervals.append(entry)

        interval_ids = artifact_bundle.db_manager.insert_reciprocal_space_interval_batch(
            compact_intervals
        )
        unique_chunks = np.unique(point_data.chunk_ids)
        artifact_bundle.db_manager.insert_interval_chunk_status_batch(
            [
                (interval_id, int(chunk_id), 0)
                for interval_id in interval_ids
                for chunk_id in unique_chunks
            ]
        )
        padded_intervals = [
            pad_interval(interval, artifact_bundle.dimension)
            for interval in compact_intervals
        ]
        return ReciprocalSpaceArtifacts(
            output_dir=output_dir,
            saver=artifact_bundle.saver,
            point_data_processor=artifact_bundle.point_data_processor,
            db_manager=artifact_bundle.db_manager,
            compact_intervals=compact_intervals,
            padded_intervals=padded_intervals,
        )
