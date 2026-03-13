"""
Refactored amplitude-delta calculator facade.

The public API remains ``compute_amplitudes_delta()``, while the runtime,
grid-generation, precompute, persistence, and chunk-processing concerns live
in dedicated helper modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from dask.distributed import Client

from core.managers.database_manager import DatabaseManager
from core.processors.amplitude_chunk_processing import process_chunks_with_intervals
from core.processors.amplitude_grid import (
    _to_interval_dict,
    reciprocal_space_points_counter,
)
from core.processors.amplitude_precompute import precompute_intervals
from core.processors.amplitude_runtime import CuPyCleanup, _is_sync_client, _quiet_db_info
from core.processors.point_data_processor import PointDataProcessor


logger = logging.getLogger(__name__)


def compute_amplitudes_delta(
    parameters: Dict[str, Any],
    FormFactorFactoryProducer,
    MaskStrategy,
    MaskStrategyParameters: Dict[str, Any],
    db_manager: DatabaseManager,
    output_dir: str,
    point_data_processor: PointDataProcessor,
    client: Client,
):
    try:
        if client is not None and not _is_sync_client(client):
            client.register_worker_plugin(CuPyCleanup(), name="cupy-cleanup")
    except ValueError:
        pass

    reciprocal_space_intervals_all = parameters["reciprocal_space_intervals_all"]
    reciprocal_space_intervals = parameters["reciprocal_space_intervals"]
    point_data_list = parameters["point_data_list"]
    original_coords = parameters["original_coords"]
    cells_origin = parameters["cells_origin"]
    elements_arr = parameters["elements"]
    vectors = parameters["vectors"]
    supercell = parameters["supercell"]
    charge = parameters.get("charge", 0.0)

    B_ = np.linalg.inv(vectors / supercell)
    unique_elements = np.unique(elements_arr)

    total_pts = sum(
        reciprocal_space_points_counter(_to_interval_dict(iv), supercell)
        for iv in reciprocal_space_intervals_all
    )
    logger.info("Total reciprocal-space integer points: %s", total_pts)

    interval_dir = Path(output_dir) / "precomputed_intervals"
    with _quiet_db_info():
        interval_files = precompute_intervals(
            reciprocal_space_intervals,
            B_=B_,
            parameters=parameters,
            unique_elements=unique_elements,
            mask_params=MaskStrategyParameters,
            MaskStrategy=MaskStrategy,
            supercell=supercell,
            out_dir=interval_dir,
            original_coords=original_coords,
            cells_origin=cells_origin,
            elements_arr=elements_arr,
            charge=charge,
            ff_factory=FormFactorFactoryProducer,
            db=db_manager,
            client=client,
        )

    chunk_ids = db_manager.get_pending_chunk_ids()
    with _quiet_db_info():
        process_chunks_with_intervals(
            interval_files,
            chunk_ids=chunk_ids,
            total_reciprocal_points=total_pts,
            point_data_list=point_data_list,
            point_data_processor=point_data_processor,
            db_manager=db_manager,
            client=client,
        )

    logger.info("Completed compute_amplitudes_delta")
