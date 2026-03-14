from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dask.distributed import Client

from core.scattering.execution import run_interval_chunk_execution
from core.scattering.planning import build_scattering_interval_chunk_work_units


def process_chunks_with_intervals(
    interval_files: Iterable[Path],
    *,
    chunk_ids: Iterable[int],
    total_reciprocal_points: int,
    point_data_list: list[dict],
    point_data_processor,
    db_manager,
    client: Client | None,
    max_inflight: int = 5_000,
) -> None:
    del interval_files
    del chunk_ids
    work_units = build_scattering_interval_chunk_work_units(
        db_manager.get_unsaved_interval_chunks(),
        dimension=int(db_manager.dimension),
        output_dir=point_data_processor.data_saver.output_dir,
    )
    run_interval_chunk_execution(
        work_units,
        total_reciprocal_points=total_reciprocal_points,
        point_data_list=point_data_list,
        db_manager=db_manager,
        client=client,
        output_dir=point_data_processor.data_saver.output_dir,
        max_inflight=max_inflight,
    )
