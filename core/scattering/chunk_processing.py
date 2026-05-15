from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from dask.distributed import Client


def process_chunks_with_intervals(
    interval_files: Iterable[Path],
    *,
    chunk_ids: Iterable[int],
    total_reciprocal_points: int,
    point_data_list: list[dict],
    point_data_processor,
    db_manager,
    client: "Client | None",
    max_inflight: int = 5_000,
) -> None:
    del (
        interval_files,
        chunk_ids,
        total_reciprocal_points,
        point_data_list,
        point_data_processor,
        db_manager,
        client,
        max_inflight,
    )
    raise RuntimeError(
        "process_chunks_with_intervals is retired for current runs. Use "
        "run_scattering_stage so stage-2 work units carry ScatteringWorkIdentity "
        "and complete through run-scoped attempt commits."
    )
