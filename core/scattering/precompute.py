from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from dask.distributed import Client

from core.scattering.execution import run_interval_precompute
from core.scattering.planning import (
    build_scattering_interval_lookup,
    build_scattering_precompute_work_units,
)
from core.storage.database_manager import DatabaseManager


def precompute_intervals(
    reciprocal_space_intervals: Iterable[dict],
    *,
    B_: np.ndarray,
    parameters: Dict[str, Any],
    unique_elements: Iterable[str],
    mask_params: Dict[str, Any],
    MaskStrategy,
    supercell: np.ndarray,
    out_dir: Path,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
    db: DatabaseManager,
    client: Client | None,
) -> List[Path]:
    intervals = list(reciprocal_space_intervals)
    output_dir = str(out_dir.parent)
    work_units = build_scattering_precompute_work_units(
        intervals,
        dimension=int(len(supercell)),
        output_dir=output_dir,
    )
    interval_lookup = build_scattering_interval_lookup(intervals)
    return run_interval_precompute(
        work_units,
        interval_lookup=interval_lookup,
        B_=B_,
        parameters=parameters,
        unique_elements=unique_elements,
        mask_params=mask_params,
        MaskStrategy=MaskStrategy,
        supercell=supercell,
        output_dir=output_dir,
        original_coords=original_coords,
        cells_origin=cells_origin,
        elements_arr=elements_arr,
        charge=charge,
        ff_factory=ff_factory,
        db=db,
        client=client,
    )
