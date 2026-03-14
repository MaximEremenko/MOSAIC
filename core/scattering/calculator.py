"""
Refactored amplitude-delta calculator facade.

The public API remains ``compute_amplitudes_delta()``, while Phase 3 scattering
execution now lives behind the explicit ``planning / execution / tasks /
accumulation / artifacts / kernels`` boundaries.
"""

from __future__ import annotations

from typing import Any, Dict

from dask.distributed import Client

from core.scattering.execution import run_scattering_stage
from core.patch_centers.point_data import PointDataProcessor
from core.storage.database_manager import DatabaseManager


execute_scattering_stage = run_scattering_stage


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
    return execute_scattering_stage(
        parameters=parameters,
        FormFactorFactoryProducer=FormFactorFactoryProducer,
        MaskStrategy=MaskStrategy,
        MaskStrategyParameters=MaskStrategyParameters,
        db_manager=db_manager,
        output_dir=output_dir,
        point_data_processor=point_data_processor,
        client=client,
    )
