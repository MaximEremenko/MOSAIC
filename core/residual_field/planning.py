from __future__ import annotations

import hashlib
import json

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items())}
    return value


def build_residual_field_parameter_digest(parameters: dict[str, object]) -> str:
    payload = {
        "postprocessing_mode": parameters.get("postprocessing_mode"),
        "supercell": _to_jsonable(parameters.get("supercell")),
        "rspace_info": _to_jsonable(parameters.get("rspace_info", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def build_residual_field_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    parameters: dict[str, object],
    output_dir: str,
) -> list[ResidualFieldWorkUnit]:
    digest = build_residual_field_parameter_digest(parameters)
    patch_scope = "chunk"
    window_spec = str(parameters.get("postprocessing_mode", "displacement"))
    return [
        ResidualFieldWorkUnit.interval_chunk(
            interval_id=int(interval_id),
            chunk_id=int(chunk_id),
            parameter_digest=digest,
            output_dir=output_dir,
            patch_scope=patch_scope,
            window_spec=window_spec,
        )
        for interval_id, chunk_id in sorted(
            {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
        )
    ]


def chunk_ids_for_work_units(work_units: list[ResidualFieldWorkUnit]) -> list[int]:
    return sorted({int(work_unit.chunk_id) for work_unit in work_units})


__all__ = [
    "build_residual_field_parameter_digest",
    "build_residual_field_work_units",
    "chunk_ids_for_work_units",
]
