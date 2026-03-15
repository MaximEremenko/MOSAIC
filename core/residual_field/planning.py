from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in sorted(asdict(value).items())}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items())}
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, int, float, bool)):
        return {
            str(key): _to_jsonable(val)
            for key, val in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return value


def _parameter_value(parameters: object, key: str, default=None):
    if isinstance(parameters, dict):
        return parameters.get(key, default)
    return getattr(parameters, key, default)


def build_residual_field_parameter_digest(parameters: object) -> str:
    payload = {
        "postprocessing_mode": _parameter_value(parameters, "postprocessing_mode"),
        "supercell": _to_jsonable(_parameter_value(parameters, "supercell")),
        "rspace_info": _to_jsonable(_parameter_value(parameters, "rspace_info", {})),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _batch_interval_chunks(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    max_intervals_per_shard: int,
) -> list[tuple[int, tuple[int, ...]]]:
    if max_intervals_per_shard <= 0:
        raise ValueError("max_intervals_per_shard must be positive.")
    grouped: dict[int, list[int]] = {}
    for interval_id, chunk_id in sorted(
        {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
    ):
        grouped.setdefault(int(chunk_id), []).append(int(interval_id))
    batches: list[tuple[int, tuple[int, ...]]] = []
    for chunk_id in sorted(grouped):
        interval_ids = grouped[chunk_id]
        for start in range(0, len(interval_ids), max_intervals_per_shard):
            batches.append((chunk_id, tuple(interval_ids[start : start + max_intervals_per_shard])))
    return batches


def build_residual_field_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    parameters: object,
    output_dir: str,
    max_intervals_per_shard: int = 1,
) -> list[ResidualFieldWorkUnit]:
    digest = build_residual_field_parameter_digest(parameters)
    patch_scope = "chunk"
    window_spec = str(_parameter_value(parameters, "postprocessing_mode", "displacement"))
    work_units: list[ResidualFieldWorkUnit] = []
    for chunk_id, interval_ids in _batch_interval_chunks(
        unsaved_interval_chunks,
        max_intervals_per_shard=max_intervals_per_shard,
    ):
        if len(interval_ids) == 1:
            work_units.append(
                ResidualFieldWorkUnit.interval_chunk(
                    interval_id=int(interval_ids[0]),
                    chunk_id=int(chunk_id),
                    parameter_digest=digest,
                    output_dir=output_dir,
                    patch_scope=patch_scope,
                    window_spec=window_spec,
                )
            )
            continue
        work_units.append(
            ResidualFieldWorkUnit.interval_chunk_batch(
                interval_ids=interval_ids,
                chunk_id=int(chunk_id),
                parameter_digest=digest,
                output_dir=output_dir,
                patch_scope=patch_scope,
                window_spec=window_spec,
            )
        )
    return work_units


def chunk_ids_for_work_units(work_units: list[ResidualFieldWorkUnit]) -> list[int]:
    return sorted({int(work_unit.chunk_id) for work_unit in work_units})


__all__ = [
    "_batch_interval_chunks",
    "build_residual_field_parameter_digest",
    "build_residual_field_work_units",
    "chunk_ids_for_work_units",
]
