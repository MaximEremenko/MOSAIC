"""Pure work-unit helper utilities for the residual-field stage.

All functions operate on ResidualFieldWorkUnit values and return plain Python
types.  None of these import from core.residual_field.execution.
"""

from __future__ import annotations

from core.residual_field.contracts import ResidualFieldWorkUnit
from core.storage.digests import digest_dict

__all__ = [
    "_hex64_or_digest",
    "_interval_inputs_for_work_unit",
    "_interval_paths_for_work_unit",
    "_reducer_target_key",
    "_sort_work_units_by_target",
    "_unique_reducer_target_keys",
    "_work_unit_expected_interval_ids",
    "_work_unit_sort_key",
]


def _interval_paths_for_work_unit(work_unit: ResidualFieldWorkUnit) -> tuple[str, ...]:
    interval_paths = tuple(
        artifact.path
        for artifact in work_unit.source_artifacts
        if artifact.kind == "interval-precompute" and artifact.path is not None
    )
    if not interval_paths:
        raise ValueError("ResidualFieldWorkUnit is missing source interval artifact paths.")
    return interval_paths


def _interval_inputs_for_work_unit(
    work_unit: ResidualFieldWorkUnit,
    *,
    transient_interval_payloads: dict[int, object] | None,
):
    interval_ids = tuple(int(interval_id) for interval_id in (work_unit.interval_ids or ()))
    if work_unit.interval_id is not None and not interval_ids:
        interval_ids = (int(work_unit.interval_id),)
    if transient_interval_payloads:
        payload_source = transient_interval_payloads
        if all(int(interval_id) in payload_source for interval_id in interval_ids):
            values = tuple(payload_source[int(interval_id)] for interval_id in interval_ids)
            return values[0] if len(values) == 1 else values
    return _interval_paths_for_work_unit(work_unit)


def _reducer_target_key(work_unit: ResidualFieldWorkUnit) -> tuple[int, int | None]:
    return int(work_unit.chunk_id), (
        int(work_unit.partition_id) if work_unit.partition_id is not None else None
    )


def _unique_reducer_target_keys(
    work_units: list[ResidualFieldWorkUnit],
) -> list[tuple[int, int | None]]:
    return list(dict.fromkeys(_reducer_target_key(work_unit) for work_unit in work_units))


def _work_unit_expected_interval_ids(work_unit: ResidualFieldWorkUnit) -> tuple[int, ...]:
    if work_unit.interval_ids:
        return tuple(int(interval_id) for interval_id in work_unit.interval_ids)
    if work_unit.interval_id is None:
        return ()
    return (int(work_unit.interval_id),)


def _work_unit_sort_key(work_unit: ResidualFieldWorkUnit) -> tuple:
    interval_ids = _work_unit_expected_interval_ids(work_unit)
    return (
        _reducer_target_key(work_unit),
        interval_ids[0] if interval_ids else -1,
        interval_ids[-1] if interval_ids else -1,
        int(len(interval_ids)),
    )


def _sort_work_units_by_target(
    work_units: list[ResidualFieldWorkUnit],
) -> list[ResidualFieldWorkUnit]:
    return sorted(work_units, key=_work_unit_sort_key)


def _hex64_or_digest(value: object, *, domain: str) -> str:
    text = "" if value is None else str(value)
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    return digest_dict({"value": text}, domain=domain)
