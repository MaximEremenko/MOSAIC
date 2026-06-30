"""Shared low-level helpers for scattering and residual-field commit payloads.

These four private helpers were previously duplicated verbatim (modulo error-
message strings) in core/scattering/commit.py and core/residual_field/commit.py.
They are lifted here so both modules import from one place.

Rules that MUST be preserved:
- ``_payload_digest`` must produce BYTE-IDENTICAL ``payload_sha256`` values to
  the originals.  The only semantic change is that the caller now supplies
  ``schema`` explicitly instead of having the function read it from
  ``attrs["schema"]`` (scattering) or receive it as a parameter (residual).
  Both callers already had the value in scope; scattering now passes
  ``str(attrs["schema"])`` and residual continues to pass its named param.
- ``_read_payload``, ``_resolve_runtime_provenance``, and ``_payload_datasets``
  were byte-for-byte identical between the two modules (the only difference was
  error-message wording in ``_payload_datasets``).  The scattering wording is
  used here; residual field previously had slightly different strings but the
  *behaviour* was identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from core.storage.fingerprint import payload_sha256
from core.runtime.gpu_admission import runtime_provenance_for_attempt


def _payload_datasets(
    *,
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
) -> dict[str, np.ndarray]:
    delta = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
    average = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
    point_id_arr = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if average.shape != delta.shape:
        raise ValueError("amplitudes_delta and amplitudes_average must have matching shapes.")
    if point_id_arr.shape != delta.shape:
        raise ValueError("point_ids must align with amplitude arrays.")
    return {
        "point_ids": point_id_arr,
        "grid_shape_nd": np.asarray(grid_shape_nd, dtype=np.int64),
        "amplitudes_delta": delta,
        "amplitudes_average": average,
    }


def _read_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as h5file:
        datasets = {name: np.asarray(h5file[name]) for name in h5file.keys()}
        attrs = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in h5file.attrs.items()
        }
    return datasets, attrs


def _resolve_runtime_provenance(
    runtime_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = dict(runtime_provenance or {})
    return runtime_provenance_for_attempt(
        fs_capability_digest=base.get("fs_capability_digest"),
        scheduler_kind=str(base.get("scheduler_kind", "local")),
        nufft_policy=base.get("nufft_policy", "auto"),
        resource_requirements=base.get("resource_requirements"),
        cuda_probe=str(base.get("nufft_policy", "auto")) in {"gpu-required", "allow-fallback"},
    )


def _payload_digest(
    *,
    schema: str,
    expected_set_digest: str,
    datasets: Mapping[str, np.ndarray],
    attrs: Mapping[str, Any],
) -> str:
    semantic_attrs = {
        key: value
        for key, value in attrs.items()
        if key not in {"attempt_id"}
    }
    return payload_sha256(
        schema=schema,
        expected_set_digest=str(expected_set_digest),
        datasets=datasets,
        attrs=semantic_attrs,
    )


__all__ = [
    "_payload_datasets",
    "_payload_digest",
    "_read_payload",
    "_resolve_runtime_provenance",
]
