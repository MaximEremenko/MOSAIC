"""The one on-disk format for a stage-1 interval payload.

Both durable modes persist the SAME object — an :class:`IntervalTask` —
and used to do it two different ways: per-interval HDF5 artifacts for the
default (precompute) mode, and an npz store with a JSON meta member for
streaming mode. Two writers, two readers, two commit protocols and two
skip-checks for one payload meant every integrity fix had to be made
twice, and a mode switch recomputed physics the other mode already had on
disk. This module owns the format; the artifact store and the streaming
store are callers.

Reads may be memory-mapped: the datasets are written contiguous and
uncompressed, so a reader can map them at their HDF5 offset and pay
evictable page cache instead of anonymous RSS (what the streaming store's
npz mmap did, kept).
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from core.scattering.half_space import normalize_half_space_role
from core.scattering.kernels import IntervalTask
from core.storage.hdf5_atomic import atomic_hdf5_write

logger = logging.getLogger(__name__)

INTERVAL_PAYLOAD_FORMAT = "mosaic.scattering.interval"
INTERVAL_PAYLOAD_SCHEMA_VERSION = 2

# Returned when nothing durable exists yet — distinct from ``None``, which
# records an interval the mask emptied (a real, reusable answer).
PAYLOAD_MISS = object()

_ARRAY_MEMBERS = ("q_grid", "q_amp", "q_amp_av")


def _q_grid_digest(q_grid) -> str:
    from core.scattering.artifacts import _q_grid_digest as _digest

    return _digest(q_grid)


def interval_payload_datasets(task: IntervalTask) -> dict[str, np.ndarray]:
    """The dataset map for one payload, in the historical artifact layout."""
    return {
        "irecip_id": np.array([int(task.irecip_id)], dtype=np.int64),
        "element": np.asarray(
            str(task.element), dtype=h5py.string_dtype("utf-8")
        ),
        "q_grid": np.ascontiguousarray(task.q_grid, dtype=np.float64),
        "q_amp": np.ascontiguousarray(
            np.asarray(task.q_amp).reshape(-1), dtype=np.complex128
        ),
        "q_amp_av": np.ascontiguousarray(
            np.asarray(task.q_amp_av).reshape(-1), dtype=np.complex128
        ),
        "q_grid_digest": np.asarray(
            task.q_grid_digest
            if task.q_grid_digest
            else _q_grid_digest(task.q_grid),
            dtype=h5py.string_dtype("ascii"),
        ),
        "half_space_role": np.asarray(
            str(task.half_space_role), dtype=h5py.string_dtype("ascii")
        ),
        "reciprocal_multiplicity": np.array(
            [int(task.reciprocal_multiplicity)], dtype=np.int64
        ),
    }


def write_interval_payload(
    path: Path | str,
    task: IntervalTask | None,
    *,
    interval_id: int | None = None,
) -> None:
    """Persist one payload atomically (temp file -> fsync -> validate ->
    rename). ``task=None`` records a mask-emptied interval, so its
    emptiness is durable too and never re-derived."""
    path = Path(path)
    if task is None:
        if interval_id is None:
            raise ValueError("Recording an empty interval requires interval_id.")
        atomic_hdf5_write(
            path,
            {"irecip_id": np.array([int(interval_id)], dtype=np.int64)},
            attrs={
                "schema_version": INTERVAL_PAYLOAD_SCHEMA_VERSION,
                "interval_id": int(interval_id),
                "format": INTERVAL_PAYLOAD_FORMAT,
                "empty": True,
            },
        )
        return
    atomic_hdf5_write(
        path,
        interval_payload_datasets(task),
        attrs={
            "schema_version": INTERVAL_PAYLOAD_SCHEMA_VERSION,
            "interval_id": int(task.irecip_id),
            "format": INTERVAL_PAYLOAD_FORMAT,
            "empty": False,
        },
    )


def mmap_h5_dataset(path: Path | str, member: str) -> np.ndarray | None:
    """Memory-map one contiguous, uncompressed HDF5 dataset.

    Returns ``None`` when the dataset is chunked, compressed or otherwise
    unmappable — callers fall back to a materializing read."""
    try:
        with h5py.File(path, "r") as handle:
            if member not in handle:
                return None
            dataset = handle[member]
            if dataset.chunks is not None or dataset.compression is not None:
                return None
            offset = dataset.id.get_offset()
            if offset is None:  # not yet allocated (empty dataset)
                return None
            shape, dtype = dataset.shape, dataset.dtype
        if dtype.hasobject:
            return None
        return np.memmap(
            path, dtype=dtype, mode="r", offset=int(offset), shape=shape
        )
    except Exception:
        return None


def read_interval_payload(
    path: Path | str,
    *,
    mmap: bool = False,
):
    """Load one payload.

    Returns an :class:`IntervalTask`, ``None`` for a recorded mask-empty
    interval, or :data:`PAYLOAD_MISS` when the file is absent. Raises for a
    present-but-invalid payload — callers that treat corruption as absence
    (the streaming store self-heals by recomputing) catch it themselves."""
    path = Path(path)
    if not path.exists():
        return PAYLOAD_MISS
    with h5py.File(path, "r") as data:
        if bool(data.attrs.get("empty", False)):
            return None
        if "half_space_role" not in data or "reciprocal_multiplicity" not in data:
            raise ValueError(
                "Interval payloads must include half_space_role and "
                "reciprocal_multiplicity metadata."
            )
        element = data["element"][()]
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        q_grid_digest = None
        if "q_grid_digest" in data:
            q_grid_digest = data["q_grid_digest"][()]
            if isinstance(q_grid_digest, bytes):
                q_grid_digest = q_grid_digest.decode("ascii")
            else:
                q_grid_digest = str(q_grid_digest)
        half_space_role = normalize_half_space_role(data["half_space_role"][()])
        reciprocal_multiplicity = int(
            np.asarray(data["reciprocal_multiplicity"]).reshape(-1)[0]
        )
        irecip_id = int(np.asarray(data["irecip_id"]).reshape(-1)[0])
        arrays = {}
        if not mmap:
            for member in _ARRAY_MEMBERS:
                arrays[member] = np.asarray(data[member])
    if mmap:
        for member in _ARRAY_MEMBERS:
            mapped = mmap_h5_dataset(path, member)
            if mapped is None:
                with h5py.File(path, "r") as data:
                    mapped = np.asarray(data[member])
            arrays[member] = mapped
    return IntervalTask(
        irecip_id,
        str(element),
        arrays["q_grid"],
        arrays["q_amp"],
        arrays["q_amp_av"],
        q_grid_digest=q_grid_digest,
        half_space_role=half_space_role,
        reciprocal_multiplicity=reciprocal_multiplicity,
    )


__all__ = [
    "INTERVAL_PAYLOAD_FORMAT",
    "INTERVAL_PAYLOAD_SCHEMA_VERSION",
    "PAYLOAD_MISS",
    "interval_payload_datasets",
    "mmap_h5_dataset",
    "read_interval_payload",
    "write_interval_payload",
]
