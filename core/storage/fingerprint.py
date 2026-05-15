from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.storage.digests import canonical_json


def _little_endian_contiguous(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise TypeError("Object dtype arrays are not valid durable payload fingerprint inputs.")
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
    elif arr.dtype.byteorder == "=":
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(arr)


def canonical_array_bytes(array: np.ndarray) -> bytes:
    return _little_endian_contiguous(np.asarray(array)).tobytes(order="C")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(
    *,
    schema: str,
    expected_set_digest: str,
    datasets: Mapping[str, np.ndarray],
    attrs: Mapping[str, Any] | None = None,
) -> str:
    normalized_datasets = {
        str(name): _little_endian_contiguous(np.asarray(value))
        for name, value in datasets.items()
    }
    header = {
        "schema": str(schema),
        "expected_set_digest": str(expected_set_digest),
        "attrs": attrs or {},
        "datasets": [
            {
                "name": name,
                "shape": list(normalized_datasets[name].shape),
                "dtype": normalized_datasets[name].dtype.str,
            }
            for name in sorted(normalized_datasets)
        ],
    }
    digest = hashlib.sha256(canonical_json(header).encode("utf-8"))
    for name in sorted(normalized_datasets):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized_datasets[name].tobytes(order="C"))
    return digest.hexdigest()


__all__ = ["canonical_array_bytes", "file_sha256", "payload_sha256"]
