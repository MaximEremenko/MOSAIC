"""Atomic HDF5 payload writer.

This is a storage primitive shared by every Map-Reduce stage that writes
immutable payloads (scattering, residual field). It lives in ``core.storage``
so stage modules do not have to import each other's private helpers, which
would create a ``scattering.artifacts`` <-> ``scattering.commit`` import cycle
and a cross-stage dependency from ``residual_field.commit`` into
``scattering.artifacts``.

The write protocol matches the manifest write protocol in
:mod:`core.storage.atomic`: write a sibling temp file, fsync it, reopen and
validate datasets, atomically rename into place, then fsync the parent
directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from core.storage.atomic import fsync_parent, fsync_path

__all__ = ["atomic_hdf5_write"]


def atomic_hdf5_write(
    out_path: Path,
    datasets: dict[str, np.ndarray],
    *,
    attrs: dict[str, object] | None = None,
) -> None:
    """Write ``datasets`` to ``out_path`` atomically.

    The payload is materialized in a sibling temp file, fsynced, reopened and
    shape-validated, then renamed over ``out_path``. A crash before the rename
    leaves only the temp file, which callers treat as an incomplete artifact.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with h5py.File(temp_path, "w") as h5file:
            for name, values in datasets.items():
                data = np.asarray(values)
                h5file.create_dataset(name, data=data)
            if attrs:
                for name, value in attrs.items():
                    h5file.attrs[name] = value
            h5file.flush()
        fsync_path(temp_path)
        with h5py.File(temp_path, "r") as h5file:
            for name, expected in datasets.items():
                if name not in h5file:
                    raise OSError(f"HDF5 validation failed: missing dataset {name!r}")
                if h5file[name].shape != np.asarray(expected).shape:
                    raise OSError(
                        f"HDF5 validation failed for {name!r}: "
                        f"{h5file[name].shape} != {np.asarray(expected).shape}"
                    )
        os.replace(temp_path, out_path)
        fsync_parent(out_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
