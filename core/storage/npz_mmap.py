"""Zero-copy reads of arrays stored inside uncompressed ``.npz`` archives.

``np.savez`` writes members with ``ZIP_STORED``, so every embedded ``.npy``
sits contiguously in the file and can be memory-mapped read-only at its
offset. Mapped members cost evictable page cache instead of anonymous RSS —
materializing multi-GB members with ``np.asarray`` is what OOM-killed the
hkl40 finalize driver twice before this existed.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np


def mmap_npz_member(npz_path: Path | str, member: str) -> np.ndarray | None:
    """Memory-map one array inside an uncompressed ``.npz``.

    Returns ``None`` when the member is compressed, Fortran-ordered, an
    object array, or otherwise unmappable — callers fall back to a
    materializing load."""
    npz_path = Path(npz_path)
    try:
        with zipfile.ZipFile(npz_path) as archive:
            info = archive.getinfo(member + ".npy")
            if info.compress_type != zipfile.ZIP_STORED:
                return None
            with archive.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    return None
                npy_header_bytes = handle.tell()
            if fortran or dtype.hasobject:
                return None
        # Absolute payload offset = zip LOCAL header (30 fixed bytes + name +
        # extra, read from the file itself: the local extra field can differ
        # from the central directory's) + the .npy header just measured.
        with open(npz_path, "rb") as raw:
            raw.seek(info.header_offset)
            local_header = raw.read(30)
            if local_header[:4] != b"PK\x03\x04":
                return None
            name_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")
        data_offset = info.header_offset + 30 + name_len + extra_len + npy_header_bytes
        return np.memmap(
            npz_path, mode="r", dtype=dtype, shape=shape, offset=data_offset
        )
    except (KeyError, OSError, ValueError):
        return None


__all__ = ["mmap_npz_member"]
