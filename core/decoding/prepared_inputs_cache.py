"""Per-chunk prepared decoder inputs, carried from training to decode.

decoder.source='current' runs the whole per-chunk input pipeline twice:
the training pass computes features for every site and keeps only the
training subset; the decode pass then redoes the 6.78 GB residual read,
grid regeneration, groups build and feature extraction (~10+ min per
hkl40 case). This cache hands the training pass's features to the decode
pass instead.

Correctness stance: validity is checked with a token over (chunk_id,
output_dir, site count, ordered central point ids) and entries are
pop-on-read — any mismatch is a plain cache miss and the decode pass
falls back to the full recompute. Fail-safe, never fail-wrong. The token
is never persisted, never enters any digest, and never names a file.

Leaf module: imports only stdlib/numpy + core.runtime.cpu_resources
(the decoder import-cycle guards scan for core.decoding.displacement*).
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedChunkInputs:
    output_dir: str
    token: str
    cids_all: list
    decoder_keys_all: list
    features_all: Sequence
    nbytes: int


def prepared_inputs_token(chunk_id, point_data_list, output_dir) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(chunk_id)).encode())
    digest.update(str(output_dir).encode())
    digest.update(str(len(point_data_list)).encode())
    for point_data in point_data_list:
        digest.update(str(int(point_data["central_point_id"])).encode())
    return digest.hexdigest()


def prepared_cache_max_bytes() -> int:
    raw = os.getenv("MOSAIC_DECODER_PREPARED_CACHE_MAX_BYTES")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    from core.runtime.cpu_resources import available_memory_bytes

    available = available_memory_bytes()
    if available is None:
        return 8 << 30
    return int(min(0.25 * available, 32 << 30))


class PreparedInputsCache:
    """RAM-budgeted, disk-spilling, pop-on-read cache of prepared inputs."""

    def __init__(self, *, max_bytes: int, spill_dir: str | None, logger=logger):
        self.max_bytes = int(max_bytes)
        self.spill_dir = (
            Path(spill_dir) / ".decoder_prepared_cache" if spill_dir else None
        )
        self.logger = logger
        self._entries: dict[int, PreparedChunkInputs] = {}
        self._spilled: dict[int, tuple[PreparedChunkInputs, Path]] = {}
        self._ram_bytes = 0

    def put(self, chunk_id: int, entry: PreparedChunkInputs) -> bool:
        if self.max_bytes <= 0:
            return False
        chunk_id = int(chunk_id)
        if self._ram_bytes + entry.nbytes <= self.max_bytes:
            self._entries[chunk_id] = entry
            self._ram_bytes += int(entry.nbytes)
            return True
        # RAM budget exceeded: spill uniform-size float64 features to one
        # memmap (raw .npy bytes — values are untouched). Ragged families
        # (mixed-P) are refused; a refused put is just a later cache miss.
        features = list(entry.features_all)
        if (
            self.spill_dir is None
            or not features
            or any(
                not isinstance(f, np.ndarray)
                or f.dtype != np.float64
                or f.shape != features[0].shape
                for f in features
            )
        ):
            return False
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            path = self.spill_dir / f"chunk_{chunk_id}_features.npy"
            spilled = np.lib.format.open_memmap(
                str(path),
                mode="w+",
                dtype=np.float64,
                shape=(len(features),) + features[0].shape,
            )
            for row, feature in enumerate(features):
                spilled[row] = feature
            spilled.flush()
            del spilled
        except OSError:
            return False
        self._spilled[chunk_id] = (entry, path)
        return True

    def pop(
        self, chunk_id: int, *, token: str, output_dir: str
    ) -> PreparedChunkInputs | None:
        chunk_id = int(chunk_id)
        entry = self._entries.pop(chunk_id, None)
        if entry is not None:
            self._ram_bytes -= int(entry.nbytes)
        else:
            spilled = self._spilled.pop(chunk_id, None)
            if spilled is not None:
                meta, path = spilled
                try:
                    mapped = np.load(str(path), mmap_mode="r")
                except OSError:
                    mapped = None
                if mapped is not None:
                    entry = PreparedChunkInputs(
                        output_dir=meta.output_dir,
                        token=meta.token,
                        cids_all=meta.cids_all,
                        decoder_keys_all=meta.decoder_keys_all,
                        features_all=[mapped[row] for row in range(mapped.shape[0])],
                        nbytes=meta.nbytes,
                    )
                path.unlink(missing_ok=True)
        if entry is None:
            self.logger.info(
                "Prepared decoder inputs cache MISS for chunk %s (not cached).",
                chunk_id,
            )
            return None
        if entry.token != token or str(entry.output_dir) != str(output_dir):
            self.logger.info(
                "Prepared decoder inputs cache MISS for chunk %s "
                "(token/output_dir mismatch — falling back to full recompute).",
                chunk_id,
            )
            return None
        self.logger.info(
            "Prepared decoder inputs cache HIT for chunk %s (%d sites, %.2f GB).",
            chunk_id,
            len(entry.cids_all),
            entry.nbytes / 1e9,
        )
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self._ram_bytes = 0
        for _meta, path in self._spilled.values():
            path.unlink(missing_ok=True)
        self._spilled.clear()
        if self.spill_dir is not None:
            shutil.rmtree(self.spill_dir, ignore_errors=True)

    close = clear
