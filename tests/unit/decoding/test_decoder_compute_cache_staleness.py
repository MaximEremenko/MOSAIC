"""Compute-mode decoder cache must not outlive its residual source data.

The compute-mode cache filename encodes only the configuration hash, so a
recomputed residual field in the same compute directory used to be paired
silently with a decoder trained on the old data. Reuse now requires the
recorded residual-source identity sidecar to match the artifacts on disk.
"""
from __future__ import annotations

import os

import numpy as np

from core.decoding.decoder_cache import (
    decoder_cache_source_identity_path,
    load_decoder_cache_source_identity,
    resolve_local_residual_source_identity,
    save_decoder_cache_source_identity,
)
from core.decoding.decoder_training import (
    _compute_decoder_cache_matches_residual_source,
)


class _NoopLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _write_residual_artifacts(source_dir, *, seed=0):
    rng = np.random.default_rng(seed)
    (source_dir / "residual_chunk_0_amplitudes.hdf5").write_bytes(
        rng.bytes(64)
    )
    (source_dir / "residual_chunk_0_amplitudes_av.hdf5").write_bytes(
        rng.bytes(64)
    )


def _cache_with_identity(tmp_path, source_dir):
    cache_path = str(tmp_path / "decoder_M_cache.npz")
    identity = resolve_local_residual_source_identity(output_dir=source_dir)
    assert identity is not None
    save_decoder_cache_source_identity(cache_path, identity, _NoopLogger())
    return cache_path


def test_sidecar_round_trip(tmp_path):
    source_dir = tmp_path / "processed_point_data"
    source_dir.mkdir()
    _write_residual_artifacts(source_dir)
    cache_path = _cache_with_identity(tmp_path, source_dir)
    assert decoder_cache_source_identity_path(cache_path).is_file()
    recorded = load_decoder_cache_source_identity(cache_path)
    assert recorded is not None
    assert recorded["source_identity_digest"]


def test_reuse_allowed_when_identity_matches(tmp_path):
    source_dir = tmp_path / "processed_point_data"
    source_dir.mkdir()
    _write_residual_artifacts(source_dir)
    cache_path = _cache_with_identity(tmp_path, source_dir)
    assert _compute_decoder_cache_matches_residual_source(
        cache_path, source_dir=source_dir, logger=_NoopLogger()
    )


def test_reuse_refused_when_residual_data_changed(tmp_path):
    source_dir = tmp_path / "processed_point_data"
    source_dir.mkdir()
    _write_residual_artifacts(source_dir, seed=0)
    cache_path = _cache_with_identity(tmp_path, source_dir)
    # Residual recompute: same file names, different content and mtime.
    _write_residual_artifacts(source_dir, seed=1)
    artifact = source_dir / "residual_chunk_0_amplitudes.hdf5"
    stat = artifact.stat()
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    assert not _compute_decoder_cache_matches_residual_source(
        cache_path, source_dir=source_dir, logger=_NoopLogger()
    )


def test_reuse_refused_when_sidecar_missing(tmp_path):
    source_dir = tmp_path / "processed_point_data"
    source_dir.mkdir()
    _write_residual_artifacts(source_dir)
    cache_path = str(tmp_path / "decoder_M_cache.npz")
    assert not _compute_decoder_cache_matches_residual_source(
        cache_path, source_dir=source_dir, logger=_NoopLogger()
    )


def test_reuse_allowed_when_no_residual_artifacts_to_verify(tmp_path):
    source_dir = tmp_path / "processed_point_data"
    source_dir.mkdir()
    cache_path = str(tmp_path / "decoder_M_cache.npz")
    assert _compute_decoder_cache_matches_residual_source(
        cache_path, source_dir=source_dir, logger=_NoopLogger()
    )
