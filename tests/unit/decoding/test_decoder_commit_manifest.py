from __future__ import annotations

import numpy as np

from core.decoding.commit import decoder_commit_path, write_decoder_commit
from core.decoding.decoder_cache import save_decoder_cache


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_decoder_commit_manifest_records_cache_identity_and_file_hash(tmp_path):
    cache_path = tmp_path / "decoder_M_test.npz"
    cache_identity = {
        "schema": "mosaic.decoder.cache_identity",
        "schema_version": 1,
        "decoder_cache_digest": "d" * 64,
    }
    save_decoder_cache(str(cache_path), np.eye(2), 2, logger=_NoopLogger())

    manifest = write_decoder_commit(
        output_dir=tmp_path,
        run_digest="run123",
        decoder_cache_path=cache_path,
        decoder_cache_identity=cache_identity,
    )
    repeat = write_decoder_commit(
        output_dir=tmp_path,
        run_digest="run123",
        decoder_cache_path=cache_path,
        decoder_cache_identity=cache_identity,
    )

    assert repeat == manifest
    assert manifest.decoder_cache_path == "decoder_M_test.npz"
    assert manifest.decoder_cache_identity == cache_identity
    assert len(manifest.decoder_cache_file_sha256) == 64
    assert decoder_commit_path(tmp_path, "run123").exists()
