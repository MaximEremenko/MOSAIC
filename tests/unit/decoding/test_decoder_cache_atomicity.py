from __future__ import annotations

import json

import numpy as np
import pytest

from core.decoding.decoder_cache import (
    load_decoder_cache,
    save_decoder_cache,
    save_decoder_provenance,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[object, ...]] = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        self.warnings.append(args)


def test_save_decoder_cache_raises_and_preserves_existing_cache_on_write_failure(
    tmp_path,
    monkeypatch,
):
    cache_path = tmp_path / "decoder.npz"
    save_decoder_cache(str(cache_path), np.eye(2), 2, logger=_RecordingLogger())

    def fail_savez(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("core.decoding.decoder_cache.np.savez", fail_savez)

    with pytest.raises(OSError, match="disk full"):
        save_decoder_cache(str(cache_path), np.ones((2, 2)), 2, logger=_RecordingLogger())

    loaded, feature_dim = load_decoder_cache(str(cache_path), logger=_RecordingLogger())
    np.testing.assert_allclose(loaded, np.eye(2))
    assert feature_dim == 2


def test_save_decoder_provenance_validates_cache_before_publish(tmp_path):
    provenance_path = tmp_path / "decoder_source_provenance.json"
    provenance_path.write_text(
        json.dumps({"mode": "cache", "decoder_cache_path": "old.npz"}),
        encoding="utf-8",
    )
    corrupt_cache = tmp_path / "decoder.npz"
    corrupt_cache.write_bytes(b"not an npz")

    with pytest.raises(Exception):
        save_decoder_provenance(
            str(tmp_path),
            {"mode": "cache", "decoder_cache_path": str(corrupt_cache)},
            logger=_RecordingLogger(),
        )

    assert json.loads(provenance_path.read_text(encoding="utf-8")) == {
        "mode": "cache",
        "decoder_cache_path": "old.npz",
    }


def test_load_decoder_cache_preserves_corrupt_cache_warning_and_retrain_semantics(tmp_path):
    cache_path = tmp_path / "decoder.npz"
    cache_path.write_bytes(b"not an npz")
    logger = _RecordingLogger()

    decoder, feature_dim = load_decoder_cache(str(cache_path), logger=logger)

    assert decoder is None
    assert feature_dim is None
    assert logger.warnings
    assert "Will retrain" in str(logger.warnings[0])
