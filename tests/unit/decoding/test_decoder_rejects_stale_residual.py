from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.decoding.decoder_cache import (
    build_decoder_cache_path,
    resolve_current_residual_source_identity,
    save_decoder_cache,
)


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_current_source_rejects_stale_residual_before_decoder_cache_use(tmp_path):
    params = {"supercell": [1], "points": [0]}
    cache_path = build_decoder_cache_path(params, str(tmp_path))
    save_decoder_cache(cache_path, np.eye(1), 1, logger=_NoopLogger())
    (tmp_path / "residual_chunk_0_amplitudes.hdf5").write_text(
        "stale residual data",
        encoding="utf-8",
    )

    assert Path(cache_path).exists()
    with pytest.raises(RuntimeError, match="Loose residual chunk"):
        resolve_current_residual_source_identity(
            output_dir=tmp_path,
            run_digest="run123",
        )
