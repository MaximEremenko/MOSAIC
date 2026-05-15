from __future__ import annotations

import pytest

from core.residual_field.commit import (
    require_residual_no_output_manifest,
    write_residual_no_output_manifest,
)


def test_empty_replacement_requires_run_scoped_no_output_manifest(tmp_path):
    stale = tmp_path / "residual_chunk_5_amplitudes.hdf5"
    stale.write_text("stale residual output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Stale residual_chunk"):
        require_residual_no_output_manifest(
            output_dir=tmp_path,
            run_digest="run123",
        )

    manifest = write_residual_no_output_manifest(
        output_dir=tmp_path,
        run_digest="run123",
        source_scattering_commit_digest="a" * 64,
    )
    loaded = require_residual_no_output_manifest(
        output_dir=tmp_path,
        run_digest="run123",
    )

    assert loaded == manifest
    assert loaded.reason == "empty replacement coverage"
    assert (tmp_path / ".mosaic" / "runs" / "run123" / "residual_field" / "no_output.json").exists()
