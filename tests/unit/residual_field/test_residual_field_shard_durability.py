from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.residual_field.artifacts import (
    load_residual_field_shard_payload,
    persist_residual_field_generation_checkpoint,
    persist_residual_field_shard_checkpoint,
)
from core.residual_field.contracts import ResidualFieldWorkUnit


def _work_unit(tmp_path: Path) -> ResidualFieldWorkUnit:
    return ResidualFieldWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
    )


def test_residual_shard_scratch_publish_writes_durable_payload_before_manifest(tmp_path):
    scratch_root = tmp_path / "scratch"

    manifest = persist_residual_field_shard_checkpoint(
        _work_unit(tmp_path),
        grid_shape_nd=np.array([[2]]),
        total_reciprocal_points=11,
        contribution_reciprocal_points=5,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path),
        scratch_root=str(scratch_root),
        quiet_logs=True,
    )

    payload_path = Path(manifest.artifacts[0].path)
    manifest_path = Path(manifest.artifacts[1].path)
    assert payload_path.exists()
    assert manifest_path.exists()
    payload = load_residual_field_shard_payload(manifest)
    np.testing.assert_array_equal(payload["point_ids"], np.array([10, 11]))
    assert list((scratch_root / "residual_checkpoints" / "chunk_3").glob("*")) == []


def test_residual_shard_manifest_is_not_published_when_durable_replace_fails(
    tmp_path,
    monkeypatch,
):
    real_replace = __import__("os").replace

    def fail_durable_payload_replace(src, dst):
        dst_path = Path(dst)
        if (
            dst_path.suffix == ".hdf5"
            and "residual_checkpoints" in dst_path.parts
            and "scratch" not in dst_path.parts
        ):
            raise OSError("replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr("core.residual_field.artifacts.os.replace", fail_durable_payload_replace)

    with pytest.raises(OSError, match="replace failed"):
        persist_residual_field_shard_checkpoint(
            _work_unit(tmp_path),
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            point_ids=np.array([10, 11]),
            output_dir=str(tmp_path),
            scratch_root=str(tmp_path / "scratch"),
            quiet_logs=True,
        )

    shard_dir = tmp_path / "residual_checkpoints" / "chunk_3"
    assert list(shard_dir.glob("*.manifest.json")) == []


def test_residual_generation_checkpoint_uses_same_durable_publish_order(tmp_path):
    scratch_root = tmp_path / "scratch"

    manifest = persist_residual_field_generation_checkpoint(
        chunk_id=3,
        parameter_digest="abc123",
        partition_id=None,
        generation_seq=1,
        incorporated_interval_ids=(7, 8),
        grid_shape_nd=np.array([[2]]),
        reciprocal_point_count=5,
        total_reciprocal_points=11,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        point_ids=np.array([10, 11]),
        output_dir=str(tmp_path),
        scratch_root=str(scratch_root),
        quiet_logs=True,
    )

    assert Path(manifest.artifacts[0].path).exists()
    assert Path(manifest.artifacts[1].path).exists()
    assert list((scratch_root / "residual_checkpoints" / "chunk_3").glob("*")) == []
