from __future__ import annotations

import json

import pytest

from core.storage.fs_capability import (
    build_fs_capability_digest,
    read_fs_capability_manifest,
    write_fs_capability_manifest,
)
from core.storage.manifest import ManifestError, validate_manifest_payload
from core.storage.public_manifest import (
    build_public_manifest_digest,
    read_public_manifest,
    write_public_manifest,
)


def test_fs_capability_manifest_roundtrip_and_digest(tmp_path):
    manifest = write_fs_capability_manifest(
        output_dir=tmp_path,
        run_digest="run123",
        capabilities={"atomic_replace": True, "durable_directory_fsync": False},
    )

    loaded = read_fs_capability_manifest(output_dir=tmp_path, run_digest="run123")

    assert loaded == manifest
    assert loaded.capability_digest == build_fs_capability_digest(
        run_digest="run123",
        capabilities={"atomic_replace": True, "durable_directory_fsync": False},
    )


def test_fs_capability_manifest_rejects_digest_mismatch(tmp_path):
    write_fs_capability_manifest(
        output_dir=tmp_path,
        run_digest="run123",
        capabilities={"atomic_replace": True},
    )
    path = tmp_path / ".mosaic" / "runs" / "run123" / "fs_capability.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["capabilities"]["atomic_replace"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="digest"):
        read_fs_capability_manifest(output_dir=tmp_path, run_digest="run123")


def test_public_manifest_roundtrip_and_schema_rejects_extra_fields(tmp_path):
    source_identity = {
        "residual_stage_digest": "a" * 64,
        "residual_payload_hashes": [],
        "upstream_scattering_identity": [],
    }
    published_file_records = (
        {
            "path": "processed_point_data/manifest.json",
            "stage": "residual_field",
            "kind": "manifest",
            "chunk_id": None,
            "file_sha256": "a" * 64,
            "payload_nbytes": 12,
        },
    )
    manifest = write_public_manifest(
        tmp_path / "public_manifest.json",
        output_dir=tmp_path,
        run_digest="run123",
        source_stage="residual_field",
        source_identity=source_identity,
        published_files=("processed_point_data/manifest.json",),
        published_file_records=published_file_records,
    )

    loaded = read_public_manifest(tmp_path / "public_manifest.json", output_dir=tmp_path)

    assert loaded == manifest
    assert loaded.public_manifest_digest == build_public_manifest_digest(
        run_digest="run123",
        source_stage="residual_field",
        source_identity=source_identity,
        published_files=("processed_point_data/manifest.json",),
        published_file_records=published_file_records,
    )
    with pytest.raises(ManifestError, match="unexpected"):
        validate_manifest_payload(
            {
                **manifest.to_payload(),
                "extra": True,
            },
            expected_schema="mosaic.public_manifest",
            expected_version=1,
            output_dir=tmp_path,
        )


def test_public_manifest_rejects_published_files_without_records(tmp_path):
    with pytest.raises(ManifestError, match="published_files"):
        write_public_manifest(
            tmp_path / "public_manifest.json",
            output_dir=tmp_path,
            run_digest="run123",
            source_stage="residual_field",
            source_identity={
                "residual_stage_digest": "a" * 64,
                "residual_payload_hashes": [],
                "upstream_scattering_identity": [],
            },
            published_files=("residual_chunk_0_amplitudes.hdf5",),
        )
