from __future__ import annotations

import pytest

from core.storage.attempt_store import (
    attempt_manifest_path,
    attempt_payload_path,
    relative_to_output,
    work_unit_attempts_root,
)


def test_attempt_directory_paths_are_sharded_by_work_unit_digest_prefix(tmp_path):
    digest = "abcdef123456"
    root = work_unit_attempts_root(
        tmp_path,
        "run123",
        "scattering",
        3,
        digest,
    )
    payload = attempt_payload_path(
        tmp_path,
        "run123",
        "scattering",
        3,
        digest,
        "try1",
    )
    manifest = attempt_manifest_path(
        tmp_path,
        "run123",
        "scattering",
        3,
        digest,
        "try1",
    )

    assert relative_to_output(root, output_dir=tmp_path) == (
        ".mosaic/runs/run123/scattering/chunks/chunk_3/attempts/ab/abcdef123456"
    )
    assert relative_to_output(payload, output_dir=tmp_path) == (
        ".mosaic/runs/run123/scattering/chunks/chunk_3/"
        "attempts/ab/abcdef123456/attempt_try1/payload.hdf5"
    )
    assert relative_to_output(manifest, output_dir=tmp_path) == (
        ".mosaic/runs/run123/scattering/chunks/chunk_3/"
        "attempts/ab/abcdef123456/attempt_try1/attempt.json"
    )


def test_attempt_sharding_rejects_short_or_unsafe_digest(tmp_path):
    with pytest.raises(ValueError, match="at least two characters"):
        work_unit_attempts_root(tmp_path, "run123", "scattering", 3, "a")

    with pytest.raises(ValueError, match="invalid path segment"):
        work_unit_attempts_root(tmp_path, "run123", "scattering", 3, "../abcdef")
