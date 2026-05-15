from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.fs_capability import (
    FilesystemCapabilityError,
    VisibilityProbeResult,
    cross_host_read_after_rename_probe,
    profile_output_filesystem,
)
from core.storage.fs_capability import read_fs_capability_manifest


def test_profile_output_filesystem_writes_manifest(tmp_path):
    manifest = profile_output_filesystem(tmp_path, run_digest="run123")

    assert manifest.run_digest == "run123"
    assert manifest.capabilities["same_directory_atomic_replace_visible"] is True
    assert manifest.capabilities["hardlink_required"] is False
    assert manifest.capabilities["symlink_required"] is False

    loaded = read_fs_capability_manifest(output_dir=tmp_path, run_digest="run123")
    assert loaded.capability_digest == manifest.capability_digest


def test_cross_host_visibility_probe_reports_hash_mismatch(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    class FakeClient:
        def run(self, func, path_text, expected_hash):
            return {
                "worker-a": {
                    "host": "host-a",
                    "ok": False,
                    "file_sha256": "bad",
                    "error": "hash mismatch",
                }
            }

    result = cross_host_read_after_rename_probe(
        path=path,
        expected_hash="expected",
        client=FakeClient(),
    )

    assert result == (
        VisibilityProbeResult(
            host="host-a",
            ok=False,
            file_sha256="bad",
            error="hash mismatch",
        ),
    )


def test_profile_output_filesystem_fails_when_required_cross_host_probe_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "core.runtime.fs_capability._client_worker_hosts",
        lambda client: ("host-a", "host-b"),
    )
    monkeypatch.setattr(
        "core.runtime.fs_capability.cross_host_read_after_rename_probe",
        lambda **kwargs: (
            VisibilityProbeResult(
                host="host-b",
                ok=False,
                file_sha256=None,
                error="not visible",
            ),
        ),
    )

    with pytest.raises(FilesystemCapabilityError, match="cross-host read-after-rename"):
        profile_output_filesystem(
            tmp_path,
            run_digest="run123",
            client=object(),
            require_cross_host=True,
        )
