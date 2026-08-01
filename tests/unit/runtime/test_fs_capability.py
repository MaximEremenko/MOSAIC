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


def test_file_lock_probe_passes_locally_and_is_recorded(tmp_path):
    from core.runtime.fs_capability import _probe_file_lock

    result = _probe_file_lock(str(tmp_path / "lock_probe.dat"))
    assert result["ok"] is True

    manifest = profile_output_filesystem(tmp_path, run_digest="run123")
    assert manifest.capabilities["file_lock_functional"] is True
    assert manifest.capabilities["file_lock_required"] is False


def test_multi_node_fails_closed_when_locking_unavailable(tmp_path, monkeypatch):
    """The one capability the multi-node reducer commit depends on — a
    mount without functional flock must abort at profile time, not corrupt
    the reducer-progress manifest hours later."""
    monkeypatch.setattr(
        "core.runtime.fs_capability.cross_host_file_lock_probe",
        lambda **kwargs: (
            {"host": "host-b", "ok": False, "error": "ENOLCK"},
        ),
    )
    with pytest.raises(FilesystemCapabilityError, match="file locking unavailable"):
        profile_output_filesystem(
            tmp_path,
            run_digest="run123",
            require_cross_host=True,
        )
