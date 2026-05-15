from __future__ import annotations

from core.runtime.fs_capability import cross_host_read_after_rename_probe
from core.storage.fingerprint import file_sha256


def test_cross_host_visibility_probe_uses_client_run(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    expected = file_sha256(path)

    class FakeClient:
        def run(self, func, path_text, expected_hash):
            return {
                "worker-a": func(path_text, expected_hash),
                "worker-b": func(path_text, expected_hash),
            }

    results = cross_host_read_after_rename_probe(
        path=path,
        expected_hash=expected,
        client=FakeClient(),
    )

    assert len(results) == 2
    assert all(result.ok for result in results)
    assert {result.file_sha256 for result in results} == {expected}
