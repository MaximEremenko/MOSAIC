from __future__ import annotations

import pytest

from core.runtime.quiescence import require_chunk_quiescence
from core.storage.attempt_store import attempt_manifest_path


class FakeFuture:
    def __init__(self, status="finished"):
        self.status = status

    def done(self):
        return True

    def exception(self):
        return RuntimeError("boom") if self.status == "error" else None


def test_quiescence_requires_attempt_manifest_for_each_work_unit(tmp_path):
    digest = "a" * 64
    path = attempt_manifest_path(
        tmp_path,
        "run123",
        "scattering",
        7,
        digest,
        "attempt-1",
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    report = require_chunk_quiescence(
        [FakeFuture()],
        client=None,
        output_dir=tmp_path,
        run_digest="run123",
        stage="scattering",
        chunk_id=7,
        expected_work_unit_digests=(digest,),
    )

    assert report.terminal_futures == 1
    assert report.discovered_attempt_manifests == 1


def test_quiescence_fails_when_attempt_manifest_missing(tmp_path):
    with pytest.raises(RuntimeError, match="missing attempt manifests"):
        require_chunk_quiescence(
            [FakeFuture()],
            client=None,
            output_dir=tmp_path,
            run_digest="run123",
            stage="scattering",
            chunk_id=7,
            expected_work_unit_digests=("a" * 64,),
        )


def test_quiescence_fails_on_terminal_failed_future(tmp_path):
    digest = "a" * 64
    path = attempt_manifest_path(
        tmp_path,
        "run123",
        "scattering",
        7,
        digest,
        "attempt-1",
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed future"):
        require_chunk_quiescence(
            [FakeFuture(status="error")],
            client=None,
            output_dir=tmp_path,
            run_digest="run123",
            stage="scattering",
            chunk_id=7,
            expected_work_unit_digests=(digest,),
        )
