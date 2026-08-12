from __future__ import annotations

from core.runtime.quiescence import require_chunk_quiescence
from core.storage.attempt_store import attempt_manifest_path


def test_retry_recovery_requires_manifest_after_terminal_attempts(tmp_path):
    digest = "b" * 64
    path = attempt_manifest_path(
        tmp_path,
        "run123",
        "residual_field",
        3,
        digest,
        "attempt-retry-2",
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    report = require_chunk_quiescence(
        [],
        client=None,
        output_dir=tmp_path,
        run_digest="run123",
        stage="residual_field",
        chunk_id=3,
        expected_work_unit_digests=(digest,),
    )

    assert report.discovered_attempt_manifests == 1
