"""Streaming resume credit: COMMITTED reducer progress manifests must
survive the SQLite rebuild — streaming writes no payload manifests, so
without this source a completed case re-derives its whole residual on
retry. Manifests stay the authority: wrong digest, non-committed status,
pending work, or missing final artifacts all leave rows unsaved."""
import json
from pathlib import Path

from core.storage.run_state_cache import _committed_streaming_residual_credits

DIGEST = "abc123def456"


def _write_manifest(
    tmp_path: Path,
    chunk_id: int,
    *,
    digest: str = DIGEST,
    status: str = "committed",
    interval_ids=(1, 2, 3),
    pending_interval_ids=(),
    artifact_exists: bool = True,
) -> None:
    chunk_dir = tmp_path / "residual_checkpoints" / f"chunk_{chunk_id}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / f"residual_chunk_{chunk_id}_amplitudes.hdf5"
    if artifact_exists:
        artifact.write_bytes(b"x")
    manifest = {
        "schema_version": 1,
        "chunk_id": chunk_id,
        "parameter_digest": digest,
        "completion_status": status,
        "incorporated_interval_ids": list(interval_ids),
        "pending_interval_ids": list(pending_interval_ids),
        "pending_shard_keys": [],
        "final_artifacts": [
            {"key": "k", "kind": "chunk-residual-values", "path": str(artifact)}
        ],
    }
    (chunk_dir / f"reducer_progress_params_{digest}.manifest.json").write_text(
        json.dumps(manifest)
    )


def test_committed_manifests_credit_rows(tmp_path):
    _write_manifest(tmp_path, 0, interval_ids=(1, 2))
    _write_manifest(tmp_path, 1, interval_ids=(1, 2))
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == [
        (0, (1, 2)),
        (1, (1, 2)),
    ]


def test_wrong_digest_never_credits(tmp_path):
    _write_manifest(tmp_path, 0, digest="stalecfg0000")
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == []


def test_non_committed_or_pending_never_credits(tmp_path):
    _write_manifest(tmp_path, 0, status="materialized")
    _write_manifest(tmp_path, 1, pending_interval_ids=(7,))
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == []


def test_missing_final_artifact_never_credits(tmp_path):
    _write_manifest(tmp_path, 0, artifact_exists=False)
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == []


def test_rebuild_overlay_applies_credit(tmp_path, monkeypatch):
    from core.storage import run_state_cache as rsc
    from core.storage.database_manager import ManifestOnlyDatabaseManager

    _write_manifest(tmp_path, 0, interval_ids=(1, 2))
    db = ManifestOnlyDatabaseManager()
    db.cache_enabled = True
    for interval_id in (1, 2, 3):
        db.update_interval_chunk_status(interval_id, 0, saved=True)

    class _EmptyStage:
        selected_payloads = ()

    class _Snapshot:
        scattering = _EmptyStage()
        residual_field = _EmptyStage()

    monkeypatch.setattr(rsc, "scan_run_state", lambda *a, **k: _Snapshot())
    monkeypatch.setattr(rsc, "write_performance_metrics", lambda **k: None)

    rsc.rebuild_sqlite_cache_from_manifests(
        db,
        output_dir=tmp_path,
        run_digest="r" * 12,
        residual_parameter_digest=DIGEST,
    )
    # credited pairs survive the reset; the uncredited one is wiped
    assert db._status[(1, 0)] is True
    assert db._status[(2, 0)] is True
    assert db._status[(3, 0)] is False
