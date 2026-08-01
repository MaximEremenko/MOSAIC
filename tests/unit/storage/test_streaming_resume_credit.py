"""Streaming resume credit: COMMITTED reducer progress manifests must
survive the SQLite rebuild — streaming writes no payload manifests, so
without this source a completed case re-derives its whole residual on
retry. Manifests stay the authority: wrong digest, non-committed status,
pending work, or missing final artifacts all leave rows unsaved.

The scan goes through the typed manifest loader, and artifact presence is
re-derived from output_dir (not the write-time-absolute paths embedded in
the manifest) so a moved output directory keeps its credits."""
from pathlib import Path

from core.contracts import CompletionStatus
from core.residual_field.artifacts import build_residual_field_output_artifact_refs
from core.residual_field.manifest_io import (
    _build_residual_field_reducer_progress_manifest,
    write_residual_field_reducer_progress_manifest,
)
from core.storage.run_state_cache import _committed_streaming_residual_credits

DIGEST = "abc123def456"


def _write_manifest(
    tmp_path: Path,
    chunk_id: int,
    *,
    digest: str = DIGEST,
    status: CompletionStatus = CompletionStatus.COMMITTED,
    interval_ids=(1, 2, 3),
    pending_interval_ids=(),
    artifacts_exist: bool = True,
) -> None:
    final_artifacts = build_residual_field_output_artifact_refs(
        str(tmp_path), chunk_id
    )
    if artifacts_exist:
        for ref in final_artifacts:
            Path(ref.path).write_bytes(b"x")
    manifest = _build_residual_field_reducer_progress_manifest(
        output_dir=str(tmp_path),
        chunk_id=chunk_id,
        parameter_digest=digest,
        completion_status=status,
        incorporated_shard_keys=(f"local-accumulator-snapshot:{chunk_id}:0:seq-1",),
        incorporated_interval_ids=tuple(interval_ids),
        reclaimable_shard_keys=(),
        final_artifacts=final_artifacts,
        pending_interval_ids=tuple(pending_interval_ids),
    )
    write_residual_field_reducer_progress_manifest(manifest)


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
    _write_manifest(tmp_path, 0, status=CompletionStatus.MATERIALIZED)
    _write_manifest(tmp_path, 1, pending_interval_ids=(7,))
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == []


def test_missing_final_artifact_never_credits(tmp_path):
    _write_manifest(tmp_path, 0, artifacts_exist=False)
    assert _committed_streaming_residual_credits(tmp_path, DIGEST) == []


def test_credits_survive_output_dir_relocation(tmp_path):
    """The write-time output dir differs from where the scan runs — a
    workstation-to-SLURM remount. Presence must be judged against the
    CURRENT output_dir, not the absolute paths written into the manifest
    (which point at a directory that no longer exists)."""
    import shutil

    original = tmp_path / "original"
    original.mkdir()
    _write_manifest(original, 0, interval_ids=(4, 5))
    relocated = tmp_path / "relocated"
    shutil.move(str(original), str(relocated))
    assert _committed_streaming_residual_credits(relocated, DIGEST) == [
        (0, (4, 5))
    ]


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
