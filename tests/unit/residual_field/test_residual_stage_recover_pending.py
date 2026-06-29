from __future__ import annotations

from types import SimpleNamespace

from core.residual_field.stage import ResidualFieldStage


def _workflow_parameters():
    return SimpleNamespace(runtime_info={})


def test_recover_pending_noop_without_db_manager():
    stage = ResidualFieldStage()
    artifacts = SimpleNamespace(output_dir="ignored")
    assert (
        stage.recover_pending(
            workflow_parameters=_workflow_parameters(),
            artifacts=artifacts,
            client=None,
        )
        == []
    )


def test_recover_pending_noop_when_backend_not_local(monkeypatch):
    stage = ResidualFieldStage()
    artifacts = SimpleNamespace(
        output_dir="ignored",
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [1, 2],
            db_path="ignored.db",
        ),
    )
    backend = SimpleNamespace(
        layout=SimpleNamespace(kind="durable_shared_restartable"),
        load_progress_manifest=lambda **kwargs: object(),
        finalize_chunk=lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "core.residual_field.stage.resolve_residual_field_reducer_backend",
        lambda **kwargs: backend,
    )
    assert (
        stage.recover_pending(
            workflow_parameters=_workflow_parameters(),
            artifacts=artifacts,
            client=None,
        )
        == []
    )


def test_recover_pending_finalizes_committed_local_chunks(monkeypatch, tmp_path):
    stage = ResidualFieldStage()
    artifacts = SimpleNamespace(
        output_dir=str(tmp_path),
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [5, 3, 7],
            db_path=str(tmp_path / "state.db"),
        ),
    )

    finalize_calls: list[dict] = []

    def fake_finalize_chunk(**kwargs):
        finalize_calls.append(kwargs)
        # Chunk 7 has no committed final artifacts yet -> not recovered.
        return None if kwargs["chunk_id"] == 7 else object()

    backend = SimpleNamespace(
        layout=SimpleNamespace(kind="local_restartable"),
        # Chunk 3 has no progress manifest -> skipped before finalize.
        load_progress_manifest=lambda *, output_dir, chunk_id, parameter_digest: (
            None if chunk_id == 3 else object()
        ),
        finalize_chunk=fake_finalize_chunk,
    )
    monkeypatch.setattr(
        "core.residual_field.stage.resolve_residual_field_reducer_backend",
        lambda **kwargs: backend,
    )

    recovered = stage.recover_pending(
        workflow_parameters=_workflow_parameters(),
        artifacts=artifacts,
        client=None,
    )

    # Chunk 3 skipped (no progress), chunk 7 not finalized -> only 5 recovered.
    assert recovered == [5]
    # finalize is attempted for chunks with progress, in sorted order.
    assert [call["chunk_id"] for call in finalize_calls] == [5, 7]
    assert all(call["cleanup_policy"] == "off" for call in finalize_calls)
