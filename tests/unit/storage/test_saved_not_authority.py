from __future__ import annotations

from core.storage.database_manager import DatabaseManager
from core.workflow.run_state_cache import pending_scattering_interval_chunks, scan_run_state


def test_saved_rows_do_not_make_missing_manifests_complete(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [
            {"h_range": (0.0, 0.5)},
            {"h_range": (0.5, 1.0)},
        ]
    )
    db.insert_interval_chunk_status_batch([(interval_id, 3, 1) for interval_id in interval_ids])

    snapshot = scan_run_state(tmp_path, "run123")
    pending = pending_scattering_interval_chunks(
        snapshot,
        db.get_interval_chunks(),
    )

    assert db.get_unsaved_interval_chunks() == []
    assert pending == [(interval_ids[0], 3), (interval_ids[1], 3)]
    assert not snapshot.scattering.complete
    db.close()
