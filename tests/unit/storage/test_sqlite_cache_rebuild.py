from __future__ import annotations

import numpy as np

from core.scattering.commit import (
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    write_scattering_attempt,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)
from core.storage.database_manager import DatabaseManager
from core.workflow.run_state_cache import (
    rebuild_sqlite_cache_from_manifests,
    scan_run_state,
)


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _seed_db(db):
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [
            {"h_range": (0.0, 0.5)},
            {"h_range": (0.5, 1.0)},
        ]
    )
    db.insert_interval_chunk_status_batch([(interval_id, 7, 0) for interval_id in interval_ids])
    return interval_ids


def _write_scattering_commit(output_dir, interval_ids):
    for interval_id in interval_ids:
        write_scattering_attempt(
            output_dir=output_dir,
            interval_id=interval_id,
            chunk_id=7,
            attempt_id=f"try-{interval_id}",
            grid_shape_nd=np.array([[2]], dtype=np.int64),
            amplitudes_delta=np.array([complex(interval_id), complex(interval_id + 1)]),
            amplitudes_average=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
            contribution_reciprocal_points=1,
            point_ids=np.array([10, 11], dtype=np.int64),
            **IDENTITY,
        )
    candidate = create_scattering_commit_candidate(
        output_dir=output_dir,
        run_digest="run123",
        chunk_id=7,
        expected_interval_ids=tuple(interval_ids),
    )
    write_scattering_stage_plan(
        output_dir=output_dir,
        run_digest="run123",
        expected_by_chunk={7: tuple(interval_ids)},
    )
    promote_scattering_chunk_commit(output_dir=output_dir, candidate=candidate)
    write_scattering_stage_commit(
        output_dir=output_dir,
        run_digest="run123",
        chunk_ids=(7,),
    )


def test_sqlite_cache_rebuild_from_valid_manifests_after_db_loss(tmp_path):
    db_path = tmp_path / "state.db"
    db = DatabaseManager(str(db_path), dimension=1)
    interval_ids = _seed_db(db)
    _write_scattering_commit(tmp_path, interval_ids)
    db.close()
    db_path.unlink()

    rebuilt = DatabaseManager(str(db_path), dimension=1)
    _seed_db(rebuilt)
    snapshot = rebuild_sqlite_cache_from_manifests(
        rebuilt,
        output_dir=tmp_path,
        run_digest="run123",
    )

    assert snapshot.scattering.complete
    assert rebuilt.get_unsaved_interval_chunks() == []
    assert scan_run_state(tmp_path, "run123").scattering.valid_chunk_ids == (7,)
    rebuilt.close()
