from __future__ import annotations

import numpy as np
import pytest

from core.scattering.artifacts import (
    ScatteringArtifactStore,
    persist_scattering_interval_chunk_shard,
    reduce_scattering_shards_for_chunk,
)
from core.scattering.contracts import ScatteringWorkUnit
from core.storage.database_manager import DatabaseManager


def _db_with_chunk(tmp_path, interval_count: int = 2):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    interval_ids = db.insert_reciprocal_space_interval_batch(
        [{"h_range": (float(i), float(i + 1))} for i in range(interval_count)]
    )
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 10 + row,
                "coordinates": [0.0],
                "dist_from_atom_center": [0.0],
                "step_in_frac": [0.0],
                "chunk_id": 3,
                "grid_amplitude_initialized": 1,
            }
            for row in range(2)
        ]
    )
    db.insert_interval_chunk_status_batch([(interval_id, 3, 0) for interval_id in interval_ids])
    return db, interval_ids


def test_scattering_stage2_reducer_publishes_after_complete_coverage(tmp_path):
    db, interval_ids = _db_with_chunk(tmp_path)
    try:
        for scale, interval_id in enumerate(interval_ids, start=1):
            work_unit = ScatteringWorkUnit.interval_chunk(
                interval_id=interval_id,
                chunk_id=3,
                dimension=1,
                output_dir=str(tmp_path),
            )
            persist_scattering_interval_chunk_shard(
                work_unit,
                grid_shape_nd=np.array([[2]], dtype=np.int64),
                total_reciprocal_points=9,
                contribution_reciprocal_points=scale,
                amplitudes_delta=np.array([scale + 0j, scale + 1j]),
                amplitudes_average=np.array([10 * scale + 0j, 20 * scale + 0j]),
                output_dir=str(tmp_path),
                quiet_logs=True,
            )

        reduce_scattering_shards_for_chunk(
            chunk_id=3,
            expected_interval_ids=tuple(interval_ids),
            total_reciprocal_points=9,
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        store = ScatteringArtifactStore(str(tmp_path))
        amplitudes, amplitudes_av, nrec, _ = store.load_chunk_payloads(3)
        assert db.get_unsaved_interval_chunks() == []
        assert nrec == 3
        np.testing.assert_allclose(amplitudes[:, 1], np.array([3 + 0j, 3 + 2j]))
        np.testing.assert_allclose(amplitudes_av[:, 1], np.array([30 + 0j, 60 + 0j]))
    finally:
        db.close()


def test_scattering_stage2_reducer_refuses_partial_coverage(tmp_path):
    db, interval_ids = _db_with_chunk(tmp_path)
    try:
        work_unit = ScatteringWorkUnit.interval_chunk(
            interval_id=interval_ids[0],
            chunk_id=3,
            dimension=1,
            output_dir=str(tmp_path),
        )
        persist_scattering_interval_chunk_shard(
            work_unit,
            grid_shape_nd=np.array([[2]], dtype=np.int64),
            total_reciprocal_points=9,
            contribution_reciprocal_points=1,
            amplitudes_delta=np.array([1 + 0j, 1 + 1j]),
            amplitudes_average=np.array([10 + 0j, 20 + 0j]),
            output_dir=str(tmp_path),
            quiet_logs=True,
        )

        with pytest.raises(RuntimeError, match="missing interval-chunk shard coverage"):
            reduce_scattering_shards_for_chunk(
                chunk_id=3,
                expected_interval_ids=tuple(interval_ids),
                total_reciprocal_points=9,
                output_dir=str(tmp_path),
                db_path=db.db_path,
                quiet_logs=True,
            )

        assert set(db.get_unsaved_interval_chunks()) == {
            (interval_ids[0], 3),
            (interval_ids[1], 3),
        }
    finally:
        db.close()
