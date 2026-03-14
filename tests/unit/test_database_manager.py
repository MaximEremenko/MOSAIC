from core.domain.models import ReciprocalInterval
from core.infrastructure.persistence.database_manager import (
    DatabaseManager,
    create_db_manager_for_thread,
)


def test_database_manager_uses_split_repositories(tmp_path):
    db_path = tmp_path / "state.db"
    manager = DatabaseManager(str(db_path), dimension=2)
    try:
        point_ids = manager.insert_point_data_batch(
            [
                {
                    "central_point_id": 11,
                    "coordinates": [0.0, 0.1],
                    "dist_from_atom_center": [0.2, 0.3],
                    "step_in_frac": [0.01, 0.02],
                    "chunk_id": 7,
                    "grid_amplitude_initialized": 0,
                }
            ]
        )
        assert point_ids == [1]
        assert manager.get_pending_chunk_ids() == [7]

        interval_ids = manager.insert_reciprocal_space_interval_batch(
            [{"h_range": (0.0, 1.0), "k_range": (0.5, 1.5)}]
        )
        assert interval_ids == [1]
        manager.insert_interval_chunk_status_batch([(interval_ids[0], 7, 0)])

        point_rows = manager.get_point_data_for_point_ids(point_ids)
        assert point_rows[0]["id"] == 1
        assert point_rows[0]["central_point_id"] == 11

        intervals = manager.get_intervals_by_ids(interval_ids)
        assert intervals == [
            ReciprocalInterval(
                interval_id=1,
                h_range=(0.0, 1.0),
                k_range=(0.5, 1.5),
                l_range=(0.0, 0.0),
            )
        ]
        assert manager.get_unsaved_interval_chunks() == [(1, 7)]
    finally:
        manager.close()


def test_create_db_manager_for_thread_has_independent_connection(tmp_path):
    db_path = tmp_path / "threaded.db"
    manager = DatabaseManager(str(db_path), dimension=1)
    threaded = create_db_manager_for_thread(str(db_path), dimension=1)
    try:
        assert manager.connection is not threaded.connection
    finally:
        manager.close()
        threaded.close()
