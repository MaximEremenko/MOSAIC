from __future__ import annotations

from core.storage.database_manager import DatabaseManager


def test_corrupt_sqlite_cache_is_discarded_not_repaired_in_place(tmp_path):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"not a sqlite database")

    db = DatabaseManager(str(db_path), dimension=1)

    assert db.discarded_corrupt_db_path is not None
    assert db_path.exists()
    assert db.get_interval_chunks() == []
    db.close()
