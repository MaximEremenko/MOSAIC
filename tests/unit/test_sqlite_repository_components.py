from core.storage.sqlite_connection import create_connection
from core.storage.sqlite_repositories import create_database_parts


def test_create_connection_sets_sqlite_pragmas(tmp_path):
    connection = create_connection(str(tmp_path / "db.sqlite"))
    try:
        row = connection.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1
    finally:
        connection.close()


def test_create_database_parts_returns_split_components(tmp_path):
    parts = create_database_parts(tmp_path / "db.sqlite", 1, "test-db")
    assert len(parts) == 5
