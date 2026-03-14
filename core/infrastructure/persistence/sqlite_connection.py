from __future__ import annotations

import sqlite3


PRECISION = 6
CHUNK_SIZE = 150
SQL_IN_CHUNK = 150


def create_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(
        db_path,
        timeout=60,
        check_same_thread=False,
    )
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA journal_mode = DELETE;")
    connection.execute("PRAGMA synchronous = NORMAL;")
    connection.execute("PRAGMA busy_timeout = 60000;")
    return connection
