from __future__ import annotations

import logging
import sqlite3


class SQLiteSchemaManager:
    def __init__(self, connection: sqlite3.Connection, logger: logging.Logger) -> None:
        self.connection = connection
        self.logger = logger

    def ensure_schema(self) -> None:
        try:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS PointData (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    central_point_id INTEGER UNIQUE,
                    coordinates TEXT,
                    dist_from_atom_center TEXT,
                    step_in_frac TEXT,
                    chunk_id INTEGER,
                    grid_amplitude_initialized INTEGER
                );

                CREATE TABLE IF NOT EXISTS ReciprocalSpaceInterval (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    h_start REAL,
                    h_end REAL,
                    k_start REAL,
                    k_end REAL,
                    l_start REAL,
                    l_end REAL,
                    precomputed INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(h_start, h_end, k_start, k_end, l_start, l_end)
                );

                CREATE TABLE IF NOT EXISTS Interval_Chunk_Status (
                    reciprocal_space_id INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    saved INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (reciprocal_space_id, chunk_id),
                    FOREIGN KEY (reciprocal_space_id) REFERENCES ReciprocalSpaceInterval(id)
                ) WITHOUT ROWID;

                CREATE INDEX IF NOT EXISTS idx_pointdata_cpid
                    ON PointData (central_point_id);

                CREATE INDEX IF NOT EXISTS idx_intervalchunk_unsaved
                    ON Interval_Chunk_Status(reciprocal_space_id, chunk_id)
                    WHERE saved = 0;
                """
            )
            self.connection.commit()
            self.logger.debug("Database schema ready.")
        except sqlite3.Error as exc:
            self.logger.error("Schema init error: %s", exc)
            raise
