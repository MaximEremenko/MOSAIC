from __future__ import annotations

import json
import logging
import sqlite3
from itertools import chain
from pathlib import Path
from typing import Any

from core.domain.models import ReciprocalInterval

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


class SQLitePointRepository:
    def __init__(self, connection: sqlite3.Connection, logger: logging.Logger) -> None:
        self.connection = connection
        self.logger = logger

    def get_point_data_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        try:
            cursor = self.connection.execute(
                """
                SELECT central_point_id,
                       coordinates,
                       dist_from_atom_center,
                       step_in_frac,
                       chunk_id,
                       grid_amplitude_initialized
                FROM PointData
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )
            return [
                {
                    "central_point_id": row[0],
                    "coordinates": json.loads(row[1]),
                    "dist_from_atom_center": json.loads(row[2]),
                    "step_in_frac": json.loads(row[3]),
                    "chunk_id": row[4],
                    "grid_amplitude_initialized": row[5],
                }
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as exc:
            self.logger.error("Error retrieving chunk %s: %s", chunk_id, exc)
            return []

    def get_pending_chunk_ids(self) -> list[int]:
        try:
            cursor = self.connection.execute("SELECT DISTINCT chunk_id FROM PointData")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            self.logger.error("get_pending_chunk_ids failed: %s", exc)
            return []

    def get_point_data_for_point_ids(self, point_ids: list[int]) -> list[dict[str, Any]]:
        if not point_ids:
            return []
        placeholders = ",".join("?" * len(point_ids))
        try:
            cursor = self.connection.execute(
                f"""
                SELECT id,
                       central_point_id,
                       coordinates,
                       dist_from_atom_center,
                       step_in_frac,
                       chunk_id,
                       grid_amplitude_initialized
                FROM PointData
                WHERE id IN ({placeholders})
                """,
                point_ids,
            )
            return [
                {
                    "id": row[0],
                    "central_point_id": row[1],
                    "coordinates": json.loads(row[2]),
                    "dist_from_atom_center": json.loads(row[3]),
                    "step_in_frac": json.loads(row[4]),
                    "chunk_id": row[5],
                    "grid_amplitude_initialized": row[6],
                }
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as exc:
            self.logger.error("get_point_data_for_point_ids failed: %s", exc)
            return []

    def _fetch_point_ids(self, central_ids: list[int]) -> dict[int, int]:
        id_map: dict[int, int] = {}
        for offset in range(0, len(central_ids), SQL_IN_CHUNK):
            chunk = central_ids[offset : offset + SQL_IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cursor = self.connection.execute(
                f"""
                SELECT central_point_id, id
                FROM PointData
                WHERE central_point_id IN ({placeholders})
                """,
                chunk,
            )
            id_map.update(dict(cursor.fetchall()))
        return id_map

    def insert_point_data_batch(self, point_data_list: list[dict[str, Any]]) -> list[int]:
        if not point_data_list:
            return []

        try:
            with self.connection:
                self.connection.executemany(
                    """
                    INSERT OR IGNORE INTO PointData
                      (central_point_id,
                       coordinates,
                       dist_from_atom_center,
                       step_in_frac,
                       chunk_id,
                       grid_amplitude_initialized)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            point_data["central_point_id"],
                            json.dumps(point_data["coordinates"]),
                            json.dumps(point_data["dist_from_atom_center"]),
                            json.dumps(point_data["step_in_frac"]),
                            point_data["chunk_id"],
                            point_data["grid_amplitude_initialized"],
                        )
                        for point_data in point_data_list
                    ],
                )
                central_ids = [point_data["central_point_id"] for point_data in point_data_list]
                id_map = self._fetch_point_ids(central_ids)
            return [id_map.get(central_id, -1) for central_id in central_ids]
        except sqlite3.Error as exc:
            self.logger.error("insert_point_data_batch failed: %s", exc)
            return []


class SQLiteIntervalRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        logger: logging.Logger,
        dimension: int,
    ) -> None:
        self.connection = connection
        self.logger = logger
        self.dimension = dimension

    def get_pending_parts(self) -> list[dict[str, Any]]:
        try:
            cursor = self.connection.execute(
                """
                SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                FROM ReciprocalSpaceInterval
                """
            )
            result: list[dict[str, Any]] = []
            for row in cursor.fetchall():
                interval = ReciprocalInterval(
                    interval_id=row[0],
                    h_range=(row[1], row[2]),
                    k_range=(row[3], row[4]) if self.dimension > 1 else (0.0, 0.0),
                    l_range=(row[5], row[6]) if self.dimension > 2 else (0.0, 0.0),
                )
                result.append(interval.to_mapping(self.dimension))
            return result
        except sqlite3.Error as exc:
            self.logger.error("get_pending_parts failed: %s", exc)
            return []

    def get_intervals_by_ids(self, interval_ids: list[int]) -> list[ReciprocalInterval]:
        if not interval_ids:
            return []
        placeholders = ",".join("?" * len(interval_ids))
        cursor = self.connection.execute(
            f"""
            SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
            FROM ReciprocalSpaceInterval
            WHERE id IN ({placeholders})
            """,
            interval_ids,
        )
        return [
            ReciprocalInterval(
                interval_id=row[0],
                h_range=(row[1], row[2]),
                k_range=(row[3], row[4]) if self.dimension > 1 else (0.0, 0.0),
                l_range=(row[5], row[6]) if self.dimension > 2 else (0.0, 0.0),
            )
            for row in cursor.fetchall()
        ]

    def _pad_ranges(self, interval: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        h_start, h_end = (round(value, PRECISION) for value in interval["h_range"])
        if self.dimension > 1:
            k_start, k_end = (round(value, PRECISION) for value in interval["k_range"])
        else:
            k_start = k_end = 0.0
        if self.dimension > 2:
            l_start, l_end = (round(value, PRECISION) for value in interval["l_range"])
        else:
            l_start = l_end = 0.0
        return (h_start, h_end, k_start, k_end, l_start, l_end)

    def insert_reciprocal_space_interval_batch(
        self, interval_list: list[dict[str, Any]]
    ) -> list[int]:
        if not interval_list:
            return []

        tuples = [self._pad_ranges(interval) for interval in interval_list]
        existing: dict[tuple[float, ...], int] = {}

        with self.connection:
            for offset in range(0, len(tuples), CHUNK_SIZE):
                chunk = tuples[offset : offset + CHUNK_SIZE]
                placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
                cursor = self.connection.execute(
                    f"""
                    SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                    FROM ReciprocalSpaceInterval
                    WHERE (h_start, h_end, k_start, k_end, l_start, l_end)
                          IN ({placeholders})
                    """,
                    list(chain.from_iterable(chunk)),
                )
                existing.update({tuple(row[1:]): row[0] for row in cursor.fetchall()})

            missing = [interval_tuple for interval_tuple in tuples if interval_tuple not in existing]
            for offset in range(0, len(missing), CHUNK_SIZE):
                chunk = missing[offset : offset + CHUNK_SIZE]
                self.connection.executemany(
                    """
                    INSERT INTO ReciprocalSpaceInterval
                      (h_start, h_end, k_start, k_end, l_start, l_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    chunk,
                )

            if missing:
                for offset in range(0, len(missing), CHUNK_SIZE):
                    chunk = missing[offset : offset + CHUNK_SIZE]
                    placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
                    cursor = self.connection.execute(
                        f"""
                        SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                        FROM ReciprocalSpaceInterval
                        WHERE (h_start, h_end, k_start, k_end, l_start, l_end)
                              IN ({placeholders})
                        """,
                        list(chain.from_iterable(chunk)),
                    )
                    existing.update({tuple(row[1:]): row[0] for row in cursor.fetchall()})

        return [existing[interval_tuple] for interval_tuple in tuples]

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE ReciprocalSpaceInterval SET precomputed = ? WHERE id = ?",
                (1 if done else 0, interval_id),
            )

    def is_interval_precomputed(self, interval_id: int) -> bool:
        cursor = self.connection.execute(
            "SELECT precomputed FROM ReciprocalSpaceInterval WHERE id = ?",
            (interval_id,),
        )
        row = cursor.fetchone()
        return bool(row and row[0])


class SQLiteProcessingStateRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        logger: logging.Logger,
    ) -> None:
        self.connection = connection
        self.logger = logger

    def insert_interval_chunk_status_batch(
        self, status_list: list[tuple[int, int, int | bool]]
    ) -> None:
        if not status_list:
            return
        try:
            with self.connection:
                self.connection.executemany(
                    """
                    INSERT OR IGNORE INTO Interval_Chunk_Status
                    (reciprocal_space_id, chunk_id, saved)
                    VALUES (?, ?, ?)
                    """,
                    status_list,
                )
        except sqlite3.Error as exc:
            self.logger.error("insert_interval_chunk_status_batch failed: %s", exc)

    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None:
        del associations
        return

    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None:
        try:
            if point_id is not None and chunk_id is None:
                cursor = self.connection.execute(
                    "SELECT chunk_id FROM PointData WHERE id = ?",
                    (point_id,),
                )
                row = cursor.fetchone()
                if row:
                    chunk_id = row[0]

            if chunk_id is None:
                return

            self.update_interval_chunk_status(reciprocal_space_id, chunk_id, saved)
        except sqlite3.Error as exc:
            self.logger.error("update_saved_status_for_chunk_or_point failed: %s", exc)

    def get_unsaved_associations(self) -> list[tuple[int, int]]:
        try:
            cursor = self.connection.execute(
                """
                SELECT reciprocal_space_id, chunk_id
                FROM Interval_Chunk_Status
                WHERE saved = 0
                """
            )
            todo_pairs = cursor.fetchall()
            if not todo_pairs:
                return []

            result: list[tuple[int, int]] = []
            for interval_id, chunk_id in todo_pairs:
                point_cursor = self.connection.execute(
                    "SELECT central_point_id FROM PointData WHERE chunk_id = ?",
                    (chunk_id,),
                )
                result.extend(
                    (point_id, interval_id) for (point_id,) in point_cursor.fetchall()
                )
            return result
        except sqlite3.Error as exc:
            self.logger.error("get_unsaved_associations failed: %s", exc)
            return []

    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO Interval_Chunk_Status
                    (reciprocal_space_id, chunk_id, saved)
                    VALUES (?, ?, ?)
                    ON CONFLICT(reciprocal_space_id, chunk_id)
                    DO UPDATE SET saved = excluded.saved
                    """,
                    (interval_id, chunk_id, int(saved)),
                )
        except sqlite3.Error as exc:
            self.logger.error("update_interval_chunk_status failed: %s", exc)

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]:
        try:
            cursor = self.connection.execute(
                """
                SELECT reciprocal_space_id, chunk_id
                FROM Interval_Chunk_Status
                WHERE saved = 0
                """
            )
            return cursor.fetchall()
        except sqlite3.Error as exc:
            self.logger.error("get_unsaved_interval_chunks failed: %s", exc)
            return []


def create_database_parts(
    db_path: str | Path,
    dimension: int,
    logger_name: str,
) -> tuple[
    sqlite3.Connection,
    SQLiteSchemaManager,
    SQLitePointRepository,
    SQLiteIntervalRepository,
    SQLiteProcessingStateRepository,
]:
    connection = create_connection(str(db_path))
    logger = logging.getLogger(logger_name)
    schema_manager = SQLiteSchemaManager(connection, logger)
    schema_manager.ensure_schema()
    point_repository = SQLitePointRepository(connection, logger)
    interval_repository = SQLiteIntervalRepository(connection, logger, dimension)
    state_repository = SQLiteProcessingStateRepository(connection, logger)
    return (
        connection,
        schema_manager,
        point_repository,
        interval_repository,
        state_repository,
    )
