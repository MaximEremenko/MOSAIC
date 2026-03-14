from __future__ import annotations

import logging
import sqlite3


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
