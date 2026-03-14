from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from core.infrastructure.persistence.sqlite_connection import SQL_IN_CHUNK


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
