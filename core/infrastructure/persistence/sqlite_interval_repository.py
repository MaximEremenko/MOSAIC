from __future__ import annotations

import logging
import sqlite3
from itertools import chain
from typing import Any

from core.domain.models import ReciprocalInterval
from core.infrastructure.persistence.sqlite_connection import CHUNK_SIZE, PRECISION


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
