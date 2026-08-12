from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.models import ReciprocalInterval
from core.storage.sqlite_repositories import create_database_parts


@runtime_checkable
class DatabaseManagerProtocol(Protocol):
    """Shared contract of the SQLite-backed DatabaseManager and the in-memory
    ManifestOnlyDatabaseManager.

    Consumers used to duck-type this surface with per-call-site hasattr/getattr
    probes and row-at-a-time fallbacks; both implementations provide every
    method below, so call sites may invoke them directly.
    """

    db_path: str
    dimension: int
    cache_enabled: bool

    def get_point_data_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]: ...

    def get_pending_chunk_ids(self) -> list[int]: ...

    def get_pending_parts(self) -> list[dict[str, Any]]: ...

    def get_point_data_for_point_ids(self, point_ids: list[int]) -> list[dict[str, Any]]: ...

    def get_intervals_by_ids(self, interval_ids: list[int]) -> list[ReciprocalInterval]: ...

    def insert_point_data_batch(self, point_data_list: list[dict[str, Any]]) -> list[int]: ...

    def insert_reciprocal_space_interval_batch(
        self, interval_list: list[dict[str, Any]]
    ) -> list[int]: ...

    def insert_interval_chunk_status_batch(
        self, status_list: list[tuple[int, int, int | bool]]
    ) -> None: ...

    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None: ...

    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None: ...

    def get_unsaved_associations(self) -> list[tuple[int, int]]: ...

    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ) -> None: ...

    def update_interval_chunk_status_batch(
        self, status_rows: "list[tuple[int, int, int | bool]]"
    ) -> None: ...

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]: ...

    def get_interval_chunks(self) -> list[tuple[int, int]]: ...

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None: ...

    def is_interval_precomputed(self, interval_id: int) -> bool: ...

    def close(self) -> None: ...


def create_db_manager_for_thread(
    db_path: str | Path, dimension: int = 3
) -> "DatabaseManager":
    return globals()["DatabaseManager"](str(db_path), dimension)


class DatabaseManager:
    def __init__(self, db_path: str, dimension: int = 3):
        self.db_path = db_path
        self.dimension = dimension
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_enabled = True
        self.discarded_corrupt_db_path: str | None = None
        try:
            self._open()
        except sqlite3.DatabaseError:
            if str(db_path) == ":memory:":
                raise
            self.discarded_corrupt_db_path = self._discard_bad_cache(db_path)
            self._open()

    def _open(self) -> None:
        (
            self.connection,
            self.schema_manager,
            self.point_repository,
            self.interval_repository,
            self.processing_state_repository,
        ) = create_database_parts(
            db_path=self.db_path,
            dimension=self.dimension,
            logger_name=self.__class__.__name__,
        )
        self.cursor = self.connection.cursor()

    def _discard_bad_cache(self, db_path: str) -> str:
        path = Path(db_path)
        if not path.exists():
            return ""
        target = path.with_name(
            f"{path.name}.discarded-{time.strftime('%Y%m%d%H%M%S')}"
        )
        try:
            os.replace(path, target)
        except OSError:
            self.logger.exception("Failed to discard corrupt SQLite cache %s", path)
            raise
        self.logger.warning("Discarded corrupt SQLite cache %s -> %s", path, target)
        return str(target)

    def get_point_data_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        return self.point_repository.get_point_data_for_chunk(chunk_id)

    def get_pending_chunk_ids(self) -> list[int]:
        return self.point_repository.get_pending_chunk_ids()

    def get_pending_parts(self) -> list[dict[str, Any]]:
        return self.interval_repository.get_pending_parts()

    def get_point_data_for_point_ids(self, point_ids: list[int]) -> list[dict[str, Any]]:
        return self.point_repository.get_point_data_for_point_ids(point_ids)

    def get_intervals_by_ids(self, interval_ids: list[int]):
        return self.interval_repository.get_intervals_by_ids(interval_ids)

    def insert_point_data_batch(self, point_data_list: list[dict[str, Any]]) -> list[int]:
        return self.point_repository.insert_point_data_batch(point_data_list)

    def insert_reciprocal_space_interval_batch(
        self, interval_list: list[dict[str, Any]]
    ) -> list[int]:
        return self.interval_repository.insert_reciprocal_space_interval_batch(interval_list)

    def insert_interval_chunk_status_batch(
        self, status_list: list[tuple[int, int, int | bool]]
    ) -> None:
        self.processing_state_repository.insert_interval_chunk_status_batch(status_list)

    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None:
        self.processing_state_repository.associate_point_reciprocal_space_batch(
            associations
        )

    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None:
        self.processing_state_repository.update_saved_status_for_chunk_or_point(
            reciprocal_space_id=reciprocal_space_id,
            point_id=point_id,
            chunk_id=chunk_id,
            saved=saved,
        )

    def get_unsaved_associations(self) -> list[tuple[int, int]]:
        return self.processing_state_repository.get_unsaved_associations()

    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ) -> None:
        self.processing_state_repository.update_interval_chunk_status(
            interval_id=interval_id,
            chunk_id=chunk_id,
            saved=saved,
        )

    def update_interval_chunk_status_batch(
        self, status_rows: "list[tuple[int, int, int | bool]]"
    ) -> None:
        self.processing_state_repository.update_interval_chunk_status_batch(
            status_rows
        )

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]:
        return self.processing_state_repository.get_unsaved_interval_chunks()

    def get_interval_chunks(self) -> list[tuple[int, int]]:
        return self.processing_state_repository.get_interval_chunks()

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None:
        self.interval_repository.mark_interval_precomputed(interval_id, done)

    def is_interval_precomputed(self, interval_id: int) -> bool:
        return self.interval_repository.is_interval_precomputed(interval_id)

    def close(self) -> None:
        self.connection.close()
        self.logger.debug("DB connection closed.")


class ManifestOnlyDatabaseManager:
    """In-memory driver cache used when durable manifests are the only authority."""

    def __init__(self, db_path: str = ":manifest-only:", dimension: int = 3):
        self.db_path = db_path
        self.dimension = int(dimension)
        self.cache_enabled = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self._points: list[dict[str, Any]] = []
        self._point_ids_by_central: dict[int, int] = {}
        self._intervals: dict[int, dict[str, Any]] = {}
        self._interval_keys: dict[tuple[float, float, float, float, float, float], int] = {}
        self._precomputed: dict[int, bool] = {}
        self._status: dict[tuple[int, int], bool] = {}

    def get_point_data_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self._points if int(row["chunk_id"]) == int(chunk_id)]

    def get_pending_chunk_ids(self) -> list[int]:
        return sorted({int(row["chunk_id"]) for row in self._points})

    def get_pending_parts(self) -> list[dict[str, Any]]:
        return [dict(value) for _key, value in sorted(self._intervals.items())]

    def get_point_data_for_point_ids(self, point_ids: list[int]) -> list[dict[str, Any]]:
        wanted = {int(point_id) for point_id in point_ids}
        return [dict(row) for row in self._points if int(row.get("id", -1)) in wanted]

    def get_intervals_by_ids(self, interval_ids: list[int]):
        result = []
        for interval_id in interval_ids:
            item = self._intervals.get(int(interval_id))
            if item is None:
                continue
            result.append(
                ReciprocalInterval(
                    interval_id=int(interval_id),
                    h_range=tuple(item["h_range"]),
                    k_range=tuple(item.get("k_range", (0.0, 0.0))),
                    l_range=tuple(item.get("l_range", (0.0, 0.0))),
                )
            )
        return result

    def insert_point_data_batch(self, point_data_list: list[dict[str, Any]]) -> list[int]:
        ids: list[int] = []
        for point_data in point_data_list:
            central_id = int(point_data["central_point_id"])
            if central_id in self._point_ids_by_central:
                ids.append(self._point_ids_by_central[central_id])
                continue
            point_id = len(self._points) + 1
            row = dict(point_data)
            row["id"] = point_id
            self._points.append(row)
            self._point_ids_by_central[central_id] = point_id
            ids.append(point_id)
        return ids

    def _interval_key(self, interval: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        h_start, h_end = (round(float(value), 6) for value in interval["h_range"])
        if self.dimension > 1:
            k_start, k_end = (round(float(value), 6) for value in interval.get("k_range", (0, 0)))
        else:
            k_start = k_end = 0.0
        if self.dimension > 2:
            l_start, l_end = (round(float(value), 6) for value in interval.get("l_range", (0, 0)))
        else:
            l_start = l_end = 0.0
        return (h_start, h_end, k_start, k_end, l_start, l_end)

    def insert_reciprocal_space_interval_batch(
        self, interval_list: list[dict[str, Any]]
    ) -> list[int]:
        ids: list[int] = []
        for interval in interval_list:
            key = self._interval_key(interval)
            interval_id = self._interval_keys.get(key)
            if interval_id is None:
                interval_id = len(self._intervals) + 1
                self._interval_keys[key] = interval_id
                self._intervals[interval_id] = {
                    "id": interval_id,
                    "h_range": (key[0], key[1]),
                    "k_range": (key[2], key[3]),
                    "l_range": (key[4], key[5]),
                }
                self._precomputed[interval_id] = False
            ids.append(interval_id)
        return ids

    def insert_interval_chunk_status_batch(
        self, status_list: list[tuple[int, int, int | bool]]
    ) -> None:
        # setdefault mirrors the SQLite manager's INSERT OR IGNORE:
        # first-write-wins, an existing row's saved flag is never touched.
        # (update_interval_chunk_status[_batch] mirror the upsert and DO
        # overwrite.) Pinned by the DatabaseManager parity test.
        for interval_id, chunk_id, saved in status_list:
            self._status.setdefault((int(interval_id), int(chunk_id)), bool(saved))

    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None:
        del associations

    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None:
        if chunk_id is None and point_id is not None:
            for row in self._points:
                if int(row.get("id", -1)) == int(point_id):
                    chunk_id = int(row["chunk_id"])
                    break
        if chunk_id is None:
            return
        self.update_interval_chunk_status(reciprocal_space_id, chunk_id, saved)

    def get_unsaved_associations(self) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for interval_id, chunk_id in self.get_unsaved_interval_chunks():
            for row in self.get_point_data_for_chunk(chunk_id):
                result.append((int(row["central_point_id"]), int(interval_id)))
        return result

    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ) -> None:
        self._status[(int(interval_id), int(chunk_id))] = bool(saved)

    def update_interval_chunk_status_batch(
        self, status_rows: "list[tuple[int, int, int | bool]]"
    ) -> None:
        # Overwrite (upsert) — mirrors the SQLite ON CONFLICT DO UPDATE.
        for interval_id, chunk_id, saved in status_rows:
            self._status[(int(interval_id), int(chunk_id))] = bool(saved)

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]:
        return sorted(
            (interval_id, chunk_id)
            for (interval_id, chunk_id), saved in self._status.items()
            if not saved
        )

    def get_interval_chunks(self) -> list[tuple[int, int]]:
        return sorted(self._status)

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None:
        self._precomputed[int(interval_id)] = bool(done)

    def is_interval_precomputed(self, interval_id: int) -> bool:
        return bool(self._precomputed.get(int(interval_id), False))

    def close(self) -> None:
        return None
