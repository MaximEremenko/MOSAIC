from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.storage.sqlite_repositories import create_database_parts


def create_db_manager_for_thread(
    db_path: str | Path, dimension: int = 3
) -> "DatabaseManager":
    return DatabaseManager(str(db_path), dimension)


class DatabaseManager:
    def __init__(self, db_path: str, dimension: int = 3):
        self.db_path = db_path
        self.dimension = dimension
        self.logger = logging.getLogger(self.__class__.__name__)
        (
            self.connection,
            self.schema_manager,
            self.point_repository,
            self.interval_repository,
            self.processing_state_repository,
        ) = create_database_parts(
            db_path=db_path,
            dimension=dimension,
            logger_name=self.__class__.__name__,
        )
        self.cursor = self.connection.cursor()

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

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]:
        return self.processing_state_repository.get_unsaved_interval_chunks()

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None:
        self.interval_repository.mark_interval_precomputed(interval_id, done)

    def is_interval_precomputed(self, interval_id: int) -> bool:
        return self.interval_repository.is_interval_precomputed(interval_id)

    def close(self) -> None:
        self.connection.close()
        self.logger.debug("DB connection closed.")
