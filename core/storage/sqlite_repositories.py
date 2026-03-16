from __future__ import annotations

import logging
from pathlib import Path

from core.storage.sqlite_connection import create_connection
from core.storage.sqlite_interval_repository import (
    SQLiteIntervalRepository,
)
from core.storage.sqlite_point_repository import SQLitePointRepository
from core.storage.sqlite_processing_state_repository import (
    SQLiteProcessingStateRepository,
)
from core.storage.sqlite_schema import SQLiteSchemaManager


def create_database_parts(
    db_path: str | Path,
    dimension: int,
    logger_name: str,
):
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
