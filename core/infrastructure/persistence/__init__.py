"""Persistence adapters and repositories."""

from .database_manager import DatabaseManager, create_db_manager_for_thread

__all__ = ["DatabaseManager", "create_db_manager_for_thread"]
