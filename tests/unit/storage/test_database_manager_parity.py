"""DatabaseManager vs ManifestOnlyDatabaseManager parity.

Both implementations satisfy DatabaseManagerProtocol, and the interval-chunk
status semantics must be identical on the same call sequence:

- ``insert_interval_chunk_status_batch`` is first-write-wins (SQLite:
  INSERT OR IGNORE; manifest-only: dict.setdefault) — an existing row's
  saved flag is never touched;
- ``update_interval_chunk_status[_batch]`` is an overwriting upsert (SQLite:
  ON CONFLICT DO UPDATE) — it inserts missing rows and overwrites existing
  ones.
"""
from __future__ import annotations

import pytest

from core.storage.database_manager import (
    DatabaseManager,
    DatabaseManagerProtocol,
    ManifestOnlyDatabaseManager,
)


def _managers(tmp_path):
    sqlite_manager = DatabaseManager(str(tmp_path / "parity.db"), dimension=1)
    manifest_manager = ManifestOnlyDatabaseManager(dimension=1)
    return sqlite_manager, manifest_manager


def _status_pairs(db):
    all_pairs = sorted((int(i), int(c)) for i, c in db.get_interval_chunks())
    unsaved = sorted((int(i), int(c)) for i, c in db.get_unsaved_interval_chunks())
    return all_pairs, unsaved


def test_both_managers_satisfy_protocol(tmp_path):
    sqlite_manager, manifest_manager = _managers(tmp_path)
    try:
        assert isinstance(sqlite_manager, DatabaseManagerProtocol)
        assert isinstance(manifest_manager, DatabaseManagerProtocol)
    finally:
        sqlite_manager.close()
        manifest_manager.close()


def test_interval_chunk_status_semantics_identical(tmp_path):
    sqlite_manager, manifest_manager = _managers(tmp_path)
    observed = []
    try:
        for db in (sqlite_manager, manifest_manager):
            interval_a, interval_b = db.insert_reciprocal_space_interval_batch(
                [{"h_range": (0.0, 0.5)}, {"h_range": (0.5, 1.0)}]
            )
            steps = []

            # Insert is first-write-wins: the second batch must not flip
            # either existing row.
            db.insert_interval_chunk_status_batch(
                [(interval_a, 1, 0), (interval_b, 1, 1)]
            )
            db.insert_interval_chunk_status_batch(
                [(interval_a, 1, 1), (interval_b, 1, 0)]
            )
            steps.append(_status_pairs(db))

            # Single-row update overwrites.
            db.update_interval_chunk_status(interval_a, 1, saved=1)
            steps.append(_status_pairs(db))

            # Batch update upserts: inserts the missing (a, 2) row AND
            # overwrites (b, 1).
            db.update_interval_chunk_status_batch(
                [(interval_a, 2, 1), (interval_b, 1, 0)]
            )
            steps.append(_status_pairs(db))

            # Batch update overwrites a row it just created.
            db.update_interval_chunk_status_batch([(interval_a, 2, 0)])
            steps.append(_status_pairs(db))

            observed.append(((int(interval_a), int(interval_b)), steps))
    finally:
        sqlite_manager.close()
        manifest_manager.close()

    (ids_sqlite, steps_sqlite), (ids_manifest, steps_manifest) = observed
    assert ids_sqlite == ids_manifest
    assert steps_sqlite == steps_manifest

    interval_a, interval_b = ids_sqlite
    # Pin the expected end-to-end sequence, not only cross-implementation
    # equality, so a symmetric regression cannot slip through.
    assert steps_sqlite == [
        (
            [(interval_a, 1), (interval_b, 1)],
            [(interval_a, 1)],  # insert: first write (saved=0/1) won
        ),
        (
            [(interval_a, 1), (interval_b, 1)],
            [],  # update overwrote (a, 1) to saved
        ),
        (
            [(interval_a, 1), (interval_a, 2), (interval_b, 1)],
            [(interval_b, 1)],  # upsert inserted (a, 2) and flipped (b, 1)
        ),
        (
            [(interval_a, 1), (interval_a, 2), (interval_b, 1)],
            [(interval_a, 2), (interval_b, 1)],  # overwrote the new row
        ),
    ]


def test_insert_first_write_wins_within_one_batch(tmp_path):
    # Duplicate keys inside a single batch: executemany applies rows in
    # order, so the FIRST occurrence wins under INSERT OR IGNORE; setdefault
    # mirrors that.
    sqlite_manager, manifest_manager = _managers(tmp_path)
    try:
        results = []
        for db in (sqlite_manager, manifest_manager):
            (interval_id,) = db.insert_reciprocal_space_interval_batch(
                [{"h_range": (0.0, 0.5)}]
            )
            db.insert_interval_chunk_status_batch(
                [(interval_id, 1, 0), (interval_id, 1, 1)]
            )
            results.append(_status_pairs(db))
        assert results[0] == results[1]
        assert results[0][1] == results[0][0]  # the saved=0 first write won
    finally:
        sqlite_manager.close()
        manifest_manager.close()
