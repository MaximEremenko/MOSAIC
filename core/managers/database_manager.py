# managers/database_manager.py
"""
SQLite helper for point–interval bookkeeping
-------------------------------------------
• Same public API as before – no call-site changes required.
"""

from __future__ import annotations

import sqlite3
import json
import logging
from typing import List, Dict, Tuple

from pathlib import Path
from itertools import chain
def create_db_manager_for_thread(db_path: str | Path, dimension: int = 3) -> "DatabaseManager":
    """
    Return a fresh DatabaseManager with its own SQLite connection.
    Thread-safe because each call opens a *new* sqlite3.Connection.
    """
    return DatabaseManager(str(db_path), dimension)

_PRECISION = 6           # digits to keep in ranges
_CHUNK_SIZE = 150        # 150 × 6 = 900 placeholders per statement

_SQL_IN_CHUNK = 150              # 150 × 6 columns = 900 bound variables

class DatabaseManager:
    # ------------------------------------------------------------------ #
    #  INIT & PRAGMAS                                                    #
    # ------------------------------------------------------------------ #
    def __init__(self, db_path: str, dimension: int = 3):
        self.db_path = db_path
        self.dimension = dimension
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

        self.cursor.executescript(
            """
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = NORMAL;
            """
        )

        self._initialize_database()

    # ------------------------------------------------------------------ #
    #  PUBLIC READ HELPERS (unchanged signatures)                        #
    # ------------------------------------------------------------------ #
    def get_point_data_for_chunk(self, chunk_id: int) -> List[Dict]:
        try:
            self.cursor.execute(
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
                    "central_point_id": r[0],
                    "coordinates": json.loads(r[1]),
                    "dist_from_atom_center": json.loads(r[2]),
                    "step_in_frac": json.loads(r[3]),
                    "chunk_id": r[4],
                    "grid_amplitude_initialized": r[5],
                }
                for r in self.cursor.fetchall()
            ]
        except sqlite3.Error as e:
            self.logger.error("Error retrieving chunk %s: %s", chunk_id, e)
            return []

    def get_pending_chunk_ids(self) -> List[int]:
        try:
            self.cursor.execute("SELECT DISTINCT chunk_id FROM PointData")
            ids = [r[0] for r in self.cursor.fetchall()]
            self.logger.debug("Found %d distinct chunk_ids", len(ids))
            return ids
        except sqlite3.Error as e:
            self.logger.error("get_pending_chunk_ids failed: %s", e)
            return []

    def get_pending_parts(self) -> List[Dict]:
        try:
            self.cursor.execute(
                """
                SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                FROM ReciprocalSpaceInterval
                """
            )
            res = []
            for r in self.cursor.fetchall():
                entry = {"id": r[0], "h_range": [r[1], r[2]]}
                if self.dimension > 1:
                    entry["k_range"] = [r[3], r[4]]
                if self.dimension > 2:
                    entry["l_range"] = [r[5], r[6]]
                res.append(entry)
            return res
        except sqlite3.Error as e:
            self.logger.error("get_pending_parts failed: %s", e)
            return []

    def get_point_data_for_point_ids(self, point_ids: List[int]) -> List[Dict]:
        if not point_ids:
            return []
        placeholders = ",".join("?" * len(point_ids))
        try:
            self.cursor.execute(
                f"""
                SELECT central_point_id,
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
                    "central_point_id": r[0],
                    "coordinates": json.loads(r[1]),
                    "dist_from_atom_center": json.loads(r[2]),
                    "step_in_frac": json.loads(r[3]),
                    "chunk_id": r[4],
                    "grid_amplitude_initialized": r[5],
                    "id": r[0],
                }
                for r in self.cursor.fetchall()
            ]
        except sqlite3.Error as e:
            self.logger.error("get_point_data_for_point_ids failed: %s", e)
            return []

    # ------------------------------------------------------------------ #
    #  SCHEMA INITIALISATION                                             #
    # ------------------------------------------------------------------ #
    def _initialize_database(self):
        try:
            self.cursor.executescript(
                """
                -----------------------------------------------------------
                -- POINTS
                -----------------------------------------------------------
                CREATE TABLE IF NOT EXISTS PointData (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    central_point_id INTEGER UNIQUE,
                    coordinates           TEXT,
                    dist_from_atom_center TEXT,
                    step_in_frac          TEXT,
                    chunk_id              INTEGER,
                    grid_amplitude_initialized INTEGER
                );

                -----------------------------------------------------------
                -- INTERVALS
                -----------------------------------------------------------
                CREATE TABLE IF NOT EXISTS ReciprocalSpaceInterval (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    h_start REAL, h_end REAL,
                    k_start REAL, k_end REAL,
                    l_start REAL, l_end REAL,
                    precomputed INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(h_start, h_end, k_start, k_end, l_start, l_end)
                );
                -----------------------------------------------------------
                -- NEW:  INTERVAL ↔ CHUNK STATUS
                -----------------------------------------------------------
                CREATE TABLE IF NOT EXISTS Interval_Chunk_Status (
                    reciprocal_space_id INTEGER NOT NULL,
                    chunk_id            INTEGER NOT NULL,
                    saved               INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (reciprocal_space_id, chunk_id),
                    FOREIGN KEY (reciprocal_space_id) REFERENCES ReciprocalSpaceInterval(id)
                ) WITHOUT ROWID;

                -----------------------------------------------------------
                -- INDEXES
                -----------------------------------------------------------
                CREATE INDEX IF NOT EXISTS idx_pointdata_cpid
                    ON PointData (central_point_id);

                CREATE INDEX IF NOT EXISTS idx_intervalchunk_unsaved
                    ON Interval_Chunk_Status(reciprocal_space_id, chunk_id)
                    WHERE saved = 0;
                """
            )
            self.connection.commit()
            self.logger.info("Database schema ready.")
        except sqlite3.Error as e:
            self.logger.error("Schema init error: %s", e)
   # put this near the top of the module, next to _PRECISION / _CHUNK_SIZE
    
    
    # ---------------------------------------------------------------------- #
    #  inside class DatabaseManager                                          #
    # ---------------------------------------------------------------------- #
    def _fetch_point_ids(self, central_ids: list[int]) -> dict[int, int]:
        """
        Return a mapping  {central_point_id -> row_id}  for the requested ids,
        fetching in blocks small enough to satisfy SQLite’s 999-variable limit.
        """
        id_map: dict[int, int] = {}
        for ofs in range(0, len(central_ids), _SQL_IN_CHUNK):
            chunk = central_ids[ofs : ofs + _SQL_IN_CHUNK]
            ph    = ",".join("?" * len(chunk))
            self.cursor.execute(
                f"""
                SELECT central_point_id, id
                FROM   PointData
                WHERE  central_point_id IN ({ph})
                """,
                chunk,
            )
            id_map.update(dict(self.cursor.fetchall()))
        return id_map


    def insert_point_data_batch(self, point_data_list: list[dict]) -> list[int]:
        """
        Batch-insert `point_data_list` and return the *database* ids
        that correspond to every element of the list (same order).
        Safe for arbitrarily large batches – never exceeds SQLite’s
        999-variable limit.
        """
        if not point_data_list:
            return []
    
        try:
            with self.connection:
                # ① bulk INSERT – one row per execute, so never trips the limit
                self.cursor.executemany(
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
                            pd["central_point_id"],
                            json.dumps(pd["coordinates"]),
                            json.dumps(pd["dist_from_atom_center"]),
                            json.dumps(pd["step_in_frac"]),
                            pd["chunk_id"],
                            pd["grid_amplitude_initialized"],
                        )
                        for pd in point_data_list
                    ],
                )
    
                # ② fetch row-ids in chunks of _SQL_IN_CHUNK
                central_ids = [pd["central_point_id"] for pd in point_data_list]
                id_map      = self._fetch_point_ids(central_ids)
    
            # ③ build the result list in the caller’s original order
            return [id_map.get(cid, -1) for cid in central_ids]
    
        except sqlite3.Error as exc:
            self.logger.error("insert_point_data_batch failed: %s", exc)
            return []

    # def insert_reciprocal_space_interval_batch(
    #     self, interval_list: List[Dict]
    # ) -> List[int]:
    #     """
    #     Batch-insert ReciprocalSpaceInterval records and return their DB IDs
    #     in the same order as `interval_list`.
    #     """
    #     rs_ids: List[int] = []
    #     if not interval_list:
    #         return rs_ids

    #     try:
    #         # Prepare tuples with zeroed components for lower dimensions
    #         tuples = []
    #         for iv in interval_list:
    #             h_start, h_end = iv["h_range"]
    #             if self.dimension > 1:
    #                 k_start, k_end = iv["k_range"]
    #             else:
    #                 k_start = k_end = 0.0
    #             if self.dimension > 2:
    #                 l_start, l_end = iv["l_range"]
    #             else:
    #                 l_start = l_end = 0.0
    #             tuples.append((h_start, h_end, k_start, k_end, l_start, l_end))

    #         with self.connection:
    #             self.cursor.executemany(
    #                 """
    #                 INSERT OR IGNORE INTO ReciprocalSpaceInterval
    #                 (h_start, h_end, k_start, k_end, l_start, l_end)
    #                 VALUES (?, ?, ?, ?, ?, ?)
    #                 """,
    #                 tuples,
    #             )

    #             # Fetch IDs
    #             placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(tuples))
    #             self.cursor.execute(
    #                 f"""
    #                 SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
    #                 FROM   ReciprocalSpaceInterval
    #                 WHERE  (h_start, h_end, k_start, k_end, l_start, l_end)
    #                        IN ({placeholders})
    #                 """,
    #                 [val for t in tuples for val in t],
    #             )
    #             id_map = {tuple(r[1:]): r[0] for r in self.cursor.fetchall()}
    #             rs_ids = [id_map[t] for t in tuples]
    #         return rs_ids
    #     except sqlite3.Error as e:
    #         self.logger.error("insert_reciprocal_space_interval_batch failed: %s", e)
    #         return rs_ids


    def _pad_ranges(self, iv: Dict) -> Tuple[float, float, float, float, float, float]:
        """Return a 6-tuple [h_start, h_end, k_start, k_end, l_start, l_end]."""
        h_start, h_end = (round(x, _PRECISION) for x in iv["h_range"])
        if self.dimension > 1:
            k_start, k_end = (round(x, _PRECISION) for x in iv["k_range"])
        else:
            k_start = k_end = 0.0
        if self.dimension > 2:
            l_start, l_end = (round(x, _PRECISION) for x in iv["l_range"])
        else:
            l_start = l_end = 0.0
        return (h_start, h_end, k_start, k_end, l_start, l_end)



# ---------------------------------------------------------------------------
    
    def insert_reciprocal_space_interval_batch(
        self, interval_list: List[Dict]
    ) -> List[int]:
        """
        Return DB-ids for `interval_list` in the original order, without ever
        burning gaps in AUTOINCREMENT.  Strategy:
    
        1.  Canonicalise every interval → 6-tuple (rounded).
        2.  In CHUNK_SIZE blocks ask SQLite which tuples already exist.
        3.  Insert only the missing tuples (again in chunks).
        4.  Fetch ids for the newly inserted tuples, build a single mapping,
            and return ids preserving the caller’s order.
        """
        if not interval_list:
            return []
    
        tuples = [self._pad_ranges(iv) for iv in interval_list]
        existing: dict[Tuple[float, ...], int] = {}
    
        # -- 1) find already-present intervals ----------------------------------
        with self.connection:          # one transaction for everything
    
            for ofs in range(0, len(tuples), _CHUNK_SIZE):
                chunk = tuples[ofs : ofs + _CHUNK_SIZE]
                placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
                self.cursor.execute(
                    f"""
                    SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                    FROM   ReciprocalSpaceInterval
                    WHERE  (h_start, h_end, k_start, k_end, l_start, l_end)
                           IN ({placeholders})
                    """,
                    list(chain.from_iterable(chunk)),
                )
                existing.update({tuple(r[1:]): r[0] for r in self.cursor.fetchall()})
    
            # -- 2) insert the truly new ones -----------------------------------
            missing = [t for t in tuples if t not in existing]
    
            for ofs in range(0, len(missing), _CHUNK_SIZE):
                chunk = missing[ofs : ofs + _CHUNK_SIZE]
                self.cursor.executemany(
                    """
                    INSERT INTO ReciprocalSpaceInterval
                      (h_start, h_end, k_start, k_end, l_start, l_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    chunk,
                )
    
            # -- 3) fetch ids for the new rows ----------------------------------
            if missing:
                for ofs in range(0, len(missing), _CHUNK_SIZE):
                    chunk = missing[ofs : ofs + _CHUNK_SIZE]
                    placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
                    self.cursor.execute(
                        f"""
                        SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                        FROM   ReciprocalSpaceInterval
                        WHERE  (h_start, h_end, k_start, k_end, l_start, l_end)
                               IN ({placeholders})
                        """,
                        list(chain.from_iterable(chunk)),
                    )
                    existing.update({tuple(r[1:]): r[0] for r in self.cursor.fetchall()})
    
        # -- 4) return ids in caller’s order -------------------------------------
        return [existing[t] for t in tuples]

    def insert_interval_chunk_status_batch(
            self, status_list: List[Tuple[int, int, int | bool]]
        ):
            """
            Bulk insert (interval_id, chunk_id, saved_flag).
            `saved_flag` should be 0/1 or False/True.
            """
            if not status_list:
                return
            try:
                with self.connection:
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO Interval_Chunk_Status
                        (reciprocal_space_id, chunk_id, saved)
                        VALUES (?, ?, ?)
                        """,
                        status_list,
                    )
            except sqlite3.Error as e:
                self.logger.error("insert_interval_chunk_status_batch failed: %s", e)
    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None:
        """Kept for API compatibility – no longer needed."""
        return


    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None:
        """
        Flip the *chunk-level* flag.  If the caller still passes `point_id`
        we simply look up its chunk and forward the request.
        """
        try:
            if point_id is not None and chunk_id is None:
                self.cursor.execute("SELECT chunk_id FROM PointData WHERE id = ?", (point_id,))
                row = self.cursor.fetchone()
                if row:
                    chunk_id = row[0]
    
            if chunk_id is None:          # nothing to do
                return
    
            self.update_interval_chunk_status(reciprocal_space_id, chunk_id, saved)
    
        except sqlite3.Error as exc:
            self.logger.error("update_saved_status_for_chunk_or_point failed: %s", exc)
    
    
    def get_unsaved_associations(self) -> list[tuple[int, int]]:
        """
        Return *every* (central_point_id, interval_id) pair that still needs
        to be processed, but **construct it on-the-fly** from chunk metadata.
        """
        try:
            # ① all (interval, chunk) that are still incomplete
            self.cursor.execute(
                """
                SELECT reciprocal_space_id, chunk_id
                FROM   Interval_Chunk_Status
                WHERE  saved = 0
                """
            )
            todo_pairs = self.cursor.fetchall()       # list[(interval_id, chunk_id)]
    
            if not todo_pairs:
                return []
    
            # ② build one big result by joining PointData once per chunk
            result: list[tuple[int, int]] = []
            for interval_id, chunk_id in todo_pairs:
                self.cursor.execute(
                    "SELECT central_point_id FROM PointData WHERE chunk_id = ?",
                    (chunk_id,),
                )
                result.extend((pid, interval_id) for (pid,) in self.cursor.fetchall())
    
            return result
    
        except sqlite3.Error as exc:
            self.logger.error("get_unsaved_associations failed: %s", exc)
            return []
    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ):
        """
        Mark a single (interval_id, chunk_id) pair as saved / unsaved.
        """
        try:
            with self.connection:
                self.cursor.execute(
                    """
                    INSERT INTO Interval_Chunk_Status
                    (reciprocal_space_id, chunk_id, saved)
                    VALUES (?, ?, ?)
                    ON CONFLICT(reciprocal_space_id, chunk_id)
                    DO UPDATE SET saved = excluded.saved
                    """,
                    (interval_id, chunk_id, int(saved)),
                )
        except sqlite3.Error as e:
            self.logger.error("update_interval_chunk_status failed: %s", e)

    def get_unsaved_interval_chunks(self) -> List[Tuple[int, int]]:
        """
        Return every (interval_id, chunk_id) where saved = 0.
        """
        try:
            self.cursor.execute(
                """
                SELECT reciprocal_space_id, chunk_id
                FROM   Interval_Chunk_Status
                WHERE  saved = 0
                """
            )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error("get_unsaved_interval_chunks failed: %s", e)
            return []
    def mark_interval_precomputed(self, interval_id: int, done: bool = True):
        with self.connection:
            self.cursor.execute(
                "UPDATE ReciprocalSpaceInterval SET precomputed=? WHERE id=?",
                (1 if done else 0, interval_id)
            )
            
    def is_interval_precomputed(self, interval_id: int) -> bool:
        self.cursor.execute(
            "SELECT precomputed FROM ReciprocalSpaceInterval WHERE id=?",
            (interval_id,)
        )
        row = self.cursor.fetchone()
        return bool(row and row[0])        
    # -------------------------------------------------------------- #
    #  CLOSE                                                         #
    # -------------------------------------------------------------- #
    def close(self):
        self.connection.close()
        self.logger.info("DB connection closed.")
