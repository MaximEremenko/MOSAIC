# managers/database_manager.py

import sqlite3
import json
import os
import logging
from typing import List, Dict, Tuple

class DatabaseManager:
    def __init__(self, db_path: str):
        """
        Initializes the DatabaseManager.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._initialize_database()

    def _initialize_database(self):
        """
        Creates tables if they do not exist.
        """
        try:
            # Create PointData table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS PointData (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    central_point_id INTEGER UNIQUE,
                    coordinates TEXT,
                    dist_from_atom_center TEXT,
                    step_in_frac TEXT,
                    chunk_id INTEGER,
                    grid_amplitude_initialized INTEGER
                )
            """)

            # Create HKLInterval table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS HKLInterval (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    h_start REAL,
                    h_end REAL,
                    k_start REAL,
                    k_end REAL,
                    l_start REAL,
                    l_end REAL,
                    UNIQUE(h_start, h_end, k_start, k_end, l_start, l_end)
                )
            """)

            # Create Point_HKL_Association table with `saved` boolean field
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS Point_HKL_Association (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    point_id INTEGER,
                    hkl_id INTEGER,
                    saved INTEGER DEFAULT 0,
                    FOREIGN KEY (point_id) REFERENCES PointData(id),
                    FOREIGN KEY (hkl_id) REFERENCES HKLInterval(id),
                    UNIQUE(point_id, hkl_id)
                )
            """)

            # Create indexes for faster lookups
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pointdata_central_point_id ON PointData (central_point_id);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_hklinterval_ranges ON HKLInterval (h_start, h_end, k_start, k_end, l_start, l_end);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_association_point_id ON Point_HKL_Association (point_id);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_association_hkl_id ON Point_HKL_Association (hkl_id);")

            # Commit changes
            self.connection.commit()
            self.logger.info("Database tables and indexes initialized successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred initializing the database: {e}")

    def update_saved_status_for_chunk_or_point(self, hkl_id: int, point_id: int = None, chunk_id: int = None, saved: int = 1):
        """
        Updates the `saved` status for associations based on either `chunk_id` or `point_id`, combined with `hkl_id`.

        Args:
            hkl_id (int): The ID of the HKLInterval.
            point_id (int, optional): The ID of the PointData. Default is None.
            chunk_id (int, optional): The chunk ID for updating based on `chunk_id`. Default is None.
            saved (int, optional): The status value for `saved`. Default is 1 (indicating saved).
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            # Update by point_id if provided
            if point_id is not None:
                self.cursor.execute("""
                    UPDATE Point_HKL_Association
                    SET saved = ?
                    WHERE point_id = ? AND hkl_id = ?
                """, (saved, point_id, hkl_id))
                self.logger.debug(f"Updated saved status for PointData ID={point_id} and HKLInterval ID={hkl_id}.")

            # Update by chunk_id if provided
            elif chunk_id is not None:
                # First, find all point_ids associated with the given chunk_id
                self.cursor.execute("""
                    SELECT id FROM PointData
                    WHERE chunk_id = ?
                """, (chunk_id,))
                point_ids = [row[0] for row in self.cursor.fetchall()]

                # Update all associations for the given chunk_id
                self.cursor.executemany("""
                    UPDATE Point_HKL_Association
                    SET saved = ?
                    WHERE point_id = ? AND hkl_id = ?
                """, [(saved, point_id, hkl_id) for point_id in point_ids])

                self.logger.debug(f"Updated saved status for chunk_id={chunk_id} and HKLInterval ID={hkl_id} for {len(point_ids)} PointData entries.")

            # Commit changes
            self.connection.commit()
            self.logger.info("Saved status updated successfully.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to update saved status for hkl_id={hkl_id}, point_id={point_id}, chunk_id={chunk_id}: {e}")
            self.connection.rollback()

    def insert_point_data_batch(self, point_data_list: List[Dict]) -> List[int]:
        """
        Inserts multiple PointData entries into the database.

        Args:
            point_data_list (List[Dict]): List of dictionaries containing PointData information.

        Returns:
            List[int]: List of PointData IDs (inserted or existing).
        """
        point_ids = []
        try:
            # Prepare data for insertion
            insert_data = []
            for pd in point_data_list:
                data_json = json.dumps({
                    'coordinates': pd['coordinates'],
                    'dist_from_atom_center': pd['dist_from_atom_center'],
                    'step_in_frac': pd['step_in_frac']
                })
                insert_data.append((
                    pd['central_point_id'],
                    data_json,
                    data_json,  # Assuming same structure; adjust if necessary
                    data_json,  # Assuming same structure; adjust if necessary
                    pd['chunk_id'],
                    pd['grid_amplitude_initialized']
                ))

            # Begin transaction
            self.connection.execute('BEGIN')
            
            # Insert PointData entries, ignoring duplicates
            self.cursor.executemany("""
                INSERT OR IGNORE INTO PointData (
                    central_point_id,
                    coordinates,
                    dist_from_atom_center,
                    step_in_frac,
                    chunk_id,
                    grid_amplitude_initialized
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, insert_data)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Inserted or ignored {len(insert_data)} PointData entries.")

            # Retrieve IDs for all central_point_ids
            central_point_ids = [pd['central_point_id'] for pd in point_data_list]
            placeholders = ','.join(['?'] * len(central_point_ids))
            query = f"SELECT central_point_id, id FROM PointData WHERE central_point_id IN ({placeholders})"
            self.cursor.execute(query, central_point_ids)
            results = self.cursor.fetchall()
            point_id_map = {cp_id: pid for cp_id, pid in results}

            # Map point_ids to the list
            for cp_id in central_point_ids:
                point_id = point_id_map.get(cp_id)
                if point_id:
                    point_ids.append(point_id)
                else:
                    self.logger.warning(f"PointData with central_point_id={cp_id} was not inserted or found.")

            return point_ids

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch insert PointData: {e}")
            self.connection.rollback()
            return point_ids

    def insert_hkl_interval_batch(self, hkl_interval_list: List[Dict]) -> List[int]:
        """
        Inserts multiple HKLInterval entries into the database.

        Args:
            hkl_interval_list (List[Dict]): List of dictionaries containing HKLInterval information.

        Returns:
            List[int]: List of HKLInterval IDs (inserted or existing).
        """
        hkl_ids = []
        try:
            # Prepare data for insertion
            insert_data = []
            for interval in hkl_interval_list:
                insert_data.append((
                    interval['h_range'][0],
                    interval['h_range'][1],
                    interval['k_range'][0],
                    interval['k_range'][1],
                    interval['l_range'][0],
                    interval['l_range'][1]
                ))

            # Begin transaction
            self.connection.execute('BEGIN')

            # Insert HKLInterval entries, ignoring duplicates
            self.cursor.executemany("""
                INSERT OR IGNORE INTO HKLInterval (
                    h_start, h_end, k_start, k_end, l_start, l_end
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, insert_data)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Inserted or ignored {len(insert_data)} HKLInterval entries.")

            # Retrieve IDs for all inserted intervals
            query = """
                SELECT id, h_start, h_end, k_start, k_end, l_start, l_end FROM HKLInterval
                WHERE (h_start, h_end, k_start, k_end, l_start, l_end) IN (
                    """ + ','.join(['(?, ?, ?, ?, ?, ?)'] * len(hkl_interval_list)) + ")"

            flat_insert_data = [item for sublist in insert_data for item in sublist]
            self.cursor.execute(query, flat_insert_data)
            results = self.cursor.fetchall()

            # Map interval tuples to their IDs
            interval_map = {tuple(row[1:]): row[0] for row in results}

            # Assign IDs based on the interval data
            for interval in hkl_interval_list:
                key = tuple(interval['h_range'] + interval['k_range'] + interval['l_range'])
                hkl_id = interval_map.get(key)
                if hkl_id:
                    hkl_ids.append(hkl_id)
                else:
                    self.logger.warning(f"HKLInterval {interval} was not inserted or found.")

            return hkl_ids

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch insert HKLIntervals: {e}")
            self.connection.rollback()
            return hkl_ids

    def associate_point_hkl_batch(self, associations: List[Tuple[int, int]]):
        """
        Associates multiple PointData entries with HKLInterval entries in bulk.

        Args:
            associations (List[Tuple[int, int]]): List of (point_id, hkl_id) tuples.
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            # Prepare data for insertion
            insert_data = associations

            # Insert associations, ignoring duplicates
            self.cursor.executemany("""
                INSERT OR IGNORE INTO Point_HKL_Association (
                    point_id,
                    hkl_id,
                    saved
                ) VALUES (?, ?, 0)
            """, insert_data)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Associated {len(insert_data)} PointData and HKLInterval pairs.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch associate PointData with HKLIntervals: {e}")
            self.connection.rollback()

    def update_saved_status_batch(self, updates: List[Tuple[int, int, int]]):
        """
        Updates the saved status for multiple Point_HKL_Association entries in bulk.

        Args:
            updates (List[Tuple[int, int, int]]): List of (saved, point_id, hkl_id) tuples.
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            # Update saved status
            self.cursor.executemany("""
                UPDATE Point_HKL_Association
                SET saved = ?
                WHERE point_id = ? AND hkl_id = ?
            """, updates)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Updated saved status for {len(updates)} associations.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch update saved status: {e}")
            self.connection.rollback()

    def get_unsaved_associations(self) -> List[Tuple[int, int]]:
        """
        Retrieves all Point_HKL_Association entries that have not been saved.

        Returns:
            List[Tuple[int, int]]: List of (point_id, hkl_id) tuples.
        """
        try:
            self.cursor.execute("""
                SELECT point_id, hkl_id
                FROM Point_HKL_Association
                WHERE saved = 0
            """)
            rows = self.cursor.fetchall()
            self.logger.debug(f"Retrieved {len(rows)} unsaved associations.")
            return rows
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve unsaved associations: {e}")
            return []

    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()
        self.logger.info("Database connection closed.")
