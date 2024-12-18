# managers/database_manager.py

import sqlite3
import json
import os
import logging
from typing import List, Dict, Tuple

class DatabaseManager:
    def __init__(self, db_path: str, dimension: int = 3):
        """
        Initializes the DatabaseManager.

        Args:
            db_path (str): Path to the SQLite database file.
            dimension (int): Dimension of data (1, 2, or 3).
                1D: Only h_range is used.
                2D: h_range and k_range are used.
                3D: h_range, k_range, and l_range are used.
        """
        self.db_path = db_path
        self.dimension = dimension
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._initialize_database()

    def get_pending_chunk_ids(self) -> List[int]:
        """
        Retrieves unique chunk_ids from PointData that are pending processing.

        Returns:
            List[int]: List of unique chunk_ids.
        """
        try:
            self.cursor.execute("""
                SELECT DISTINCT chunk_id
                FROM PointData
            """)
            rows = self.cursor.fetchall()
            chunk_ids = [row[0] for row in rows]
            self.logger.debug(f"Retrieved {len(chunk_ids)} unique chunk_ids for processing.")
            return chunk_ids
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve chunk_ids: {e}")
            return []
        
    def get_pending_parts(self) -> List[Dict]:
        """
        Retrieves pending ReciprocalSpaceInterval entries that need processing.

        Based on the dimension, we only return the ranges that apply.
        
        Returns:
            List[Dict]: List of interval dictionaries with dimension-appropriate keys.
        """
        try:
            self.cursor.execute("""
                SELECT id, h_start, h_end, k_start, k_end, l_start, l_end
                FROM ReciprocalSpaceInterval
            """)
            rows = self.cursor.fetchall()
            pending_parts = []
            for row in rows:
                interval_id = row[0]
                h_range = [row[1], row[2]]
                k_range = [row[3], row[4]] if self.dimension > 1 else [0.0, 0.0]
                l_range = [row[5], row[6]] if self.dimension > 2 else [0.0, 0.0]

                interval_dict = {'id': interval_id, 'h_range': h_range}
                if self.dimension > 1:
                    interval_dict['k_range'] = k_range
                if self.dimension > 2:
                    interval_dict['l_range'] = l_range
                pending_parts.append(interval_dict)

            self.logger.debug(f"Retrieved {len(pending_parts)} ReciprocalSpaceInterval entries for processing.")
            return pending_parts
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve ReciprocalSpaceInterval entries: {e}")
            return []

    def get_point_data_for_point_ids(self, point_ids: List[int]) -> List[Dict]:
        """
        Retrieves point data for the given list of point_ids.

        Args:
            point_ids (List[int]): List of point_ids to retrieve data for.

        Returns:
            List[Dict]: List of dictionaries containing point data.
        """
        if not point_ids:
            return []

        placeholders = ','.join(['?'] * len(point_ids))
        query = f"""
            SELECT central_point_id, coordinates, dist_from_atom_center, step_in_frac, chunk_id, grid_amplitude_initialized
            FROM PointData
            WHERE id IN ({placeholders})
        """

        try:
            self.cursor.execute(query, point_ids)
            rows = self.cursor.fetchall()
            point_data_list = []
            for row in rows:
                point_data = {
                    'central_point_id': row[0],
                    'coordinates': json.loads(row[1]),
                    'dist_from_atom_center': json.loads(row[2]),
                    'step_in_frac': json.loads(row[3]),
                    'chunk_id': row[4],
                    'grid_amplitude_initialized': row[5],
                    'id': row[0]  # Assuming central_point_id is unique and serves as 'id'
                }
                point_data_list.append(point_data)
            self.logger.debug(f"Retrieved point data for {len(point_data_list)} points.")
            return point_data_list
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve point data for point_ids={point_ids}: {e}")
            return []

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

            # Create ReciprocalSpaceInterval table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS ReciprocalSpaceInterval (
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

            # Create Point_ReciprocalSpace_Association table with `saved` boolean field
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS Point_ReciprocalSpace_Association (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    point_id INTEGER,
                    reciprocal_space_id INTEGER,
                    saved INTEGER DEFAULT 0,
                    FOREIGN KEY (point_id) REFERENCES PointData(id),
                    FOREIGN KEY (reciprocal_space_id) REFERENCES ReciprocalSpaceInterval(id),
                    UNIQUE(point_id, reciprocal_space_id)
                )
            """)

            # Create indexes for faster lookups
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pointdata_central_point_id ON PointData (central_point_id);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_reciprocal_spaceinterval_ranges ON ReciprocalSpaceInterval (h_start, h_end, k_start, k_end, l_start, l_end);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_association_point_id ON Point_ReciprocalSpace_Association (point_id);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_association_reciprocal_space_id ON Point_ReciprocalSpace_Association (reciprocal_space_id);")

            # Commit changes
            self.connection.commit()
            self.logger.info("Database tables and indexes initialized successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred initializing the database: {e}")

    def update_saved_status_for_chunk_or_point(self, reciprocal_space_id: int, point_id: int = None, chunk_id: int = None, saved: int = 0):
        """
        Updates the `saved` status for associations based on either `chunk_id` or `point_id`, combined with `reciprocal_space_id`.
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            if point_id is not None:
                self.cursor.execute("""
                    UPDATE Point_ReciprocalSpace_Association
                    SET saved = ?
                    WHERE point_id = ? AND reciprocal_space_id = ?
                """, (saved, point_id, reciprocal_space_id))
                self.logger.debug(f"Updated saved status for PointData ID={point_id} and ReciprocalSpaceInterval ID={reciprocal_space_id}.")
            elif chunk_id is not None:
                # Find all point_ids associated with the given chunk_id
                self.cursor.execute("""
                    SELECT id FROM PointData
                    WHERE chunk_id = ?
                """, (chunk_id,))
                point_ids = [row[0] for row in self.cursor.fetchall()]

                # Update all associations for the given chunk_id
                self.cursor.executemany("""
                    UPDATE Point_ReciprocalSpace_Association
                    SET saved = ?
                    WHERE point_id = ? AND reciprocal_space_id = ?
                """, [(saved, pid, reciprocal_space_id) for pid in point_ids])

                self.logger.debug(f"Updated saved status for chunk_id={chunk_id} and ReciprocalSpaceInterval ID={reciprocal_space_id} for {len(point_ids)} PointData entries.")

            # Commit changes
            self.connection.commit()
            self.logger.info("Saved status updated successfully.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to update saved status for reciprocal_space_id={reciprocal_space_id}, point_id={point_id}, chunk_id={chunk_id}: {e}")
            self.connection.rollback()

    def insert_point_data_batch(self, point_data_list: List[Dict]) -> List[int]:
        """
        Inserts multiple PointData entries into the database.
        """
        point_ids = []
        try:
            # Prepare data for insertion
            insert_data = []
            for pd in point_data_list:
                coordinates_json = json.dumps(pd['coordinates'])
                dist_json = json.dumps(pd['dist_from_atom_center'])
                step_json = json.dumps(pd['step_in_frac'])
            
                insert_data.append((
                    pd['central_point_id'],
                    coordinates_json,   # Only coordinates data
                    dist_json,          # Only dist_from_atom_center data
                    step_json,          # Only step_in_frac data
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

    def insert_reciprocal_space_interval_batch(self, reciprocal_space_interval_list: List[Dict]) -> List[int]:
        """
        Inserts multiple ReciprocalSpaceInterval entries into the database, adapted for dimension.

        For dimension < 3, we store l_range as 0.0,0.0
        For dimension < 2, we store k_range as 0.0,0.0
        """
        reciprocal_space_ids = []
        try:
            # Prepare data for insertion
            insert_data = []
            for interval in reciprocal_space_interval_list:
                h_start, h_end = interval['h_range']
                if self.dimension > 1:
                    k_start, k_end = interval['k_range']
                else:
                    k_start, k_end = 0.0, 0.0
                if self.dimension > 2:
                    l_start, l_end = interval['l_range']
                else:
                    l_start, l_end = 0.0, 0.0

                insert_data.append((h_start, h_end, k_start, k_end, l_start, l_end))

            # Begin transaction
            self.connection.execute('BEGIN')

            # Insert ReciprocalSpaceInterval entries, ignoring duplicates
            self.cursor.executemany("""
                INSERT OR IGNORE INTO ReciprocalSpaceInterval (
                    h_start, h_end, k_start, k_end, l_start, l_end
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, insert_data)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Inserted or ignored {len(insert_data)} ReciprocalSpaceInterval entries.")

            # Retrieve IDs for all inserted intervals
            placeholders = ','.join(['(?, ?, ?, ?, ?, ?)'] * len(insert_data))
            query = f"""
                SELECT id, h_start, h_end, k_start, k_end, l_start, l_end FROM ReciprocalSpaceInterval
                WHERE (h_start, h_end, k_start, k_end, l_start, l_end) IN ({placeholders})
            """
            flat_insert_data = [item for sublist in insert_data for item in sublist]
            self.cursor.execute(query, flat_insert_data)
            results = self.cursor.fetchall()

            # Map interval tuples to their IDs
            interval_map = {tuple(r[1:]): r[0] for r in results}

            # Assign IDs based on the interval data
            for interval in reciprocal_space_interval_list:
                h_start, h_end = interval['h_range']
                if self.dimension > 1:
                    k_start, k_end = interval['k_range']
                else:
                    k_start, k_end = 0.0, 0.0
                if self.dimension > 2:
                    l_start, l_end = interval['l_range']
                else:
                    l_start, l_end = 0.0, 0.0

                key = (h_start, h_end, k_start, k_end, l_start, l_end)
                reciprocal_space_id = interval_map.get(key)
                if reciprocal_space_id:
                    reciprocal_space_ids.append(reciprocal_space_id)
                else:
                    self.logger.warning(f"ReciprocalSpaceInterval {interval} was not inserted or found.")

            return reciprocal_space_ids

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch insert ReciprocalSpaceIntervals: {e}")
            self.connection.rollback()
            return reciprocal_space_ids

    def associate_point_reciprocal_space_batch(self, associations: List[Tuple[int, int]]):
        """
        Associates multiple PointData entries with ReciprocalSpaceInterval entries in bulk.
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            # Insert associations, ignoring duplicates
            self.cursor.executemany("""
                INSERT OR IGNORE INTO Point_ReciprocalSpace_Association (
                    point_id,
                    reciprocal_space_id,
                    saved
                ) VALUES (?, ?, 0)
            """, associations)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Associated {len(associations)} PointData and ReciprocalSpaceInterval pairs.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch associate PointData with ReciprocalSpaceIntervals: {e}")
            self.connection.rollback()

    def update_saved_status_batch(self, updates: List[Tuple[int, int, int]]):
        """
        Updates the saved status for multiple Point_ReciprocalSpace_Association entries in bulk.
        """
        try:
            # Begin transaction
            self.connection.execute('BEGIN')

            self.cursor.executemany("""
                UPDATE Point_ReciprocalSpace_Association
                SET saved = ?
                WHERE point_id = ? AND reciprocal_space_id = ?
            """, updates)

            # Commit the transaction
            self.connection.commit()
            self.logger.debug(f"Updated saved status for {len(updates)} associations.")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to batch update saved status: {e}")
            self.connection.rollback()
            
    def get_unsaved_associations(self) -> List[Tuple[int, int]]:
        """
        Retrieves (point_id, reciprocal_space_id) tuples for all (chunk_id, reciprocal_space_id) pairs
        where ALL associations are unsaved (saved=0).
        
        Returns:
            List[Tuple[int, int]]: A list of (point_id, reciprocal_space_id) that are fully unsaved
            within their respective (chunk_id, reciprocal_space_id) group.
        """
        try:
            # 1. Identify fully unsaved (chunk_id, reciprocal_space_id) pairs
            self.cursor.execute("""
                SELECT pd.chunk_id,
                       pha.reciprocal_space_id,
                       COUNT(*) AS total_count,
                       SUM(CASE WHEN pha.saved = 0 THEN 1 ELSE 0 END) AS unsaved_count
                FROM Point_ReciprocalSpace_Association pha
                JOIN PointData pd ON pha.point_id = pd.id
                GROUP BY pd.chunk_id, pha.reciprocal_space_id
            """)
            
            rows = self.cursor.fetchall()
            fully_unsaved_pairs = [(r[0], r[1]) for r in rows if r[2] == r[3] and r[2] > 0]
    
            if not fully_unsaved_pairs:
                self.logger.debug("No (chunk_id, reciprocal_space_id) pairs are fully unsaved.")
                return []
            
            # 2. Retrieve all unsaved (point_id, reciprocal_space_id) for these fully unsaved pairs
            conditions = []
            params = []
            for chunk_id, reciprocal_space_id in fully_unsaved_pairs:
                conditions.append("(pd.chunk_id = ? AND pha.reciprocal_space_id = ?)")
                params.extend([chunk_id, reciprocal_space_id])
            
            where_clause = " OR ".join(conditions)
            query = f"""
                SELECT pha.point_id, pha.reciprocal_space_id
                FROM Point_ReciprocalSpace_Association pha
                JOIN PointData pd ON pha.point_id = pd.id
                WHERE pha.saved = 0
                  AND ({where_clause})
            """
    
            self.cursor.execute(query, params)
            all_unsaved_associations = self.cursor.fetchall()
    
            self.logger.debug(
                f"Retrieved {len(all_unsaved_associations)} unsaved associations "
                f"for {len(fully_unsaved_pairs)} fully unsaved (chunk_id, reciprocal_space_id) pairs."
            )
            
            return all_unsaved_associations
    
        except sqlite3.Error as e:
            self.logger.error(f"Database error while retrieving consistently unsaved associations: {e}")
            return []

    def close(self):
        """
        Closes the database connection.
        """
        self.connection.close()
        self.logger.info("Database connection closed.")
