# processors/from_average_point_processor.py
from typing import Optional
import logging
import os
import h5py
import numpy as np
from core.structure import angstrom_to_fractional
from core.models import PointData


class FromAveragePointProcessor:
    def __init__(self, parameters: dict, average_structure: dict, num_chunks: int = 10):
        self.parameters = parameters
        self.average_structure = average_structure
        self.num_chunks = num_chunks
        self.point_data: PointData = None  # Includes central_point_ids and grid_amplitude_initialized
        self.metric = average_structure.get('metric')
        self.vectors = average_structure.get('vectors')
        self.hdf5_file_path = self.parameters.get('hdf5_file_path', 'point_data.hdf5')
        self.logger = logging.getLogger(__name__)

    # ------------------------- NEW HELPERS (cell filtering) -------------------------

    def _attach_cell_ids(self, df):
        """
        Ensure df has integer columns: cell_x, cell_y, cell_z.
        Pulls from self.average_structure['cell_ids'] which can be a DataFrame/Series/ndarray.
        If absent or shape-mismatched, logs a warning and returns df unchanged.
        """
        if all(c in df.columns for c in ("cell_x", "cell_y", "cell_z")):
            return df  # already present

        cells = self.average_structure.get("cell_ids", None)
        if cells is None:
            self.logger.warning("average_structure.cell_ids missing; not filtering by cell box.")
            return df

        # Normalize to ndarray (N, 3) aligned with df.index length
        if hasattr(cells, "to_numpy"):  # pandas object
            arr = cells.to_numpy()
            # If it's a Series of tuples/lists, expand
            if arr.ndim == 1 and len(arr) == len(df):
                try:
                    arr = np.vstack(arr)
                except Exception:
                    pass
        else:
            arr = np.asarray(cells)

        if not (arr.ndim == 2 and arr.shape[0] == len(df) and arr.shape[1] >= 3):
            self.logger.warning(
                "average_structure.cell_ids has unexpected shape %s; "
                "expected (N,3). Skipping cell filtering.", getattr(arr, "shape", None)
            )
            return df

        df = df.copy()
        df["cell_x"] = arr[:, 0].astype(int)
        df["cell_y"] = arr[:, 1].astype(int)
        df["cell_z"] = arr[:, 2].astype(int)
        return df

    def _filter_by_cell_limits(self, df):
        """
        Filter df rows to inclusive [cells_limits_min, cells_limits_max] box.
        Requires columns: cell_x, cell_y, cell_z produced by _attach_cell_ids.
        If limits not provided, returns df unchanged.
        """
        rspace = self.parameters.get("rspace_info", {}) or {}
        if "cells_limits_min" not in rspace or "cells_limits_max" not in rspace:
            return df

        try:
            lim_min = np.asarray(rspace["cells_limits_min"], dtype=int)
            lim_max = np.asarray(rspace["cells_limits_max"], dtype=int)
        except Exception:
            self.logger.warning("cells_limits_* malformed; skipping cell filtering.")
            return df

        # Ensure columns exist
        for col in ("cell_x", "cell_y", "cell_z"):
            if col not in df.columns:
                self.logger.warning("Missing %s in DataFrame; skipping cell filtering.", col)
                return df

        cx, cy, cz = df["cell_x"], df["cell_y"], df["cell_z"]

        # Pad limits if somehow fewer than 3 provided (robustness)
        if lim_min.size < 3:
            lim_min = np.pad(lim_min, (0, 3 - lim_min.size), constant_values=-np.iinfo(np.int64).max)
        if lim_max.size < 3:
            lim_max = np.pad(lim_max, (0, 3 - lim_max.size), constant_values=+np.iinfo(np.int64).max)

        mask = (
            (cx >= lim_min[0]) & (cx <= lim_max[0]) &
            (cy >= lim_min[1]) & (cy <= lim_max[1]) &
            (cz >= lim_min[2]) & (cz <= lim_max[2])
        )
        kept = int(mask.sum())
        dropped = int((~mask).sum())
        self.logger.info(
            "Cell-box filter: kept %d points, dropped %d outside [%s .. %s].",
            kept, dropped, lim_min.tolist(), lim_max.tolist()
        )
        return df.loc[mask]

    # ------------------------------------------------------------------------------

    def process_parameters(self):
        # Extract 'fresh_start' flag from rspace_info
        rspace_info = self.parameters.get('rspace_info', {})
        fresh_start = rspace_info.get('fresh_start', True)  # Default to True if not specified

        # If fresh start requested and file exists → remove it so datasets are recreated chunked/extendible
        if fresh_start and os.path.exists(self.hdf5_file_path):
            try:
                os.remove(self.hdf5_file_path)
                self.logger.info("Removed existing HDF5 for fresh_start: %s", self.hdf5_file_path)
            except Exception as e:
                self.logger.warning("Failed to remove %s for fresh_start: %s", self.hdf5_file_path, e)

        # Check if HDF5 file exists and handle accordingly
        if os.path.exists(self.hdf5_file_path) and not fresh_start:
            self.logger.info(f"HDF5 file found: {self.hdf5_file_path}. Attempting to load existing data for appending.")
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                self.logger.info(f"Existing point data loaded from {self.hdf5_file_path}. New points will be appended.")
            else:
                self.logger.warning(f"Failed to load existing point data from {self.hdf5_file_path}. Proceeding to create new data.")
                fresh_start = True  # Fallback to fresh start if loading fails

        if fresh_start or self.point_data is None:
            self.logger.info("Processing parameters for 'from_average' method with fresh start.")
            # Initialize empty PointData
            self.point_data = PointData(
                coordinates=np.empty((0, 3)),
                dist_from_atom_center=np.empty((0, 3)),
                step_in_frac=np.empty((0, 3)),
                central_point_ids=np.empty((0,), dtype=int),
                chunk_ids=np.empty((0,), dtype=int),
                grid_amplitude_initialized=np.empty((0,), dtype=bool)
            )

        # Proceed to process parameters
        self.logger.info("Processing parameters for 'from_average' method.")
        points_params = rspace_info.get('points', [])

        average_coords = self.average_structure.get('average_coords')
        elements = self.average_structure.get('elements')
        refnumbers = self.average_structure.get('refnumbers')

        if average_coords is None or elements is None or refnumbers is None:
            self.logger.error("Average structure data is missing required components.")
            return

        # Combine average_coords, elements, and refnumbers into a DataFrame
        structure_df = average_coords.copy()
        structure_df['element'] = elements
        structure_df['refnumbers'] = refnumbers

        # 🔹 Attach cell indices and filter to requested [min..max] inclusive box
        structure_df = self._attach_cell_ids(structure_df)
        structure_df = self._filter_by_cell_limits(structure_df)

        all_coordinates = []
        all_dist_from_atom_center = []
        all_step_in_frac = []
        all_central_point_ids = []  # New list to store original point IDs

        for point_info in points_params:
            element_symbol = point_info.get('elementSymbol')
            reference_number = point_info.get('referenceNumber')
            dist_from_atom_center = point_info.get('distFromAtomCenter')
            step_in_angstrom = point_info.get('stepInAngstrom')

            if None in (element_symbol, reference_number, dist_from_atom_center, step_in_angstrom):
                self.logger.error("Missing required point parameters.")
                continue

            # Create a mask to filter rows where 'refnumbers' and 'element' match (within cell box)
            mask = (structure_df['refnumbers'] == reference_number) & (structure_df['element'] == element_symbol)
            matched_rows = structure_df[mask]

            if not matched_rows.empty:
                dimension = len(dist_from_atom_center)
                # Possible sets of coordinate columns
                coords_lower = ['x', 'y', 'z']
                coords_av    = ['Xav', 'Yav', 'Zav']

                # Figure out which column names exist in the DataFrame
                cols = matched_rows.columns

                if all(col in cols for col in coords_lower[:dimension]):
                    coord_columns = coords_lower[:dimension]
                elif all(col in cols for col in coords_av[:dimension]):
                    coord_columns = coords_av[:dimension]
                else:
                    raise ValueError(
                        f"Neither {coords_lower[:dimension]} nor {coords_av[:dimension]} "
                        "columns are all present in the DataFrame."
                    )

                # Extract the numpy array of selected columns
                central_points = matched_rows[coord_columns].values  # shape: (N, dimension)

                # central_points are in fractional coordinates already
                fractional_coords = central_points

                # Convert dist_from_atom_center and step_in_angstrom to fractional units if needed
                dist_from_atom_center_array = np.array(dist_from_atom_center).reshape(1, -1)  # (1, D)
                step_in_angstrom_array = np.array(step_in_angstrom).reshape(1, -1)            # (1, D)

                # If you need to convert to fractional, uncomment:
                # fractional_dist = angstrom_to_fractional(dist_from_atom_center_array, self.vectors)
                # fractional_step = angstrom_to_fractional(step_in_angstrom_array, self.vectors)
                fractional_dist = dist_from_atom_center_array
                fractional_step = step_in_angstrom_array

                # Repeat dist and step for each central point
                num_points = fractional_coords.shape[0]
                all_coordinates.append(fractional_coords)
                all_dist_from_atom_center.append(np.tile(fractional_dist, (num_points, 1)))  # (num_points, D)
                all_step_in_frac.append(np.tile(fractional_step, (num_points, 1)))           # (num_points, D)

                # Collect the original point IDs (assuming DataFrame index)
                central_point_ids = matched_rows.index.values  # (N,)
                all_central_point_ids.append(central_point_ids)
            else:
                self.logger.warning(
                    "Reference atom not found (or filtered out by cell box) "
                    f"for element {element_symbol} and reference number {reference_number}"
                )

        if all_coordinates:
            # Concatenate all data
            coordinates_array = np.vstack(all_coordinates)                             # (Total_N, D)
            dist_from_atom_center_array = np.vstack(all_dist_from_atom_center)         # (Total_N, D)
            step_in_frac_array = np.vstack(all_step_in_frac)                           # (Total_N, D)
            central_point_ids_array = np.concatenate(all_central_point_ids)            # (Total_N,)
            grid_amplitude_initialized_array = np.zeros(central_point_ids_array.shape[0], dtype=bool)

            # Create a new PointData instance
            new_point_data = PointData(
                coordinates=coordinates_array,
                dist_from_atom_center=dist_from_atom_center_array,
                step_in_frac=step_in_frac_array,
                central_point_ids=central_point_ids_array,
                chunk_ids=np.empty((central_point_ids_array.shape[0],), dtype=int),  # Placeholder, to assign later
                grid_amplitude_initialized=grid_amplitude_initialized_array
            )

            # Assign chunk_ids to new points based on num_chunks
            self._assign_chunk_ids(new_point_data)

            # Merge new_point_data with existing self.point_data
            self._merge_point_data(new_point_data)
        else:
            self.logger.error("No point data generated after filtering and matching.")

        # Save to HDF5
        self.save_point_data_to_hdf5(self.hdf5_file_path)

    def _assign_chunk_ids(self, new_point_data: PointData, max_chunk_size: Optional[int] = None):
        """
        Assigns chunk_ids to new points based on the specified number of chunks.

        Args:
            new_point_data (PointData): The new point data to assign chunk_ids to.
            max_chunk_size (int, optional): If provided, overrides num_chunks by specifying maximum points per chunk.
        """
        num_new_points = new_point_data.coordinates.shape[0]
        num_chunks = self.num_chunks

        if max_chunk_size:
            num_chunks = int(np.ceil(num_new_points / max_chunk_size))
            self.logger.debug(f"Overriding num_chunks to {num_chunks} based on max_chunk_size={max_chunk_size}")

        self.logger.debug(f"Assigning chunk_ids to {num_new_points} new points with num_chunks={num_chunks}")

        if self.point_data is None or self.point_data.chunk_ids.size == 0:
            current_max_chunk_id = -1
        else:
            current_max_chunk_id = self.point_data.chunk_ids.max()

        # Calculate the size of each chunk
        base_chunk_size = num_new_points // num_chunks
        remainder = num_new_points % num_chunks

        chunk_sizes = [base_chunk_size + 1 if i < remainder else base_chunk_size for i in range(num_chunks)]
        self.logger.debug(f"Chunk sizes: {chunk_sizes}")

        start = 0
        for i, size in enumerate(chunk_sizes):
            end = start + size
            chunk_id = current_max_chunk_id + 1 + i
            new_point_data.chunk_ids[start:end] = chunk_id
            self.logger.debug(f"Assigned chunk_id={chunk_id} to points {start} to {end-1}")
            start = end

    def _merge_point_data(self, new_point_data: PointData):
        """
        Merges new_point_data with existing self.point_data, avoiding duplicates.

        Args:
            new_point_data (PointData): The new point data to merge.
        """
        if self.point_data is None or self.point_data.central_point_ids.size == 0:
            self.point_data = new_point_data
            self.logger.debug("Initialized self.point_data with new_point_data.")
            return

        # Existing point_ids
        existing_ids = set(self.point_data.central_point_ids)

        # Identify unique new points
        unique_mask = ~np.isin(new_point_data.central_point_ids, list(existing_ids))
        unique_indices = np.where(unique_mask)[0]

        if unique_indices.size == 0:
            self.logger.info("No new unique points to add.")
            return

        # Append unique new points
        self.point_data.coordinates = np.vstack((self.point_data.coordinates, new_point_data.coordinates[unique_indices]))
        self.point_data.dist_from_atom_center = np.vstack((self.point_data.dist_from_atom_center, new_point_data.dist_from_atom_center[unique_indices]))
        self.point_data.step_in_frac = np.vstack((self.point_data.step_in_frac, new_point_data.step_in_frac[unique_indices]))
        self.point_data.central_point_ids = np.concatenate((self.point_data.central_point_ids, new_point_data.central_point_ids[unique_indices]))
        self.point_data.chunk_ids = np.concatenate((self.point_data.chunk_ids, new_point_data.chunk_ids[unique_indices]))
        self.point_data.grid_amplitude_initialized = np.concatenate((self.point_data.grid_amplitude_initialized, new_point_data.grid_amplitude_initialized[unique_indices]))
        self.logger.debug(f"Added {unique_indices.size} new unique points to self.point_data.")

    def get_point_data(self) -> PointData:
        return self.point_data

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        """
        Append / update the PointData table in *hdf5_file_path*.

        • Existing datasets are extended (maxshape=None on first creation)
        • If an existing dataset is not chunked/extendible,
          it is transparently migrated to a chunked, unlimited dataset.
        • The boolean bitmap grid_amplitude_initialized is stored as 0/1.
        """
        pd = self.point_data
        data_dict = {
            "coordinates"              : pd.coordinates,
            "dist_from_atom_center"    : pd.dist_from_atom_center,
            "step_in_frac"             : pd.step_in_frac,
            "central_point_ids"        : pd.central_point_ids,
            "chunk_ids"                : pd.chunk_ids,
            "grid_amplitude_initialized":
                  pd.grid_amplitude_initialized.astype(np.int8)   # store 0 / 1
        }

        def _ensure_appendable(h5, key: str, arr: np.ndarray):
            """
            Return a (dataset, n_old) tuple where dataset is chunked and
            unlimited along axis 0. If an old fixed dataset exists, migrate it.
            """
            if key not in h5:
                maxshape = (None,) + arr.shape[1:]
                ds = h5.create_dataset(
                    key, data=arr[:0],  # start empty
                    maxshape=maxshape, chunks=True, compression="gzip"
                )
                return ds, 0

            ds = h5[key]
            n_old = int(ds.shape[0])
            appendable = (ds.chunks is not None) and (ds.maxshape is not None) and (ds.maxshape[0] is None)

            if appendable:
                return ds, n_old

            # Migrate a fixed-size dataset to an appendable layout
            old = ds[()]  # read all
            del h5[key]
            maxshape_old = (None,) + old.shape[1:]
            ds_new = h5.create_dataset(
                key, data=old,
                maxshape=maxshape_old, chunks=True, compression="gzip"
            )
            return ds_new, n_old

        # open for append – creates the file if absent
        with h5py.File(hdf5_file_path, "a") as h5:
            for key, arr in data_dict.items():
                ds, n_old = _ensure_appendable(h5, key, arr)
                # if our in-memory array is shorter than the file, don't shrink; only append tail if longer
                if arr.shape[0] > n_old:
                    ds.resize(arr.shape[0], axis=0)
                    ds[n_old:] = arr[n_old:]

        self.logger.info("Point data appended/updated in %s", hdf5_file_path)

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        try:
            with h5py.File(hdf5_file_path, 'r') as h5file:
                coordinates = h5file['coordinates'][:]
                dist_from_atom_center = h5file['dist_from_atom_center'][:]
                step_in_frac_array = h5file['step_in_frac'][:]
                central_point_ids = h5file['central_point_ids'][:]
                chunk_ids = h5file['chunk_ids'][:]
                grid_amplitude_initialized = h5file['grid_amplitude_initialized'][:].astype(bool)
                self.point_data = PointData(
                    coordinates=coordinates,
                    dist_from_atom_center=dist_from_atom_center,
                    step_in_frac=step_in_frac_array,
                    central_point_ids=central_point_ids,
                    chunk_ids=chunk_ids,
                    grid_amplitude_initialized=grid_amplitude_initialized
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load point data from HDF5 file: {e}")
            return False
