import numpy as np
import logging
from contextlib import ExitStack
from core.storage.rifft_in_data_saver import RIFFTInDataSaver
from core.models import PointData
from core.runtime.log_utils import short_path
from core.patch_centers.local_grid import (
    GridGenerator1D,
    GridGenerator2D,
    GridGenerator3D,
)
from typing import Optional
import h5py
import os

# Fork-COW context for parallel chunk-file materialization: children
# inherit the processor object (including any test monkeypatches on it)
# without pickling; each child writes only its own chunk's files.
_POINT_INIT_PROCESSOR = None


def _materialize_point_chunk(task):
    chunk_id, mask = task
    _POINT_INIT_PROCESSOR._process_chunk(chunk_id, mask)
    return chunk_id


class PointDataProcessor:
    """
    Processes point data by expanding points and saving grid points and amplitudes by chunks.
    """

    def __init__(self, data_saver: RIFFTInDataSaver, save_rifft_coordinates: bool = False, max_chunk_size: int = 100000):
        """
        Initializes the PointDataProcessor with a DataSaver instance and a global flag.

        Args:
            data_saver (RIFFTInDataSaver): Instance responsible for saving data to files.
            save_rifft_coordinates (bool): Flag to determine if grid points should be saved separately.
            max_chunk_size (int): Maximum number of points per chunk.
        """
        self.data_saver = data_saver
        self.save_rifft_coordinates = save_rifft_coordinates
        self.max_chunk_size = max_chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.point_data: Optional[PointData] = None
        
    def _sync_initialisation_from_disk(self) -> None:
            """
            Make sure self.point_data.grid_amplitude_initialized is consistent
            with the bitmap stored in the HDF-5 file.
    
            • If the dataset is missing  →  create it (all-zero) on disk **and**
              in memory.
            • If it exists but the size mismatches the current PointData length
              (rare, e.g. point cloud changed) → recreate it from scratch.
            """
            path = self.data_saver.hdf5_file_path
            n_points = len(self.point_data.coordinates)
    
            # --- ensure the HDF-5 file exists -----------------------------------
            if not os.path.isfile(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with h5py.File(path, "w"):
                    pass                      # empty file
    
            with h5py.File(path, "a") as h5:
                if "grid_amplitude_initialized" in h5:
                    saved = h5["grid_amplitude_initialized"][...]
                    if saved.size != n_points:            # size mismatch
                        self.logger.warning(
                            "Existing bitmap size %d ≠ current point count %d → "
                            "re-initialising.", saved.size, n_points)
                        del h5["grid_amplitude_initialized"]
                        saved = np.zeros(n_points, dtype=np.int8)
                        h5.create_dataset("grid_amplitude_initialized",
                                          data=saved, chunks=True, compression="gzip")
                else:
                    # first ever run – create all-false dataset
                    saved = np.zeros(n_points, dtype=np.int8)
                    h5.create_dataset("grid_amplitude_initialized",
                                      data=saved, chunks=True, compression="gzip")
    
            # mirror to the in-memory structure (as **bool**)
            self.point_data.grid_amplitude_initialized = saved.astype(bool)
            self.logger.info("grid_amplitude_initialized synced (%d points).",
                             n_points)
    def process_point_data(self, point_data: PointData):
        """
        Processes the point data by generating grid points and amplitudes for chunks that haven't been processed.

        Args:
            point_data (PointData): The point data to process.
        """
        self.point_data = point_data
        
        self._sync_initialisation_from_disk()
        
        # Identify unique chunk_ids
        unique_chunk_ids = np.unique(self.point_data.chunk_ids)
        self.logger.info(f"Found {len(unique_chunk_ids)} unique chunk_ids.")

        pending: list[tuple[int, np.ndarray]] = []
        for chunk_id in unique_chunk_ids:
            # Find points in this chunk that haven't been initialized
            mask = (self.point_data.chunk_ids == chunk_id) & (~self.point_data.grid_amplitude_initialized)
            num_uninitialized = np.sum(mask)

            if num_uninitialized == 0:
                self.logger.debug(f"Chunk {chunk_id} already processed. Skipping.")
                continue

            self.logger.info(f"Processing chunk {chunk_id} with {num_uninitialized} uninitialized points.")
            pending.append((chunk_id, mask))

        # Each chunk's pre-allocation writes ~13.5 GB (hkl40) through one
        # driver thread under h5py's per-process lock — 54 GB / 51 s
        # single-threaded. Chunks write disjoint files, so a fork pool
        # sidesteps the lock entirely; children inherit this processor
        # copy-on-write and the parent owns the bitmap.
        workers = self._point_data_init_workers(len(pending))
        if workers <= 1 or len(pending) <= 1:
            for chunk_id, mask in pending:
                self._process_chunk(chunk_id, mask)
        else:
            import multiprocessing

            global _POINT_INIT_PROCESSOR
            try:
                fork_ctx = multiprocessing.get_context("fork")
            except ValueError:
                fork_ctx = None
            if fork_ctx is None:
                for chunk_id, mask in pending:
                    self._process_chunk(chunk_id, mask)
            else:
                # ProcessPoolExecutor (not mp.Pool): a SIGKILLed child makes
                # future.result() raise BrokenProcessPool instead of hanging
                # the parent forever on an unposted imap result.
                from concurrent.futures import (
                    ProcessPoolExecutor,
                    as_completed,
                )

                mask_by_chunk = {int(chunk_id): mask for chunk_id, mask in pending}
                _POINT_INIT_PROCESSOR = self
                try:
                    with ProcessPoolExecutor(
                        max_workers=workers, mp_context=fork_ctx
                    ) as pool:
                        futures = [
                            pool.submit(_materialize_point_chunk, task)
                            for task in pending
                        ]
                        for future in as_completed(futures):
                            done_chunk = future.result()
                            self.point_data.grid_amplitude_initialized[
                                mask_by_chunk[int(done_chunk)]
                            ] = True
                finally:
                    _POINT_INIT_PROCESSOR = None

        # After processing all chunks, save the updated grid_amplitude_initialized
        self.save_grid_amplitude_initialized()

    def _point_data_init_workers(self, pending_count: int) -> int:
        raw = os.getenv("MOSAIC_POINT_DATA_INIT_PROCS")
        if raw is not None and str(raw).strip() != "":
            try:
                return max(1, min(int(raw), pending_count))
            except ValueError:
                pass
        from core.runtime.cpu_resources import available_cpu_count

        # I/O-bound full-chunk writes: more than a few concurrent writers
        # just fragments the device queue.
        return max(1, min(pending_count, available_cpu_count(), 8))

    def _process_chunk(self, chunk_id: int, mask: np.ndarray):
        """
        Processes a single chunk of points.

        Args:
            chunk_id (int): The ID of the chunk.
            mask (np.ndarray): Boolean array indicating which points in the chunk are uninitialized.
        """
        indices = np.where(mask)[0]
        coordinates = self.point_data.coordinates[indices]
        dist_from_atom_center = self.point_data.dist_from_atom_center[indices]
        step_in_frac = self.point_data.step_in_frac[indices]
        central_point_ids = self.point_data.central_point_ids[indices]

        num_points = len(indices)
        dimensionality = coordinates.shape[1]
        self.logger.debug(f"Chunk {chunk_id}: Processing {num_points} uninitialized points with dimensionality {dimensionality}.")

        amplitude_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes')
        amplitude_av_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_av')
        grid_filename = None
        if self.save_rifft_coordinates:
            grid_filename = self.data_saver.generate_filename(chunk_id, suffix='_grid')
        point_counts = np.array(
            [
                self._grid_point_count(dimensionality, step, dist)
                for step, dist in zip(step_in_frac, dist_from_atom_center)
            ],
            dtype=np.int64,
        )
        total_rows = int(point_counts.sum())

        with ExitStack() as stack:
            amplitude_h5 = stack.enter_context(
                h5py.File(self._stream_file_path(amplitude_filename), "w")
            )
            amplitude_av_h5 = stack.enter_context(
                h5py.File(self._stream_file_path(amplitude_av_filename), "w")
            )
            amplitude_dataset = self._create_stream_dataset(
                amplitude_h5,
                "amplitudes",
                total_rows=total_rows,
                shape_tail=(2,),
                dtype=np.complex128,
            )
            amplitude_av_dataset = self._create_stream_dataset(
                amplitude_av_h5,
                "amplitudes_av",
                total_rows=total_rows,
                shape_tail=(2,),
                dtype=np.complex128,
            )
            grid_dataset = None
            if grid_filename is not None:
                grid_h5 = stack.enter_context(
                    h5py.File(self._stream_file_path(grid_filename), "w")
                )
                grid_dataset = self._create_stream_dataset(
                    grid_h5,
                    "grid_points",
                    total_rows=total_rows,
                    shape_tail=(dimensionality,),
                    dtype=np.float64,
                )

            self._write_amplitude_runs(
                amplitude_dataset,
                amplitude_av_dataset,
                central_point_ids=central_point_ids,
                point_counts=point_counts,
            )

            if grid_dataset is not None:
                write_start = 0
                for i in range(num_points):
                    central_point = coordinates[i]
                    dist = dist_from_atom_center[i]
                    step = step_in_frac[i]
                    central_point_id = central_point_ids[i]

                    grid_points, grid_shapeNd = self._generate_grid(chunk_id, dimensionality, step, central_point, dist, central_point_id)
                    write_stop = write_start + int(grid_points.shape[0])
                    grid_dataset[write_start:write_stop] = grid_points
                    write_start = write_stop
        
        #total_reciprocal_points_filename =  self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_ntotal_reciprocal_space_points')
        #self.data_saver.save_data({'ntotal_reciprocal_points': np.zeros([1], dtype = np.int64)}, total_reciprocal_points_filename)    
        total_reciprocal_points_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_ntotal_reciprocal_space_points')
        if not os.path.exists(self._stream_file_path(total_reciprocal_points_filename)):
            self.data_saver.save_data(                {
                    'ntotal_reciprocal_space_points': np.array([-1], dtype=np.int64),
                    'ntotal_reciprocal_points': np.array([-1], dtype=np.int64),
                },
                total_reciprocal_points_filename
            )
        
        nreciprocal_space_points_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_nreciprocal_space_points')
        self.data_saver.save_data({'nreciprocal_space_points': np.zeros([1], dtype = int)}, nreciprocal_space_points_filename)
        self.logger.info(f"Chunk {chunk_id}: Amplitudes saved to {amplitude_filename}")

        # Mark all points in this chunk as initialized
        self.point_data.grid_amplitude_initialized[mask] = True
        self.logger.debug(f"Chunk {chunk_id}: All uninitialized points marked as initialized.")

    def _stream_file_path(self, filename: str) -> str:
        if not os.path.exists(self.data_saver.output_dir):
            os.makedirs(self.data_saver.output_dir)
        return os.path.join(self.data_saver.output_dir, filename)

    def _create_stream_dataset(
        self,
        h5file: h5py.File,
        dataset_name: str,
        *,
        total_rows: int,
        shape_tail: tuple[int, ...],
        dtype,
    ) -> h5py.Dataset:
        return h5file.create_dataset(
            dataset_name,
            shape=(int(total_rows),) + tuple(shape_tail),
            maxshape=(None,) + tuple(shape_tail),
            dtype=dtype,
            chunks=self._stream_chunk_shape(total_rows, shape_tail, dtype),
            fillvalue=np.array(0, dtype=dtype).item(),
        )

    def _stream_chunk_shape(
        self,
        total_rows: int,
        shape_tail: tuple[int, ...],
        dtype,
    ) -> tuple[int, ...]:
        row_bytes = int(np.dtype(dtype).itemsize * np.prod(shape_tail or (1,)))
        target_chunk_bytes = 4 * 1024 * 1024
        chunk_rows = max(1, target_chunk_bytes // max(1, row_bytes))
        if total_rows > 0:
            chunk_rows = min(int(total_rows), chunk_rows)
        return (int(chunk_rows),) + tuple(shape_tail)

    def _write_amplitude_runs(
        self,
        amplitude_dataset: h5py.Dataset,
        amplitude_av_dataset: h5py.Dataset,
        *,
        central_point_ids: np.ndarray,
        point_counts: np.ndarray,
    ) -> None:
        if point_counts.size == 0:
            return
        row_bytes = np.dtype(np.complex128).itemsize
        buffer_rows = max(1, (64 * 1024 * 1024) // row_bytes)
        buffer = np.empty((buffer_rows,), dtype=np.complex128)
        write_start = 0
        fill = 0

        def flush() -> None:
            nonlocal write_start, fill
            if fill == 0:
                return
            write_stop = write_start + fill
            block = buffer[:fill]
            amplitude_dataset[write_start:write_stop, 0] = block
            amplitude_av_dataset[write_start:write_stop, 0] = block
            write_start = write_stop
            fill = 0

        for central_point_id, count_value in zip(central_point_ids, point_counts):
            remaining = int(count_value)
            while remaining > 0:
                capacity = buffer_rows - fill
                take = min(remaining, capacity)
                buffer[fill:fill + take] = central_point_id
                fill += take
                remaining -= take
                if fill == buffer_rows:
                    flush()
        flush()

    def _grid_point_count(self, dimensionality, step_in_frac, dist) -> int:
        if dimensionality == 1:
            return int(self._axis_grid_count(step_in_frac, dist))
        step_sizes = np.array(
            [step_in_frac] * dimensionality
            if np.isscalar(step_in_frac)
            else step_in_frac,
            dtype=float,
        ).reshape(-1)
        distances = np.array(dist, dtype=float).reshape(-1)
        if step_sizes.size != dimensionality or distances.size != dimensionality:
            raise ValueError(
                f"Expected {dimensionality}D step/dist values, got "
                f"{step_sizes.size} step values and {distances.size} distance values."
            )
        count = 1
        for step, distance in zip(step_sizes, distances):
            count *= self._axis_grid_count(step, distance)
        return int(count)

    def _axis_grid_count(self, step, dist) -> int:
        step = float(np.asarray(step).reshape(-1)[0])
        dist = float(np.asarray(dist).reshape(-1)[0])
        if step <= 0 or dist <= step:
            return 1
        epsilon = 1e-12
        return int(np.arange(-dist, dist + step - epsilon, step).size)

    def generate_grid(
        self,
        *,
        chunk_id: int,
        dimensionality,
        step_in_frac,
        central_point,
        dist,
        central_point_id,
    ):
        return self._generate_grid(
            chunk_id=chunk_id,
            dimensionality=dimensionality,
            step_in_frac=step_in_frac,
            central_point=central_point,
            dist=dist,
            central_point_id=central_point_id,
        )

    def _generate_grid(self, chunk_id: int, dimensionality, step_in_frac, central_point, dist, central_point_id):
        """
        Generates grid points around a central point.

        Args:
            chunk_id (int): The ID of the chunk.
            dimensionality (int): Dimensionality of the data (1, 2, or 3).
            step_in_frac (float or array-like): Step sizes for each dimension.
            central_point (np.ndarray): Coordinates of the central point.
            dist (np.ndarray): Distances from the central point.
            central_point_id (int or str): Original ID of the central point.

        Returns:
            np.ndarray: Array of grid points generated around the central point.
        """
        self.logger.debug(f"Chunk {chunk_id}: Generating grid for central_point_id={central_point_id} with step_in_frac={step_in_frac} and dist={dist}")

        grid_generator = self.grid_generator_factory(dimensionality, step_in_frac)
        grid_points, grid_shapeNd  = grid_generator.generate_grid_around_point(np.array(central_point), np.array(dist))
        self.logger.debug(f"Chunk {chunk_id}: Generated {grid_points.shape[0]} grid points for central_point_id={central_point_id}")

        return grid_points, grid_shapeNd

    def _generate_amplitude(self, chunk_id: int, central_point_id, grid_points):
        """
        Generates amplitude data for a set of grid points.

        Args:
            chunk_id (int): The ID of the chunk.
            central_point_id (int or str): Original ID of the central point.
            grid_points (np.ndarray): Array of grid points for which amplitudes are to be generated.

        Returns:
            np.ndarray: Array containing central_point_id and corresponding amplitude values.
        """
        self.logger.debug(f"Chunk {chunk_id}: Generating amplitude data for central_point_id={central_point_id}")

        amplitude_data = np.empty((grid_points.shape[0], 2), dtype=np.complex128)
        amplitude_data[:, 0] = central_point_id
        amplitude_data[:, 1] = 0.0
        self.logger.debug(f"Chunk {chunk_id}: Generated amplitude data with shape {amplitude_data.shape}")

        return amplitude_data

    def _save_chunk_data(self, chunk_id: int, grid_points: Optional[np.ndarray], amplitude_data: np.ndarray, amplitude_av_data: np.ndarray, nreciprocal_space_points: [np.ndarray]):
        """
        Saves the grid points and amplitude data for a chunk.

        Args:
            chunk_id (int): The ID of the chunk.
            grid_points (np.ndarray): Generated grid points, or None.
            amplitude_data (np.ndarray): Generated amplitude data.
        """
        if self.save_rifft_coordinates and grid_points is not None:
            grid_filename = self.data_saver.generate_filename(chunk_id, suffix='_grid')
            self.data_saver.save_data({'grid_points': grid_points}, grid_filename)
            self.logger.info(f"Chunk {chunk_id}: Grid points saved to {grid_filename}")

        amplitude_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes')
        self.data_saver.save_data({'amplitudes': amplitude_data}, amplitude_filename)
        
        amplitude_av_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_av')
        self.data_saver.save_data({'amplitudes_av': amplitude_av_data}, amplitude_av_filename)
        
        nreciprocal_space_points_filename = self.data_saver.generate_filename(chunk_id, suffix='_amplitudes_nreciprocal_space_points')
        self.data_saver.save_data({'nreciprocal_space_points': nreciprocal_space_points}, nreciprocal_space_points_filename)
        self.logger.info(f"Chunk {chunk_id}: Amplitudes saved to {amplitude_filename}")

    def save_chunk_data(
        self,
        *,
        chunk_id: int,
        grid_points: Optional[np.ndarray],
        amplitude_data: np.ndarray,
        amplitude_av_data: np.ndarray,
        nreciprocal_space_points,
    ) -> None:
        self._save_chunk_data(
            chunk_id,
            grid_points,
            amplitude_data,
            amplitude_av_data,
            nreciprocal_space_points,
        )

    def grid_generator_factory(self, dimensionality, step_in_frac):
        """
        Factory method to get the appropriate GridGenerator based on dimensionality.

        Args:
            dimensionality (int): Dimensionality of the data (1, 2, or 3).
            step_in_frac (float or array-like): Step sizes for each dimension.

        Returns:
            GridGenerator*: Instance of the appropriate GridGenerator class.
        """
        if dimensionality == 1:
            return GridGenerator1D(step_in_frac)
        elif dimensionality == 2:
            return GridGenerator2D(step_in_frac)
        elif dimensionality == 3:
            return GridGenerator3D(step_in_frac)
        else:
            self.logger.error(f"Unsupported dimensionality: {dimensionality}")
            raise ValueError(f"Unsupported dimensionality: {dimensionality}")

    def save_grid_amplitude_initialized(self):
        """
        Saves the updated `grid_amplitude_initialized` array to the HDF5 file.
        """
        hdf5_file_path = self.data_saver.hdf5_file_path
        try:
            with h5py.File(hdf5_file_path, 'a') as h5file:
                if 'grid_amplitude_initialized' in h5file:
                    del h5file['grid_amplitude_initialized']
                h5file.create_dataset('grid_amplitude_initialized', data=self.point_data.grid_amplitude_initialized.astype(int))
            self.logger.info("Updated grid_amplitude_initialized saved to %s", short_path(hdf5_file_path))
        except Exception as e:
            self.logger.error(f"Failed to save updated grid_amplitude_initialized: {e}")
