import logging
import os

import h5py
import numpy as np

from core.domain.lattice import angstrom_to_fractional
from core.domain.models import PointData

logger = logging.getLogger(__name__)


class FullListPointProcessor:
    def __init__(self, parameters: dict, average_structure: dict, num_chunks: int = 10):
        self.parameters = parameters
        self.average_structure = average_structure or {}
        self.num_chunks = max(1, int(num_chunks))
        self.point_data: PointData | None = None
        self.vectors = self.average_structure.get("vectors")
        self.hdf5_file_path = self.parameters.get("hdf5_file_path", "point_data.hdf5")

    def process_parameters(self):
        if os.path.exists(self.hdf5_file_path):
            if self.load_point_data_from_hdf5(self.hdf5_file_path):
                logger.info("Loaded point data from %s", self.hdf5_file_path)
                return

        logger.info("Processing parameters for 'full_list' method.")
        rspace_info = self.parameters.get("rspace_info", {})
        points_params = rspace_info.get("points", [])

        all_coordinates = []
        all_dist_from_atom_center = []
        all_step_in_frac = []

        for point_info in points_params:
            file_name = point_info.get("filename") or point_info.get("file name")
            if not file_name:
                logger.error("Missing points file in point parameters.")
                continue

            if not os.path.exists(file_name):
                logger.error("Points file not found: %s", file_name)
                continue

            points = self._read_points_file(file_name)
            if points.size == 0:
                logger.warning("No points loaded from %s", file_name)
                continue

            if self.vectors is None:
                fractional_coords = points
            else:
                fractional_coords = angstrom_to_fractional(points, self.vectors)

            dimension = fractional_coords.shape[1] if fractional_coords.ndim > 1 else 1
            zero_array = np.zeros((fractional_coords.shape[0], dimension))

            all_coordinates.append(fractional_coords)
            all_dist_from_atom_center.append(zero_array)
            all_step_in_frac.append(zero_array)

        if not all_coordinates:
            logger.error("No point data generated.")
            return

        coordinates_array = np.vstack(all_coordinates)
        dist_from_atom_center_array = np.vstack(all_dist_from_atom_center)
        step_in_frac_array = np.vstack(all_step_in_frac)
        central_point_ids = np.arange(coordinates_array.shape[0], dtype=int)
        chunk_ids = self._build_chunk_ids(coordinates_array.shape[0])
        initialized = np.zeros(coordinates_array.shape[0], dtype=bool)

        self.point_data = PointData(
            coordinates=coordinates_array,
            dist_from_atom_center=dist_from_atom_center_array,
            step_in_frac=step_in_frac_array,
            central_point_ids=central_point_ids,
            chunk_ids=chunk_ids,
            grid_amplitude_initialized=initialized,
        )
        self.save_point_data_to_hdf5(self.hdf5_file_path)

    def _read_points_file(self, file_name: str) -> np.ndarray:
        try:
            points = np.loadtxt(file_name)
            if points.ndim == 1:
                points = np.expand_dims(points, axis=0)
            logger.info("Loaded %d points from %s.", points.shape[0], file_name)
            return points
        except Exception as exc:
            logger.error("Failed to read points from file %s: %s", file_name, exc)
            return np.array([])

    def _build_chunk_ids(self, num_points: int) -> np.ndarray:
        if num_points == 0:
            return np.empty((0,), dtype=int)
        points_per_chunk = max(1, int(np.ceil(num_points / self.num_chunks)))
        chunk_ids = np.arange(num_points) // points_per_chunk
        return chunk_ids.astype(int)

    def get_point_data(self) -> PointData:
        if self.point_data is None:
            raise ValueError("Point data has not been generated yet.")
        return self.point_data

    def save_point_data_to_hdf5(self, hdf5_file_path: str):
        try:
            if self.point_data is None:
                logger.error("No point data to save.")
                return
            with h5py.File(hdf5_file_path, "w") as h5file:
                h5file.create_dataset("coordinates", data=self.point_data.coordinates)
                h5file.create_dataset(
                    "dist_from_atom_center", data=self.point_data.dist_from_atom_center
                )
                h5file.create_dataset("step_in_frac", data=self.point_data.step_in_frac)
                h5file.create_dataset(
                    "central_point_ids", data=self.point_data.central_point_ids
                )
                h5file.create_dataset("chunk_ids", data=self.point_data.chunk_ids)
                h5file.create_dataset(
                    "grid_amplitude_initialized",
                    data=self.point_data.grid_amplitude_initialized.astype(int),
                )
            logger.info("Point data saved to HDF5 file: %s", hdf5_file_path)
        except Exception as exc:
            logger.error("Failed to save point data to HDF5 file: %s", exc)

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        try:
            with h5py.File(hdf5_file_path, "r") as h5file:
                self.point_data = PointData(
                    coordinates=h5file["coordinates"][:],
                    dist_from_atom_center=h5file["dist_from_atom_center"][:],
                    step_in_frac=h5file["step_in_frac"][:],
                    central_point_ids=h5file["central_point_ids"][:],
                    chunk_ids=h5file["chunk_ids"][:],
                    grid_amplitude_initialized=h5file[
                        "grid_amplitude_initialized"
                    ][:].astype(bool),
                )
            return True
        except Exception as exc:
            logger.error("Failed to load point data from HDF5 file: %s", exc)
            return False
