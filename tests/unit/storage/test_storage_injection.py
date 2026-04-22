from __future__ import annotations

import numpy as np

from core.scattering.artifacts import ScatteringArtifactStore
from core.residual_field.artifacts import ResidualFieldArtifactStore


class FakeSaver:
    def __init__(self, output_dir: str = "/tmp/fake", file_extension: str = "hdf5") -> None:
        self.output_dir = output_dir
        self.file_extension = file_extension
        self.data: dict[str, dict] = {}

    def generate_filename(self, chunk_id, suffix=""):
        if suffix:
            return f"point_data_chunk_{chunk_id}{suffix}.{self.file_extension}"
        return f"point_data_chunk_{chunk_id}.{self.file_extension}"

    def save_data(self, data, filename, append=False):
        self.data[filename] = data
        return filename

    def load_data(self, filename):
        if filename not in self.data:
            raise FileNotFoundError(filename)
        return self.data[filename]


def test_scattering_artifact_store_accepts_injected_saver():
    saver = FakeSaver()
    store = ScatteringArtifactStore("/unused", saver=saver)

    store.ensure_grid_shape(3, np.array([[2]]))
    store.ensure_total_reciprocal_points(3, 11)
    store.save_chunk_payloads(
        3,
        amplitudes_payload=np.array([[10 + 0j, 1 + 0j]], dtype=np.complex128),
        amplitudes_average_payload=np.array([[10 + 0j, 0.5 + 0j]], dtype=np.complex128),
        reciprocal_point_count=5,
    )

    current, current_av, nrec, shape_nd = store.load_chunk_payloads(3)

    np.testing.assert_allclose(shape_nd, np.array([[2]]))
    np.testing.assert_allclose(current[:, 1], np.array([1 + 0j]))
    np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j]))
    assert nrec == 5


def test_residual_field_artifact_store_reuses_injected_saver():
    saver = FakeSaver()
    store = ResidualFieldArtifactStore("/unused", saver=saver)

    store.ensure_grid_shape(7, np.array([[1]]))
    store.save_chunk_payloads(
        7,
        amplitudes_payload=np.array([[10 + 0j, 2 + 0j]], dtype=np.complex128),
        amplitudes_average_payload=np.array([[10 + 0j, 1 + 0j]], dtype=np.complex128),
        reciprocal_point_count=3,
    )
    current, current_av, nrec, _ = store.load_chunk_payloads(7)

    assert "residual_chunk_7_shapeNd.hdf5" in saver.data
    assert "residual_chunk_7_amplitudes.hdf5" in saver.data
    assert "residual_chunk_7_amplitudes_av.hdf5" in saver.data

    np.testing.assert_allclose(current[:, 1], np.array([2 + 0j]))
    np.testing.assert_allclose(current_av[:, 1], np.array([1 + 0j]))
    assert nrec == 3


def test_residual_field_artifact_store_reads_legacy_chunk_artifacts():
    saver = FakeSaver()
    saver.save_data({"shapeNd": np.array([[1]])}, "point_data_chunk_8_shapeNd.hdf5")
    saver.save_data(
        {"amplitudes": np.array([[10 + 0j, 2 + 0j]], dtype=np.complex128)},
        "point_data_chunk_8_amplitudes.hdf5",
    )
    saver.save_data(
        {"amplitudes_av": np.array([[10 + 0j, 1 + 0j]], dtype=np.complex128)},
        "point_data_chunk_8_amplitudes_av.hdf5",
    )
    saver.save_data(
        {"nreciprocal_space_points": np.array([3], dtype=np.int64)},
        "point_data_chunk_8_amplitudes_nreciprocal_space_points.hdf5",
    )
    saver.save_data(
        {"ids": np.array([4, 5], dtype=np.int64)},
        "point_data_chunk_8_applied_interval_ids.hdf5",
    )

    store = ResidualFieldArtifactStore("/unused", saver=saver)
    current, current_av, nrec, shape_nd = store.load_chunk_payloads(8)
    applied = store.load_applied_interval_ids(8)

    np.testing.assert_allclose(shape_nd, np.array([[1]]))
    np.testing.assert_allclose(current[:, 1], np.array([2 + 0j]))
    np.testing.assert_allclose(current_av[:, 1], np.array([1 + 0j]))
    assert nrec == 3
    assert applied == {4, 5}
