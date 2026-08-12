import numpy as np
import h5py

from core.models import PointData
from core.patch_centers.point_data import PointDataProcessor
from core.storage.rifft_in_data_saver import RIFFTInDataSaver


def test_point_data_initializes_missing_chunk_state():
    point_data = PointData(
        coordinates=np.zeros((2, 3)),
        dist_from_atom_center=np.zeros((2, 3)),
        step_in_frac=np.zeros((2, 3)),
        central_point_ids=np.array([0, 1]),
        chunk_ids=np.array([], dtype=int),
        grid_amplitude_initialized=np.array([], dtype=bool),
    )
    assert point_data.chunk_ids.tolist() == [0, 0]
    assert point_data.grid_amplitude_initialized.tolist() == [False, False]


def test_point_data_processor_streams_chunk_amplitudes_without_vstack(
    monkeypatch,
    tmp_path,
):
    point_data = PointData(
        coordinates=np.array([[0.0], [1.0]], dtype=float),
        dist_from_atom_center=np.array([[0.2], [0.2]], dtype=float),
        step_in_frac=np.array([[0.1], [0.1]], dtype=float),
        central_point_ids=np.array([10, 11], dtype=int),
        chunk_ids=np.array([0, 0], dtype=int),
        grid_amplitude_initialized=np.array([False, False], dtype=bool),
    )
    processor = PointDataProcessor(
        RIFFTInDataSaver(str(tmp_path), "hdf5"),
        save_rifft_coordinates=False,
    )

    def _forbid_vstack(*args, **kwargs):
        raise AssertionError("chunk processing must not materialize via np.vstack")

    def _forbid_grid_arrays(*args, **kwargs):
        raise AssertionError("amplitude-only initialization must not build grid arrays")

    monkeypatch.setattr(np, "vstack", _forbid_vstack)
    monkeypatch.setattr(processor, "_generate_grid", _forbid_grid_arrays)

    processor.process_point_data(point_data)

    with h5py.File(tmp_path / "point_data_chunk_0_amplitudes.hdf5", "r") as h5:
        amplitudes = h5["amplitudes"][:]
    with h5py.File(tmp_path / "point_data_chunk_0_amplitudes_av.hdf5", "r") as h5:
        amplitudes_av = h5["amplitudes_av"][:]

    assert amplitudes.shape == (10, 2)
    assert amplitudes_av.shape == (10, 2)
    np.testing.assert_allclose(amplitudes[:, 1], np.zeros(10, dtype=np.complex128))
    np.testing.assert_allclose(amplitudes_av, amplitudes)
    np.testing.assert_allclose(amplitudes[:5, 0], np.full(5, 10, dtype=np.complex128))
    np.testing.assert_allclose(amplitudes[5:, 0], np.full(5, 11, dtype=np.complex128))
    assert point_data.grid_amplitude_initialized.tolist() == [True, True]
