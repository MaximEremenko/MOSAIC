import numpy as np

from core.data_structures.point_data import PointData


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

