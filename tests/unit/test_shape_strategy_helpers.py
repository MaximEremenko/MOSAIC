import numpy as np

from core.domain.masking.shape_cpu import compute_1d_mask, compute_2d_mask
from core.domain.masking.shape_math import find_val_in_interval


def test_find_val_in_interval_returns_periodic_candidates():
    values = find_val_in_interval(-1.0, 1.0, 0.25)
    assert np.any(np.isclose(values, 0.25))


def test_compute_1d_mask_marks_points_inside_interval():
    data = np.array([[0.0], [0.2], [1.5]])
    mask = compute_1d_mask(
        data,
        radii=np.array([0.3]),
        centers=np.array([0.0]),
        coord_min=-1.0,
        coord_max=2.0,
    )
    assert mask.tolist() == [True, True, False]


def test_compute_2d_mask_marks_points_inside_circle():
    data = np.array([[0.0, 0.0], [0.4, 0.0], [0.9, 0.9]])
    mask = compute_2d_mask(
        data,
        radii=np.array([0.5]),
        centers=np.array([[0.0, 0.0]]),
        coord_min=np.array([-0.1, -0.1]),
        coord_max=np.array([0.8, 0.8]),
    )
    assert mask.tolist() == [True, True, False]
