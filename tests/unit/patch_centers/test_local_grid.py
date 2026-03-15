import numpy as np

from core.patch_centers.local_grid import GridGenerator1D


def test_grid_generator_1d_returns_point_and_shape_for_single_point_case():
    generator = GridGenerator1D(step_in_frac=0.05)

    grid_points, grid_shape = generator.generate_grid_around_point(
        np.array([0.5]),
        0.01,
    )

    np.testing.assert_allclose(grid_points, np.array([[0.5]]))
    np.testing.assert_allclose(grid_shape, np.array([1]))
