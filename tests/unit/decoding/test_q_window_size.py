"""The q-window/PSF grid length for the one-cell (amorphous) box.

For a crystal the grid length per axis is the supercell repeat count. For
supercell=(1,1,1) that collapses the window to chebwin(1)=[1] and the PSF
to a single point — a silent no-op. The amorphous resolution uses the
number of integer box-hkl samples per axis, 2H+1.
"""

import numpy as np

from core.decoding.features import qspace_psf_in_r, resolve_q_window_size


def test_crystal_supercell_passes_through():
    np.testing.assert_array_equal(
        resolve_q_window_size([16, 16, 16], (40.0, 40.0, 40.0)), [16, 16, 16]
    )


def test_one_cell_box_uses_hkl_extent():
    np.testing.assert_array_equal(
        resolve_q_window_size([1, 1, 1], (84.0, 84.0, 84.0)), [169, 169, 169]
    )
    # non-integer extents round up
    np.testing.assert_array_equal(
        resolve_q_window_size([1, 1, 1], (5.5, 3.0, 2.2)), [13, 7, 7]
    )


def test_one_cell_psf_is_not_a_single_point():
    """supercell=(1,1,1) verbatim yields a 1-point identity PSF; the
    resolved size yields a real band-limit kernel."""
    degenerate = qspace_psf_in_r(
        size_aver=(1, 1, 1), hkl_max_xyz=(8.0, 8.0, 8.0), guard_frac=0.10
    )
    assert degenerate.shape == (1, 1, 1)

    resolved = qspace_psf_in_r(
        size_aver=tuple(resolve_q_window_size([1, 1, 1], (8.0, 8.0, 8.0))),
        hkl_max_xyz=(8.0, 8.0, 8.0),
        guard_frac=0.10,
    )
    assert resolved.shape == (17, 17, 17)
    # a genuine kernel: normalized, centered, with off-center mass
    assert abs(resolved.sum() - 1.0) < 1e-12
    center = tuple(n // 2 for n in resolved.shape)
    assert resolved[center] == resolved.max()
    off_center = resolved.copy()
    off_center[center] = 0.0
    assert np.abs(off_center).max() > 0.0
