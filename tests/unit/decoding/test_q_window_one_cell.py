"""The q-window PSF at supercell=(1,1,1) is the identity — by design.

The PSF grid is dimensioned in supercell reciprocal-sampling units
(real-space step cell/supercell, extent one cell); that geometry has no
counterpart on the amorphous one-cell box. Transplanting a box-hkl
window (2H+1 samples per axis) was tried and measured on the
imposed-field E2E: it attenuates the features ~300x and degrades the
decode R^2 from 0.99 to 0.83, because its grid step (L/(2H+1)) and
extent (L) are unrelated to the patch grid it is convolved with. The
degenerate 1-point PSF is the correct operator for the one-cell box.
"""

import numpy as np

from core.decoding.features import qspace_psf_in_r


def test_one_cell_psf_is_identity():
    psf = qspace_psf_in_r(
        size_aver=(1, 1, 1), hkl_max_xyz=(30.0, 30.0, 30.0), guard_frac=0.10
    )
    assert psf.shape == (1, 1, 1)
    np.testing.assert_allclose(psf, [[[1.0]]])


def test_crystal_psf_unchanged():
    psf = qspace_psf_in_r(
        size_aver=(8, 8, 8), hkl_max_xyz=(20.0, 20.0, 20.0), guard_frac=0.10
    )
    assert psf.shape == (8, 8, 8)
    assert abs(psf.sum() - 1.0) < 1e-12
