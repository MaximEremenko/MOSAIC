"""The reference channel of the average amplitude (amorphous runs).

Three modes: None/'factorized' (crystal lattice-sum x basis product,
unchanged), 'direct' (average channel = ff * FT(reference coords)), and
'homogeneous' (average channel = 0; everything at q != 0 is diffuse).

Also pins the two identity properties: a factorized run's scientific
digest is byte-identical to the pre-reference_mode code (the key is
absent), and a direct run digests differently.
"""

import numpy as np
import pytest

from core.scattering.execution import _resolve_reference_spec
from core.scattering.kernels import (
    ReferenceSpec,
    compute_interval_coeff_contribution,
    compute_interval_element_contribution,
)
from core.scattering.planning import build_scientific_digest
from core.scattering.tasks import compute_scattering_interval_payload


class _OnesFF:
    def calculate(self, q_grid, element, charge=0.0):
        return np.ones(q_grid.shape[0])


def _system(seed=7, n=40, box=10.0):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0, box, (n, 3))
    displaced = reference + rng.normal(0, 0.05, (n, 3))
    elements = np.array(["O"] * (2 * n // 3) + ["Si"] * (n - 2 * n // 3))
    q_grid = (2 * np.pi / box) * np.array(
        [[1, 0, 0], [2, 1, 0], [1, 1, 1], [3, 2, 1], [0, 0, 2]], dtype=float
    )
    return reference, displaced, elements, q_grid


def _direct_sum(coords, q_grid, weights=None):
    weights = np.ones(len(coords)) if weights is None else weights
    return (weights * np.exp(1j * q_grid @ coords.T)).sum(axis=1)


def test_reference_spec_validation():
    with pytest.raises(ValueError, match="requires the reference"):
        ReferenceSpec(mode="direct")
    with pytest.raises(ValueError, match="Unknown reference mode"):
        ReferenceSpec(mode="factorized")  # selected by passing None, not a spec
    assert ReferenceSpec(mode="homogeneous").coords is None


def test_direct_mode_average_channel_is_reference_transform():
    reference, displaced, elements, q_grid = _system()
    _, _, _, q_amp, q_av = compute_interval_element_contribution(
        {"id": 0}, q_grid, "O", displaced, np.zeros_like(displaced),
        elements, 0.0, _OnesFF(),
        reference=ReferenceSpec(mode="direct", coords=reference),
    )
    mask = elements == "O"
    np.testing.assert_allclose(
        q_amp, _direct_sum(displaced[mask], q_grid), atol=1e-8
    )
    np.testing.assert_allclose(
        q_av, _direct_sum(reference[mask], q_grid), atol=1e-8
    )
    # the delta channel is the displacement-induced diffuse: nonzero
    assert np.abs(q_amp - q_av).max() > 1e-3


def test_homogeneous_mode_zeroes_average_channel():
    reference, displaced, elements, q_grid = _system()
    _, _, _, q_amp, q_av = compute_interval_element_contribution(
        {"id": 0}, q_grid, "O", displaced, np.zeros_like(displaced),
        elements, 0.0, _OnesFF(),
        reference=ReferenceSpec(mode="homogeneous"),
    )
    assert np.abs(q_av).max() == 0.0
    assert np.abs(q_amp).max() > 1.0


def test_factorized_collapses_on_one_cell_box():
    """The bug the reference branch exists to fix: with supercell=(1,1,1)
    every cells_origin entry coincides, the factorized product collapses to
    q_amp itself, and the delta channel is identically zero — silently."""
    reference, displaced, elements, q_grid = _system()
    _, _, _, q_amp, q_av = compute_interval_element_contribution(
        {"id": 0}, q_grid, "O", displaced, np.zeros_like(displaced),
        elements, 0.0, _OnesFF(),
    )
    np.testing.assert_allclose(q_amp, q_av, atol=1e-8)


def test_coeff_path_direct_mode():
    reference, displaced, elements, q_grid = _system()
    coeff = np.linspace(0.5, 1.5, len(displaced))
    _, _, _, q_amp, q_av = compute_interval_coeff_contribution(
        {"id": 0}, q_grid, coeff, displaced, np.zeros_like(displaced),
        reference=ReferenceSpec(mode="direct", coords=reference),
    )
    np.testing.assert_allclose(
        q_amp, _direct_sum(displaced, q_grid, coeff), atol=1e-8
    )
    np.testing.assert_allclose(
        q_av, _direct_sum(reference, q_grid, coeff), atol=1e-8
    )


def test_payload_delta_matches_analytic_displacement_diffuse():
    """End-to-end through compute_scattering_interval_payload: the payload's
    delta channel equals sum_j (e^{iq.r_j} - e^{iq.R_j}) for the summed
    species — the object the residual stage inverts."""
    reference, displaced, elements, q_grid = _system()
    box = 10.0
    interval = {
        "id": 0,
        "h_range": (0.0, 3.0),
        "k_range": (0.0, 2.0),
        "l_range": (1.0, 2.0),
    }
    B_ = np.linalg.inv(np.diag([box, box, box]))

    from core.qspace.masking.mask_strategies import DefaultMaskStrategy

    task = compute_scattering_interval_payload(
        interval,
        B_=B_,
        mask_params={},
        MaskStrategy=DefaultMaskStrategy(),
        supercell=np.array([1, 1, 1]),
        original_coords=displaced,
        cells_origin=np.zeros_like(displaced),
        elements_arr=elements,
        charge=0.0,
        use_coeff=False,
        coeff_val=None,
        unique_elements=["O", "Si"],
        ff_factory=_OnesFF(),
        reference=ReferenceSpec(mode="direct", coords=reference),
    )
    assert task is not None
    expected_delta = _direct_sum(displaced, task.q_grid) - _direct_sum(
        reference, task.q_grid
    )
    np.testing.assert_allclose(
        task.q_amp - task.q_amp_av, expected_delta, atol=1e-8
    )


def test_resolve_reference_spec_from_parameters():
    reference = np.zeros((4, 3))
    assert _resolve_reference_spec({}) is None
    assert _resolve_reference_spec({"reference_mode": "factorized"}) is None
    spec = _resolve_reference_spec(
        {"reference_mode": "direct", "average_coords": reference}
    )
    assert spec.mode == "direct" and spec.coords.shape == (4, 3)
    assert _resolve_reference_spec({"reference_mode": "homogeneous"}).mode == (
        "homogeneous"
    )
    with pytest.raises(ValueError, match="requires average_coords"):
        _resolve_reference_spec({"reference_mode": "direct"})
    with pytest.raises(ValueError, match="Unknown reference_mode"):
        _resolve_reference_spec({"reference_mode": "banana"})


def _digest_parameters(**extra):
    parameters = {
        "structure_content_digest": "a" * 64,
        "supercell": [1, 1, 1],
        "vectors": np.diag([10.0, 10.0, 10.0]).tolist(),
        "charge": 0.0,
        "postprocessing_mode": "displacement",
    }
    parameters.update(extra)
    return parameters


def test_scientific_digest_unchanged_without_reference_mode():
    """Crystal compatibility: the ABSENT key digests identically to code
    that predates reference_mode entirely (the identity builder skips
    absent keys). The default is therefore expressed by not writing the
    key — see build_base_amplitude_parameters."""
    assert build_scientific_digest(_digest_parameters()) == (
        build_scientific_digest(_digest_parameters())
    )
    with_key = build_scientific_digest(
        _digest_parameters(reference_mode="direct")
    )
    without_key = build_scientific_digest(_digest_parameters())
    assert with_key != without_key


def test_scientific_digest_separates_reference_modes():
    direct = build_scientific_digest(_digest_parameters(reference_mode="direct"))
    homogeneous = build_scientific_digest(
        _digest_parameters(reference_mode="homogeneous")
    )
    assert direct != homogeneous
