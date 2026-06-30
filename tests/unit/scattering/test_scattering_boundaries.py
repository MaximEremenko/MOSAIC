from types import SimpleNamespace

import numpy as np
import pytest

from core.scattering.accumulation import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
    apply_half_space_conjugate_reconstruction,
    apply_scattering_partial_result,
    build_scattering_partial_result,
    half_space_conjugate_reconstruction_required,
)
from core.scattering.calculator import compute_amplitudes_delta
from core.scattering.half_space import classify_interval_half_space_role
from core.scattering.kernels import aggregate_interval_contributions
from core.scattering.planning import (
    ScatteringWorkIdentity,
    build_scattering_execution_plan,
)


IDENTITY = ScatteringWorkIdentity(
    run_digest="run123",
    scientific_digest="1" * 64,
    execution_digest="2" * 64,
    qspace_plan_digest="3" * 64,
    backend_policy_digest="4" * 64,
    source_structure_digest="5" * 64,
)


def test_build_scattering_execution_plan_uses_contract_work_units(tmp_path):
    class FakeDbManager:
        def get_unsaved_interval_chunks(self):
            return [(1, 3), (2, 3), (2, 4)]

    parameters = {
        "supercell": np.array([4.0]),
        "vectors": np.array([[1.0]]),
        "elements": np.array(["Na", "Cl"], dtype=object),
        "reciprocal_space_intervals": [
            {"id": 1, "h_range": (0.0, 1.0)},
            {"id": 2, "h_range": (1.0, 2.0)},
        ],
        "reciprocal_space_intervals_all": [
            {"h_range": (0.0, 1.0)},
            {"h_range": (1.0, 2.0)},
        ],
    }

    plan = build_scattering_execution_plan(
        parameters=parameters,
        db_manager=FakeDbManager(),
        output_dir=str(tmp_path),
        work_identity=IDENTITY,
    )

    assert len(plan.interval_work_units) == 2
    assert len(plan.chunk_work_units) == 3
    assert plan.chunk_ids == (3, 4)
    assert plan.interval_work_units[0].retry.idempotency_key == "scattering:interval:1"
    assert plan.chunk_work_units[0].retry.idempotency_key == "scattering:interval-chunk:1:3"
    assert all(unit.run_digest == "run123" for unit in plan.interval_work_units)
    assert all(unit.qspace_plan_digest == "3" * 64 for unit in plan.chunk_work_units)
    assert plan.total_reciprocal_points > 0


def test_interval_contribution_aggregation_uses_stable_element_order():
    q_grid = np.array([[0.0]], dtype=np.float64)
    contributions = [
        (1, "Zr", q_grid, np.array([1.0e16 + 0.0j]), np.array([1.0 + 0.0j])),
        (1, "Al", q_grid, np.array([-1.0e16 + 0.0j]), np.array([2.0 + 0.0j])),
        (1, "Na", q_grid, np.array([1.0 + 0.0j]), np.array([3.0 + 0.0j])),
    ]

    baseline = aggregate_interval_contributions(contributions, use_coeff=False)
    permuted = aggregate_interval_contributions(list(reversed(contributions)), use_coeff=False)

    np.testing.assert_array_equal(permuted.q_amp, baseline.q_amp)
    np.testing.assert_array_equal(permuted.q_amp_av, baseline.q_amp_av)


def test_scattering_accumulation_builds_and_applies_partial_results():
    current_rows = np.array([[101, 0.0 + 0.0j], [102, 0.0 + 0.0j]], dtype=np.complex128)
    current_average_rows = np.array(
        [[101, 0.0 + 0.0j], [102, 0.0 + 0.0j]],
        dtype=np.complex128,
    )

    partial = build_scattering_partial_result(
        chunk_id=3,
        interval_id=7,
        grid_shape_nd=np.array([[2, 2]]),
        amplitudes_delta=np.array([1.0 + 1.0j, 2.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        reciprocal_point_count=5,
        point_ids=np.array([101, 102]),
    )

    updated_rows, updated_average_rows, reciprocal_count = apply_scattering_partial_result(
        current_rows,
        current_average_rows,
        0,
        partial,
        mirror_conjugate_symmetry=False,
    )

    np.testing.assert_allclose(updated_rows[:, 1], np.array([1.0 + 1.0j, 2.0 + 0.0j]))
    np.testing.assert_allclose(updated_average_rows[:, 1], np.array([0.5 + 0.0j, 0.25 + 0.0j]))
    assert reciprocal_count == 5

    mirrored_rows, mirrored_average_rows, mirrored_count = apply_scattering_partial_result(
        current_rows,
        current_average_rows,
        0,
        partial,
        mirror_conjugate_symmetry=True,
    )

    np.testing.assert_allclose(
        mirrored_rows[:, 1],
        np.array([2.0 + 0.0j, 4.0 + 0.0j]),
    )
    np.testing.assert_allclose(
        mirrored_average_rows[:, 1],
        np.array([1.0 + 0.0j, 0.5 + 0.0j]),
    )
    # A5 double-count fix: the conjugate reconstruction doubles the AMPLITUDE but the
    # reciprocal_point_count (already multiplicity-applied) is added exactly ONCE, not
    # twice. Previously this asserted 10 (5 * 2); the corrected value is 5.
    assert mirrored_count == 5


def test_apply_scattering_partial_result_does_not_double_count_reciprocal_points():
    """A5 regression: mirror_conjugate_symmetry must NOT multiply reciprocal_point_count.

    The partial's reciprocal_point_count already encodes accepted x multiplicity (the
    q-normalization contract). Doubling the amplitude via delta + conj(delta) is the
    half-space reconstruction; the count is half-space-weighted already and must be
    accumulated once. This pins parity with merge_scattering_partial_results, which adds
    reciprocal_point_count by plain summation regardless of conjugate reconstruction.
    """
    current_rows = np.array([[1, 0.0 + 0.0j]], dtype=np.complex128)
    current_average_rows = np.array([[1, 0.0 + 0.0j]], dtype=np.complex128)
    partial = build_scattering_partial_result(
        chunk_id=0,
        interval_id=2,
        grid_shape_nd=np.array([[1]]),
        amplitudes_delta=np.array([1.0 + 1.0j]),
        amplitudes_average=np.array([2.0 + 0.0j]),
        reciprocal_point_count=7,
        point_ids=np.array([1]),
    )

    _, _, plain_count = apply_scattering_partial_result(
        current_rows,
        current_average_rows,
        3,
        partial,
        mirror_conjugate_symmetry=False,
    )
    _, _, mirrored_count = apply_scattering_partial_result(
        current_rows,
        current_average_rows,
        3,
        partial,
        mirror_conjugate_symmetry=True,
    )

    # Both paths add reciprocal_point_count EXACTLY ONCE (3 + 7 = 10), regardless of the
    # conjugate-symmetry reconstruction. No silent factor-of-two.
    assert plain_count == 10
    assert mirrored_count == 10


def test_half_space_conjugate_reconstruction_keeps_zero_plane_once():
    values = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128)
    zero_plane_q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    positive_l_q = np.array(
        [
            [0.0, 0.0, 0.25],
            [1.0, 0.0, 0.50],
        ],
        dtype=np.float64,
    )

    assert not half_space_conjugate_reconstruction_required(
        zero_plane_q,
        HALF_SPACE_ROLE_ZERO_PLANE,
    )
    np.testing.assert_allclose(
        apply_half_space_conjugate_reconstruction(
            values,
            zero_plane_q,
            HALF_SPACE_ROLE_ZERO_PLANE,
        ),
        values,
    )

    assert half_space_conjugate_reconstruction_required(
        positive_l_q,
        HALF_SPACE_ROLE_POSITIVE_HALF,
    )
    np.testing.assert_allclose(
        apply_half_space_conjugate_reconstruction(
            values,
            positive_l_q,
            HALF_SPACE_ROLE_POSITIVE_HALF,
        ),
        np.array([2.0 + 0.0j, 6.0 + 0.0j], dtype=np.complex128),
    )

    mixed_zero_positive_q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.50],
        ],
        dtype=np.float64,
    )
    mixed_positive_negative_q = np.array(
        [
            [0.0, 0.0, 0.25],
            [1.0, 0.0, -0.50],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="half_space_role metadata"):
        half_space_conjugate_reconstruction_required(mixed_zero_positive_q)
    with pytest.raises(ValueError, match="half_space_role metadata"):
        half_space_conjugate_reconstruction_required(mixed_positive_negative_q)


def test_half_space_reconstruction_uses_interval_role_before_cartesian_qz():
    values = np.array([1.0 + 2.0j], dtype=np.complex128)

    np.testing.assert_allclose(
        apply_half_space_conjugate_reconstruction(
            values,
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            HALF_SPACE_ROLE_POSITIVE_HALF,
        ),
        np.array([2.0 + 0.0j], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        apply_half_space_conjugate_reconstruction(
            values,
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            HALF_SPACE_ROLE_ZERO_PLANE,
        ),
        values,
    )


def test_interval_half_space_role_is_classified_from_hkl_l_indices():
    supercell = np.array([4.0, 4.0, 4.0])

    assert (
        classify_interval_half_space_role(
            {"h_range": (0.0, 1.0), "k_range": (0.0, 1.0), "l_range": (0.0, 0.0)},
            supercell,
        )
        == HALF_SPACE_ROLE_ZERO_PLANE
    )
    assert (
        classify_interval_half_space_role(
            {"h_range": (0.0, 1.0), "k_range": (0.0, 1.0), "l_range": (0.25, 1.0)},
            supercell,
        )
        == HALF_SPACE_ROLE_POSITIVE_HALF
    )
    with pytest.raises(ValueError, match="mixes L=0"):
        classify_interval_half_space_role(
            {"h_range": (0.0, 1.0), "k_range": (0.0, 1.0), "l_range": (0.0, 1.0)},
            supercell,
        )


def test_calculator_delegates_to_execution(monkeypatch):
    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "core.scattering.calculator.execute_scattering_stage",
        fake_execute,
    )

    compute_amplitudes_delta(
        parameters={"example": True},
        FormFactorFactoryProducer="ff",
        MaskStrategy="mask",
        MaskStrategyParameters={"radius": 1.0},
        db_manager=SimpleNamespace(),
        output_dir="/tmp/output",
        point_data_processor=SimpleNamespace(),
        client=None,
    )

    assert captured["parameters"] == {"example": True}
    assert captured["FormFactorFactoryProducer"] == "ff"
    assert captured["MaskStrategy"] == "mask"
    assert captured["MaskStrategyParameters"] == {"radius": 1.0}
    assert captured["output_dir"] == "/tmp/output"
