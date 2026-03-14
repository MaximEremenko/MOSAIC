from types import SimpleNamespace

import numpy as np

from core.scattering.accumulation import (
    apply_scattering_partial_result,
    build_scattering_partial_result,
)
from core.scattering.calculator import compute_amplitudes_delta
from core.scattering.planning import build_scattering_execution_plan


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
    )

    assert len(plan.interval_work_units) == 2
    assert len(plan.chunk_work_units) == 3
    assert plan.chunk_ids == (3, 4)
    assert plan.interval_work_units[0].retry.idempotency_key == "scattering:interval:1"
    assert plan.chunk_work_units[0].retry.idempotency_key == "scattering:interval-chunk:1:3"
    assert plan.total_reciprocal_points > 0


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
    assert mirrored_count == 10


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
