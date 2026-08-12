from __future__ import annotations

import numpy as np

from core.residual_field.commit import load_residual_attempt_payload
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.kernels import IntervalTask


IDENTITY = {
    "run_digest": "run123",
    "parameter_digest": "d" * 64,
    "partition_plan_digest": "e" * 64,
    "source_scattering_commit_digest": "a" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "0" * 64,
}


class UnusedReducer:
    pass


def _fake_inverse_cunufft(*, q_coords, weights, real_coords, eps):
    del eps
    q_arr = np.asarray(q_coords, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.complex128)
    real_arr = np.asarray(real_coords, dtype=np.float64)
    phase = np.exp(1j * (q_arr @ real_arr.T))
    return weight_arr @ phase


def _run_case(monkeypatch, output_dir, *, presum_enabled: bool):
    monkeypatch.setenv(
        "MOSAIC_RESIDUAL_SAME_Q_GRID_PRESUM",
        "1" if presum_enabled else "0",
    )
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        _fake_inverse_cunufft,
    )
    q_grid = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.float64,
    )
    interval_tasks = (
        IntervalTask(
            1,
            "All",
            q_grid,
            np.array([1.0 + 0.5j, 2.0 - 0.25j, -0.5 + 0.75j]),
            np.array([0.25 + 0.0j, -1.0 + 0.5j, 0.0 - 0.25j]),
        ),
        IntervalTask(
            2,
            "All",
            q_grid.copy(),
            np.array([-0.75 + 0.25j, 0.5 + 0.0j, 1.25 - 0.5j]),
            np.array([0.5 - 0.5j, 0.25 + 0.25j, -0.75 + 0.0j]),
        ),
    )
    work_unit = ResidualFieldWorkUnit.interval_chunk_batch(
        interval_ids=(1, 2),
        chunk_id=5,
        output_dir=str(output_dir),
        **IDENTITY,
    ).with_partition(
        partition_id=0,
        point_start=10,
        point_stop=12,
    )
    return run_residual_field_interval_chunk_task(
        work_unit,
        interval_tasks,
        atoms=None,
        total_reciprocal_points=3,
        output_dir=str(output_dir),
        reducer_backend=UnusedReducer(),
        quiet_logs=True,
        rifft_payload=(
            np.array([[0.0, 0.0], [0.5, 0.25]], dtype=np.float64),
            np.array([[2]], dtype=np.int64),
        ),
    )


def test_same_q_grid_presum_matches_unoptimized_path(monkeypatch, tmp_path):
    optimized_manifest = _run_case(
        monkeypatch,
        tmp_path / "optimized",
        presum_enabled=True,
    )
    unoptimized_manifest = _run_case(
        monkeypatch,
        tmp_path / "unoptimized",
        presum_enabled=False,
    )

    optimized_payload, _optimized_attrs = load_residual_attempt_payload(
        optimized_manifest,
        output_dir=tmp_path / "optimized",
    )
    unoptimized_payload, _unoptimized_attrs = load_residual_attempt_payload(
        unoptimized_manifest,
        output_dir=tmp_path / "unoptimized",
    )

    np.testing.assert_array_equal(
        optimized_payload["point_ids"],
        unoptimized_payload["point_ids"],
    )
    np.testing.assert_array_equal(
        optimized_payload["grid_shape_nd"],
        unoptimized_payload["grid_shape_nd"],
    )
    np.testing.assert_allclose(
        optimized_payload["amplitudes_delta"],
        unoptimized_payload["amplitudes_delta"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_payload["amplitudes_average"],
        unoptimized_payload["amplitudes_average"],
        rtol=1e-12,
        atol=1e-12,
    )
