from __future__ import annotations

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.half_space import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
)
from core.scattering.kernels import IntervalTask


class _CapturingReducerBackend:
    def __init__(self):
        self.calls: list[dict] = []

    def uses_local_chunk_accumulator(self):
        return False

    def persist_shard_checkpoint(self, work_unit, **kwargs):
        self.calls.append(kwargs)
        return "manifest"


def _work_unit(tmp_path):
    return ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="synthetic",
        output_dir=str(tmp_path),
    )


def test_residual_task_reconstructs_positive_half_space_from_metadata(monkeypatch, tmp_path):
    reducer = _CapturingReducerBackend()
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[5.0 + 2.0j], [6.0 - 3.0j]], dtype=np.complex128),
    )

    result = run_residual_field_interval_chunk_task(
        _work_unit(tmp_path),
        IntervalTask(
            1,
            "All",
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
            half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
            reciprocal_multiplicity=2,
        ),
        None,
        total_reciprocal_points=2,
        output_dir=str(tmp_path),
        reducer_backend=reducer,
        quiet_logs=True,
        rifft_payload=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[1]], dtype=np.int64),
        ),
    )

    assert result == "manifest"
    captured = reducer.calls[0]
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([10.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([12.0 + 0.0j]))


def test_residual_task_preserves_zero_plane_imaginary_signal(monkeypatch, tmp_path):
    reducer = _CapturingReducerBackend()
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array([[5.0 + 2.0j], [6.0 - 3.0j]], dtype=np.complex128),
    )

    result = run_residual_field_interval_chunk_task(
        _work_unit(tmp_path),
        IntervalTask(
            1,
            "All",
            np.array([[0.0, 0.0, 0.25]], dtype=np.float64),
            np.array([2.0 + 0.0j]),
            np.array([1.0 + 0.0j]),
            half_space_role=HALF_SPACE_ROLE_ZERO_PLANE,
            reciprocal_multiplicity=1,
        ),
        None,
        total_reciprocal_points=1,
        output_dir=str(tmp_path),
        reducer_backend=reducer,
        quiet_logs=True,
        rifft_payload=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[1]], dtype=np.int64),
        ),
    )

    assert result == "manifest"
    captured = reducer.calls[0]
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([5.0 + 2.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([6.0 - 3.0j]))
