from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.tasks import run_residual_field_interval_chunk_task
from core.scattering.half_space import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
)
from core.scattering.kernels import IntervalTask


BASELINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "baselines"
    / "residual_full_synthetic_v1"
)


class _CapturingReducerBackend:
    def __init__(self):
        self.calls: list[dict] = []

    def uses_local_chunk_accumulator(self):
        return False

    def persist_shard_checkpoint(self, work_unit, **kwargs):
        self.calls.append(kwargs)
        return "manifest"


def _load_baseline():
    metadata = json.loads((BASELINE_DIR / "baseline.json").read_text(encoding="utf-8"))
    arrays = np.load(BASELINE_DIR / "baseline_arrays.npz")
    return metadata, arrays


def _assert_required_provenance(metadata):
    provenance = metadata["provenance"]
    assert provenance["generator_command"] == (
        "python3 tests/fixtures/baselines/residual_full_synthetic_v1/generate.py"
    )
    assert provenance["anchor_commit"] == "f019f76"
    assert provenance["input_files"] == []
    assert provenance["synthetic_only"] is True
    assert provenance["public_representative_fixture"]["status"] == "deferred"
    dependency_versions = provenance["dependency_versions"]
    assert dependency_versions["python"]
    assert dependency_versions["numpy"]


def _work_unit(tmp_path):
    return ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=3,
        parameter_digest="synthetic",
        output_dir=str(tmp_path),
    )


def _interval_task(arrays, *, half_space_role: str) -> IntervalTask:
    q_grid_name = (
        "q_grid_positive_l"
        if half_space_role == HALF_SPACE_ROLE_POSITIVE_HALF
        else "q_grid_zero_l"
    )
    return IntervalTask(
        1,
        "All",
        arrays[q_grid_name],
        arrays["q_amp_delta"],
        arrays["q_amp_average"],
        half_space_role=half_space_role,
        reciprocal_multiplicity=2
        if half_space_role == HALF_SPACE_ROLE_POSITIVE_HALF
        else 1,
    )


def _run_residual_task(monkeypatch, tmp_path, arrays, *, half_space_role: str):
    reducer = _CapturingReducerBackend()
    interval_task = _interval_task(arrays, half_space_role=half_space_role)
    monkeypatch.setattr(
        "core.residual_field.tasks.execute_inverse_cunufft_super_batch",
        lambda **kwargs: np.array(arrays["inverse_outputs"], copy=True),
    )

    result = run_residual_field_interval_chunk_task(
        _work_unit(tmp_path),
        interval_task,
        None,
        total_reciprocal_points=int(interval_task.reciprocal_multiplicity),
        output_dir=str(tmp_path),
        reducer_backend=reducer,
        quiet_logs=True,
        rifft_payload=(arrays["rifft_grid"], arrays["grid_shape_nd"]),
    )

    assert result == "manifest"
    assert len(reducer.calls) == 1
    return reducer.calls[0]


def test_residual_full_baseline_metadata_records_reproducibility_provenance():
    metadata, _arrays = _load_baseline()

    _assert_required_provenance(metadata)


def test_residual_full_positive_half_reconstructs_real_amplitudes(monkeypatch, tmp_path):
    metadata, arrays = _load_baseline()

    captured = _run_residual_task(
        monkeypatch,
        tmp_path,
        arrays,
        half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
    )

    np.testing.assert_allclose(
        captured["amplitudes_delta"],
        arrays["positive_half_delta_expected"],
    )
    np.testing.assert_allclose(
        captured["amplitudes_average"],
        arrays["positive_half_average_expected"],
    )
    delta_ratio = np.linalg.norm(np.imag(captured["amplitudes_delta"])) / np.linalg.norm(
        np.real(captured["amplitudes_delta"])
    )
    average_ratio = np.linalg.norm(np.imag(captured["amplitudes_average"])) / np.linalg.norm(
        np.real(captured["amplitudes_average"])
    )
    assert delta_ratio < metadata["thresholds"]["positive_half_delta_imag_over_real"]
    assert average_ratio < metadata["thresholds"]["positive_half_average_imag_over_real"]
    np.testing.assert_array_equal(captured["point_ids"], arrays["point_ids_expected"])


def test_residual_full_zero_plane_preserves_imaginary_signal(monkeypatch, tmp_path):
    metadata, arrays = _load_baseline()

    captured = _run_residual_task(
        monkeypatch,
        tmp_path,
        arrays,
        half_space_role=HALF_SPACE_ROLE_ZERO_PLANE,
    )

    np.testing.assert_allclose(
        captured["amplitudes_delta"],
        arrays["zero_plane_delta_expected"],
    )
    np.testing.assert_allclose(
        captured["amplitudes_average"],
        arrays["zero_plane_average_expected"],
    )
    assert metadata["thresholds"]["zero_plane_preserves_imag"] is True
    assert np.linalg.norm(np.imag(captured["amplitudes_delta"])) > 0
    assert np.linalg.norm(np.imag(captured["amplitudes_average"])) > 0
