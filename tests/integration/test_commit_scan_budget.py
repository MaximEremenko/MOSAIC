from __future__ import annotations

import numpy as np

from core.scattering.commit import (
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit_by_scan,
    write_scattering_attempt,
)
from core.storage.manifest import read_manifest
from core.storage.performance import (
    PerformanceMetricsManifest,
    evaluate_performance_budget,
    performance_metrics_path,
)


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "s" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "q" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _write_attempt(tmp_path, *, attempt_id: str) -> None:
    write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=1,
        chunk_id=3,
        attempt_id=attempt_id,
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        **IDENTITY,
    )


def test_commit_scan_records_performance_metrics_and_budget_failures(tmp_path):
    _write_attempt(tmp_path, attempt_id="try1")
    _write_attempt(tmp_path, attempt_id="try2")
    create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
        expected_interval_ids=(1,),
    )

    promote_scattering_chunk_commit_by_scan(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=3,
    )

    metrics = read_manifest(
        performance_metrics_path(tmp_path, "run123"),
        codec=PerformanceMetricsManifest,
        output_dir=tmp_path,
    )
    assert metrics.max_attempt_leaf_entries == 2
    assert metrics.run_file_count > 0
    assert metrics.commit_scan_seconds["scattering/chunk_3"] >= 0.0
    assert evaluate_performance_budget(
        metrics,
        {
            "max_attempt_leaf_entries": 2,
            "max_commit_scan_seconds_per_chunk": 60.0,
            "max_run_file_count": 100,
        },
    ) == []
    violations = evaluate_performance_budget(
        metrics,
        {
            "max_attempt_leaf_entries": 1,
            "max_run_file_count": 0,
        },
    )
    assert any("max_attempt_leaf_entries exceeded" in item for item in violations)
    assert any("max_run_file_count exceeded" in item for item in violations)
