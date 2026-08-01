from __future__ import annotations

import numpy as np
import pytest

from core.scattering.commit import (
    ScatteringInvariantError,
    discover_scattering_attempts,
    load_scattering_attempt_partial,
    write_scattering_attempt,
)
from core.scattering.contracts import ScatteringWorkUnit, validate_scattering_work_unit


IDENTITY = {
    "run_digest": "run123",
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def test_scattering_attempt_writes_only_run_scoped_attempt_paths(tmp_path):
    manifest = write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=2,
        chunk_id=7,
        attempt_id="worker1-try1",
        grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        contribution_reciprocal_points=4,
        **IDENTITY,
    )

    assert manifest.payload_path.startswith(
        ".mosaic/runs/run123/scattering/chunks/chunk_7/attempts/"
    )
    assert "processed_point_data" not in manifest.payload_path
    assert not (tmp_path / "point_data_chunk_7_amplitudes.hdf5").exists()

    discovered = discover_scattering_attempts(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=7,
    )
    assert discovered == (manifest,)

    partial = load_scattering_attempt_partial(manifest, output_dir=tmp_path)
    assert partial.chunk_id == 7
    assert partial.contributing_interval_ids == (2,)
    assert partial.reciprocal_point_count == 4
    np.testing.assert_allclose(partial.amplitudes_delta, np.array([1.0 + 2.0j, 3.0 + 0.0j]))


def test_attempt_coverage_gate_rejects_amplitudes_inconsistent_with_grid_geometry(tmp_path):
    # coverage validation: coverage is gated against the DECLARED grid geometry
    # (sum_i prod(grid_shape_nd[i]) = 2*2 = 4 samples), NOT against len(amplitudes)
    # (= 2). A result whose amplitude length disagrees with its declared partition
    # geometry fails closed at write time instead of self-validating against
    # arange(len) -- closing the self-validation loop flagged in review.
    with pytest.raises(ScatteringInvariantError, match="point_coverage"):
        write_scattering_attempt(
            output_dir=tmp_path,
            interval_id=2,
            chunk_id=7,
            attempt_id="worker1-try1",
            grid_shape_nd=np.array([[2, 2]], dtype=np.int64),  # declares 4 samples
            amplitudes_delta=np.array([1.0 + 2.0j, 3.0 + 0.0j]),  # delivers only 2
            amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
            contribution_reciprocal_points=4,
            **IDENTITY,
        )


def test_attempt_coverage_gate_accepts_multipoint_grid_geometry(tmp_path):
    # The authoritative count spans multiple source points: grid_shape_nd rows
    # [[2], [1]] => prod-sum = 2 + 1 = 3 == len(amplitudes); the geometry-consistent
    # attempt passes (proving the gate counts per-point sub-grids, not a single prod).
    manifest = write_scattering_attempt(
        output_dir=tmp_path,
        interval_id=2,
        chunk_id=7,
        attempt_id="worker1-try1",
        grid_shape_nd=np.array([[2], [1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j, 0.1 + 0.0j]),
        contribution_reciprocal_points=4,
        **IDENTITY,
    )
    assert manifest.payload_path.startswith(
        ".mosaic/runs/run123/scattering/chunks/chunk_7/attempts/"
    )


def test_scattering_work_unit_carries_complete_p3_identity(tmp_path):
    work_unit = ScatteringWorkUnit.interval_chunk(
        interval_id=2,
        chunk_id=7,
        dimension=3,
        output_dir=str(tmp_path),
        **IDENTITY,
    )

    validate_scattering_work_unit(work_unit)

    assert work_unit.stage == "scattering"
    assert work_unit.run_digest == "run123"
    assert work_unit.scientific_digest == IDENTITY["scientific_digest"]
    assert work_unit.execution_digest == IDENTITY["execution_digest"]
    assert work_unit.qspace_plan_digest == IDENTITY["qspace_plan_digest"]
