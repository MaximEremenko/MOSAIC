from __future__ import annotations

import numpy as np

from core.scattering.commit import (
    discover_scattering_attempts,
    load_scattering_attempt_partial,
    write_scattering_attempt,
)
from core.scattering.contracts import ScatteringWorkUnit, validate_scattering_work_unit
from core.scattering.artifacts import persist_scattering_interval_chunk_shard


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


def test_identity_work_unit_shard_entrypoint_writes_attempt_not_old_shard(tmp_path):
    work_unit = ScatteringWorkUnit.interval_chunk(
        interval_id=2,
        chunk_id=7,
        dimension=3,
        output_dir=str(tmp_path),
        **IDENTITY,
    )

    persist_scattering_interval_chunk_shard(
        work_unit,
        grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
        total_reciprocal_points=5,
        contribution_reciprocal_points=4,
        amplitudes_delta=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        output_dir=str(tmp_path),
        quiet_logs=True,
    )
    persist_scattering_interval_chunk_shard(
        work_unit,
        grid_shape_nd=np.array([[2, 1]], dtype=np.int64),
        total_reciprocal_points=5,
        contribution_reciprocal_points=4,
        amplitudes_delta=np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        amplitudes_average=np.array([0.5 + 0.0j, 0.25 + 0.0j]),
        output_dir=str(tmp_path),
        quiet_logs=True,
    )

    attempts = discover_scattering_attempts(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=7,
    )
    assert len(attempts) == 2
    assert {attempt.interval_id for attempt in attempts} == {2}
    assert len({attempt.attempt_id for attempt in attempts}) == 2
    assert not (tmp_path / "scattering_shards" / "chunk_7" / "interval_2.hdf5").exists()
