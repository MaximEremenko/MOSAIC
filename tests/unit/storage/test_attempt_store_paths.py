from __future__ import annotations

import pytest

from core.storage.attempt_store import (
    attempt_manifest_path,
    attempt_payload_path,
    chunk_commit_path,
    commit_candidate_manifest_path,
    commit_candidate_payload_path,
    fs_capability_path,
    qspace_plan_path,
    relative_to_output,
    run_manifest_path,
    stage_commit_path,
    stage_plan_path,
)


def test_attempt_paths_are_run_scoped_and_deterministic(tmp_path):
    kwargs = dict(
        output_dir=tmp_path,
        run_digest="rundigest",
        stage="scattering",
        chunk_id=7,
        work_unit_digest="workdigest",
        attempt_id="attempt123",
    )

    payload = attempt_payload_path(**kwargs)
    manifest = attempt_manifest_path(**kwargs)

    assert relative_to_output(payload, output_dir=tmp_path) == (
        ".mosaic/runs/rundigest/scattering/chunks/chunk_7/"
        "attempts/wo/workdigest/attempt_attempt123/payload.hdf5"
    )
    assert manifest.name == "attempt.json"


def test_attempt_paths_reject_path_segments(tmp_path):
    with pytest.raises(ValueError):
        attempt_manifest_path(
            output_dir=tmp_path,
            run_digest="../bad",
            stage="scattering",
            chunk_id=7,
            work_unit_digest="workdigest",
            attempt_id="attempt123",
        )


def test_run_stage_and_commit_paths_are_deterministic(tmp_path):
    assert relative_to_output(
        run_manifest_path(tmp_path, "run123"),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/run_manifest.json"
    assert relative_to_output(
        qspace_plan_path(tmp_path, "run123"),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/qspace_plan.json"
    assert relative_to_output(
        fs_capability_path(tmp_path, "run123"),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/fs_capability.json"
    assert relative_to_output(
        stage_plan_path(tmp_path, "run123", "residual_field"),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/residual_field/stage_plan.json"
    assert relative_to_output(
        stage_commit_path(tmp_path, "run123", "residual_field"),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/residual_field/stage_commit.json"
    assert relative_to_output(
        chunk_commit_path(tmp_path, "run123", "residual_field", 5),
        output_dir=tmp_path,
    ) == ".mosaic/runs/run123/residual_field/chunks/chunk_5/chunk_commit.json"
    assert relative_to_output(
        commit_candidate_payload_path(tmp_path, "run123", "residual_field", 5, "commit7"),
        output_dir=tmp_path,
    ) == (
        ".mosaic/runs/run123/residual_field/chunks/chunk_5/"
        "commit_attempts/commit_commit7/chunk_payload.hdf5"
    )
    assert relative_to_output(
        commit_candidate_manifest_path(tmp_path, "run123", "residual_field", 5, "commit7"),
        output_dir=tmp_path,
    ) == (
        ".mosaic/runs/run123/residual_field/chunks/chunk_5/"
        "commit_attempts/commit_commit7/commit_candidate.json"
    )
