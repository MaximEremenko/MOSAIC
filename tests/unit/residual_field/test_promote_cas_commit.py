from __future__ import annotations

import numpy as np
import pytest

from core.residual_field.commit import (
    RESIDUAL_FIELD_STAGE,
    ResidualChunkCommitManifest,
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    promote_residual_chunk_commit_by_scan,
    write_residual_attempt,
)
from core.storage.attempt_store import chunk_commit_path
from core.storage.manifest import write_manifest


IDENTITY = {
    "run_digest": "run123",
    "parameter_digest": "d" * 64,
    "partition_plan_digest": "e" * 64,
    "source_scattering_commit_digest": "a" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "0" * 64,
}


def _attempt(tmp_path, *, partition_id, point_id, reciprocal_count=9):
    return write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=partition_id,
        point_start=partition_id,
        point_stop=partition_id + 1,
        interval_ids=(1,),
        attempt_id=f"try-{partition_id}",
        point_ids=np.array([point_id], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([partition_id + 1.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=reciprocal_count,
        **IDENTITY,
    )


def _candidate(tmp_path):
    _attempt(tmp_path, partition_id=0, point_id=10)
    _attempt(tmp_path, partition_id=1, point_id=11)
    return create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
        expected_partitions={0: (10,), 1: (11,)},
        expected_reciprocal_point_count=9,
    )


def test_promote_is_idempotent_end_to_end(tmp_path):
    candidate = _candidate(tmp_path)
    target = chunk_commit_path(tmp_path, "run123", RESIDUAL_FIELD_STAGE, 5)

    first = promote_residual_chunk_commit(output_dir=tmp_path, candidate=candidate)
    bytes_after_first = target.read_bytes()

    # Re-promote: same committed chunk, bytes unchanged, no error.
    second = promote_residual_chunk_commit(output_dir=tmp_path, candidate=candidate)

    assert first == second
    assert target.read_bytes() == bytes_after_first


def test_concurrent_double_by_scan_skips_second_no_error(tmp_path):
    """Two by-scan promotes (identical manifest) => one creates, one skips."""
    _candidate(tmp_path)
    target = chunk_commit_path(tmp_path, "run123", RESIDUAL_FIELD_STAGE, 5)

    first = promote_residual_chunk_commit_by_scan(
        output_dir=tmp_path, run_digest="run123", chunk_id=5
    )
    bytes_after_first = target.read_bytes()

    second = promote_residual_chunk_commit_by_scan(
        output_dir=tmp_path, run_digest="run123", chunk_id=5
    )

    assert first == second
    assert target.read_bytes() == bytes_after_first


def test_different_committed_candidate_raises_conflict(tmp_path):
    candidate = _candidate(tmp_path)
    target = chunk_commit_path(tmp_path, "run123", RESIDUAL_FIELD_STAGE, 5)

    # Pre-commit a chunk_commit that points at a DIFFERENT candidate id.
    forged = ResidualChunkCommitManifest(
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
        selected_candidate_id="forged-different-candidate",
        candidate_manifest_path=candidate.payload_path,
        candidate_payload_path=candidate.payload_path,
        payload_sha256=candidate.payload_sha256,
        file_sha256=candidate.file_sha256,
        payload_nbytes=int(candidate.payload_nbytes),
    )
    write_manifest(target, forged, output_dir=tmp_path)
    forged_bytes = target.read_bytes()

    with pytest.raises(RuntimeError, match="already committed to a different candidate"):
        promote_residual_chunk_commit_by_scan(
            output_dir=tmp_path, run_digest="run123", chunk_id=5
        )

    # The forged commit's bytes survive (no overwrite).
    assert target.read_bytes() == forged_bytes
