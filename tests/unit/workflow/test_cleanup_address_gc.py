"""W2.3 regression: address-based GC of superseded commit candidates.

Before P11, ``core/workflow/cleanup.py`` (then ``core/storage/cleanup.py``)
gated candidate deletion on a BITWISE rule: if the valid candidates for a chunk
did not all share one ``payload_sha256`` it retained ALL of them as
"conflicting candidates". Under P11 address-based identity, a candidate is
addressed by ``candidate_id`` (chunk + device-independent work-unit addresses),
NOT by payload bytes, and per-attempt byte integrity is verified separately at
load. The durable chunk commit names the winning ``selected_candidate_id``, so
any OTHER valid candidate is superseded and safe to delete regardless of its
bytes.

These tests construct two VALID candidates for one chunk that AGREE numerically
but differ in ``payload_sha256`` (the candidate payload digest folds in the
candidate_id) -- exactly the pre-P11 "conflicting" trigger -- and assert the
superseded one is GC'd down to the selected candidate, NOT retained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.residual_field.commit import (
    RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
    ResidualCommitCandidateManifest,
    _payload_attrs as _residual_payload_attrs,
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    validate_residual_commit_candidate,
    write_residual_attempt,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.scattering.commit import (
    ScatteringCommitCandidateManifest,
    _candidate_payload_attrs as _scattering_candidate_payload_attrs,
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    validate_scattering_commit_candidate,
    write_scattering_attempt,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)
from core.storage.attempt_store import (
    commit_candidate_manifest_path,
    commit_candidate_payload_path,
    relative_to_output,
)
from core.storage.commit_payloads import _payload_datasets, _payload_digest, _read_payload
from core.storage.fingerprint import file_sha256
from core.storage.hdf5_atomic import atomic_hdf5_write
from core.storage.manifest import write_manifest
from core.workflow.cleanup import cleanup_run_artifacts


SCATTERING_IDENTITY = {
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _forge_sibling_candidate(
    *,
    output_dir: Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    selected,
    new_candidate_id: str,
    schema: str,
    attrs: dict,
    manifest_cls,
):
    """Materialize a SECOND valid commit candidate for the same chunk at a
    different ``candidate_id``, reusing the selected candidate's real selected
    attempts and arrays. The candidate payload digest folds in the candidate_id
    (``expected_set_digest=candidate_id``), so the sibling's ``payload_sha256``
    DIFFERS from the selected candidate's even though the arrays are identical --
    reproducing the pre-P11 "len(payload_hashes) != 1" trigger with genuinely
    valid, numerically-agreeing candidates.
    """
    datasets, _ = _read_payload(output_dir / selected.payload_path)
    rebuilt = _payload_datasets(
        point_ids=np.asarray(datasets["point_ids"], dtype=np.int64),
        grid_shape_nd=np.asarray(datasets["grid_shape_nd"], dtype=np.int64),
        amplitudes_delta=np.asarray(datasets["amplitudes_delta"], dtype=np.complex128),
        amplitudes_average=np.asarray(datasets["amplitudes_average"], dtype=np.complex128),
    )
    payload_path = commit_candidate_payload_path(
        output_dir, run_digest, stage, int(chunk_id), new_candidate_id
    )
    atomic_hdf5_write(payload_path, rebuilt, attrs=attrs)
    payload_sha256 = _payload_digest(
        schema=schema,
        expected_set_digest=new_candidate_id,
        datasets=rebuilt,
        attrs=attrs,
    )
    base = selected.to_payload()
    base["candidate_id"] = new_candidate_id
    base["payload_path"] = relative_to_output(payload_path, output_dir=output_dir)
    base["payload_sha256"] = payload_sha256
    base["file_sha256"] = file_sha256(payload_path)
    base["payload_nbytes"] = int(payload_path.stat().st_size)
    sibling = manifest_cls.from_payload(base)
    write_manifest(
        commit_candidate_manifest_path(
            output_dir, run_digest, stage, int(chunk_id), new_candidate_id
        ),
        sibling,
        output_dir=output_dir,
    )
    return sibling


def _assert_gc_down_to_selected(report, *, selected, sibling, output_dir):
    selected_dir = (output_dir / selected.payload_path).parent
    sibling_dir = (output_dir / sibling.payload_path).parent
    # Address-based GC: selected kept, superseded sibling deleted -- NOT retained.
    assert selected_dir.exists()
    assert not sibling_dir.exists()
    assert any(sibling.candidate_id in path for path in report.removed_paths)
    assert not any(
        "retained conflicting candidates" in reason for reason in report.skipped_reasons
    )
    assert not any(sibling.candidate_id == path for path in report.retained_paths)


def test_cleanup_gcs_superseded_scattering_candidate_despite_byte_difference(tmp_path):
    run_digest = "run-gc-scatter"
    write_scattering_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        interval_id=1,
        chunk_id=0,
        attempt_id="try1",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        point_ids=np.array([0], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )
    selected = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_interval_ids=(1,),
    )
    write_scattering_stage_plan(
        output_dir=tmp_path, run_digest=run_digest, expected_by_chunk={0: (1,)}
    )
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=selected)
    write_scattering_stage_commit(output_dir=tmp_path, run_digest=run_digest, chunk_ids=(0,))

    new_id = "b" * 32
    sibling = _forge_sibling_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        stage="scattering",
        chunk_id=0,
        selected=selected,
        new_candidate_id=new_id,
        schema=ScatteringCommitCandidateManifest.schema,
        attrs=_scattering_candidate_payload_attrs(
            run_digest=run_digest,
            candidate_id=new_id,
            chunk_id=0,
            reciprocal_point_count=int(selected.reciprocal_point_count),
        ),
        manifest_cls=ScatteringCommitCandidateManifest,
    )

    # Preconditions: both candidates are VALID, share the same arrays, and have
    # DIFFERENT payload_sha256 (the exact pre-P11 "conflicting" trigger).
    validate_scattering_commit_candidate(selected, output_dir=tmp_path)
    validate_scattering_commit_candidate(sibling, output_dir=tmp_path)
    assert sibling.candidate_id != selected.candidate_id
    assert sibling.payload_sha256 != selected.payload_sha256
    assert (tmp_path / selected.payload_path).parent.exists()
    assert (tmp_path / sibling.payload_path).parent.exists()

    report = cleanup_run_artifacts(tmp_path, run_digest)

    _assert_gc_down_to_selected(
        report, selected=selected, sibling=sibling, output_dir=tmp_path
    )


def _complete_residual_chunk(tmp_path, *, run_digest: str):
    write_scattering_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        interval_id=1,
        chunk_id=0,
        attempt_id="try1",
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
        point_ids=np.array([0], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )
    scattering_candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_interval_ids=(1,),
    )
    write_scattering_stage_plan(
        output_dir=tmp_path, run_digest=run_digest, expected_by_chunk={0: (1,)}
    )
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=scattering_candidate)
    scattering_stage = write_scattering_stage_commit(
        output_dir=tmp_path, run_digest=run_digest, chunk_ids=(0,)
    )

    write_residual_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id="worker1-try1",
        parameter_digest="d" * 64,
        partition_plan_digest="e" * 64,
        source_scattering_commit_digest=scattering_stage.stage_digest,
        source_replacement_digest=None,
        backend_policy_digest="f" * 64,
        expected_output_digest="0" * 64,
        point_ids=np.array([0], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([0.25 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([0.125 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=1,
    )
    residual_candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_partitions={0: (0,)},
        expected_reciprocal_point_count=1,
    )
    write_residual_stage_plan(
        output_dir=tmp_path, run_digest=run_digest, expected_by_chunk={0: (0,)}
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=residual_candidate)
    write_residual_stage_commit(output_dir=tmp_path, run_digest=run_digest, chunk_ids=(0,))
    return residual_candidate


def test_cleanup_gcs_superseded_residual_candidate_despite_byte_difference(tmp_path):
    run_digest = "run-gc-residual"
    selected = _complete_residual_chunk(tmp_path, run_digest=run_digest)

    new_id = "c" * 32
    datasets, _ = _read_payload(tmp_path / selected.payload_path)
    point_stop = int(np.asarray(datasets["point_ids"]).reshape(-1).shape[0])
    sibling = _forge_sibling_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        stage="residual_field",
        chunk_id=0,
        selected=selected,
        new_candidate_id=new_id,
        schema=RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
        attrs=_residual_payload_attrs(
            schema=RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
            run_digest=run_digest,
            work_unit_digest=new_id,
            attempt_id=None,
            chunk_id=0,
            partition_id=-1,
            point_start=0,
            point_stop=point_stop,
            reciprocal_point_count=int(selected.reciprocal_point_count),
        ),
        manifest_cls=ResidualCommitCandidateManifest,
    )

    validate_residual_commit_candidate(selected, output_dir=tmp_path)
    validate_residual_commit_candidate(sibling, output_dir=tmp_path)
    assert sibling.candidate_id != selected.candidate_id
    assert sibling.payload_sha256 != selected.payload_sha256
    assert (tmp_path / selected.payload_path).parent.exists()
    assert (tmp_path / sibling.payload_path).parent.exists()

    report = cleanup_run_artifacts(tmp_path, run_digest)

    _assert_gc_down_to_selected(
        report, selected=selected, sibling=sibling, output_dir=tmp_path
    )
