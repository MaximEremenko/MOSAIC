from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from core.decoding.commit import write_decoder_commit
from core.residual_field.commit import (
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    write_residual_attempt,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.scattering.commit import (
    create_scattering_commit_candidate,
    promote_scattering_chunk_commit,
    write_scattering_attempt,
    write_scattering_stage_commit,
    write_scattering_stage_plan,
)
from core.storage.fingerprint import file_sha256
from core.storage.manifest import ManifestError
from core.storage.publisher import (
    PublicPublishError,
    publish_run,
    validate_public_manifest_files,
)


SCATTERING_IDENTITY = {
    "scientific_digest": "a" * 64,
    "execution_digest": "e" * 64,
    "qspace_plan_digest": "c" * 64,
    "backend_policy_digest": "b" * 64,
    "source_structure_digest": "a" * 64,
}


def _complete_private_run(tmp_path, *, run_digest: str, decoder: bool = False):
    write_scattering_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        interval_id=1,
        chunk_id=0,
        attempt_id="try1",
        grid_shape_nd=np.array([[2]], dtype=np.int64),
        amplitudes_delta=np.array([2.0 + 1.0j, 3.0 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([1.0 + 0.0j, 1.5 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=2,
        point_ids=np.array([0, 1], dtype=np.int64),
        **SCATTERING_IDENTITY,
    )
    scattering_candidate = create_scattering_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_interval_ids=(1,),
    )
    write_scattering_stage_plan(
        output_dir=tmp_path,
        run_digest=run_digest,
        expected_by_chunk={0: (1,)},
    )
    promote_scattering_chunk_commit(output_dir=tmp_path, candidate=scattering_candidate)
    scattering_stage = write_scattering_stage_commit(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_ids=(0,),
    )

    write_residual_attempt(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        partition_id=0,
        point_start=0,
        point_stop=2,
        interval_ids=(1,),
        attempt_id="worker1-try1",
        parameter_digest="d" * 64,
        partition_plan_digest="e" * 64,
        source_scattering_commit_digest=scattering_stage.stage_digest,
        source_replacement_digest=None,
        backend_policy_digest="f" * 64,
        expected_output_digest="0" * 64,
        point_ids=np.array([0, 1], dtype=np.int64),
        grid_shape_nd=np.array([[2]], dtype=np.int64),
        amplitudes_delta=np.array([0.25 + 0.5j, 0.5 + 0.0j], dtype=np.complex128),
        amplitudes_average=np.array([0.125 + 0.0j, 0.25 + 0.0j], dtype=np.complex128),
        contribution_reciprocal_points=2,
    )
    residual_candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_id=0,
        expected_partitions={0: (0, 1)},
        expected_reciprocal_point_count=2,
    )
    write_residual_stage_plan(
        output_dir=tmp_path,
        run_digest=run_digest,
        expected_by_chunk={0: (0,)},
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=residual_candidate)
    write_residual_stage_commit(
        output_dir=tmp_path,
        run_digest=run_digest,
        chunk_ids=(0,),
    )

    if decoder:
        cache_path = tmp_path / "decoder_M_test.npz"
        np.savez(cache_path, decoder_matrix=np.eye(2), coordinate_count=np.array([2]))
        write_decoder_commit(
            output_dir=tmp_path,
            run_digest=run_digest,
            decoder_cache_path=cache_path,
            decoder_cache_identity={
                "schema": "mosaic.decoder.cache_identity",
                "schema_version": 1,
                "decoder_cache_digest": "d" * 64,
            },
        )


def test_public_manifest_authority_ignores_loose_files_and_fails_closed(tmp_path):
    (tmp_path / "residual_chunk_0_amplitudes.hdf5").write_text(
        "loose public artifact",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        validate_public_manifest_files(tmp_path / "public_manifest.json", output_dir=tmp_path)

    (tmp_path / "public_manifest.json").write_text(
        json.dumps(
            {
                "schema": "mosaic.public_manifest",
                "schema_version": 1,
                "run_digest": "run123",
                "source_stage": "residual_field",
                "source_identity": {},
                "published_files": [],
                "published_file_records": [],
                "public_manifest_digest": "0" * 64,
                "previous_run_digest": None,
                "previous_public_manifest_sha256": None,
                "previous_public_manifest_path": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="digest"):
        validate_public_manifest_files(tmp_path / "public_manifest.json", output_dir=tmp_path)


def test_public_publisher_single_writer_refuses_conflicting_run(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    _complete_private_run(tmp_path, run_digest="run456")

    publish_run(tmp_path, "run123")

    with pytest.raises(PublicPublishError, match="different run"):
        publish_run(tmp_path, "run456")


def test_public_publish_crash_resume(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    with h5py.File(tmp_path / "point_data_chunk_0_amplitudes.hdf5", "w") as handle:
        handle.create_dataset("amplitudes", data=np.array([99.0]))

    with pytest.raises(FileNotFoundError):
        validate_public_manifest_files(tmp_path / "public_manifest.json", output_dir=tmp_path)

    manifest = publish_run(tmp_path, "run123")
    loaded = validate_public_manifest_files(tmp_path / "public_manifest.json", output_dir=tmp_path)

    assert loaded == manifest
    with h5py.File(tmp_path / "point_data_chunk_0_amplitudes.hdf5", "r") as handle:
        np.testing.assert_allclose(handle["amplitudes"][:], np.array([2.0 + 1.0j, 3.0 + 0.0j]))


def test_public_publish_replace_history(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123")
    _complete_private_run(tmp_path, run_digest="run456")
    first = publish_run(tmp_path, "run123")

    second = publish_run(tmp_path, "run456", replace=True)

    assert second.previous_run_digest == "run123"
    assert second.previous_public_manifest_sha256 is not None
    assert second.previous_public_manifest_path is not None
    history_path = tmp_path / second.previous_public_manifest_path
    assert history_path.exists()
    assert file_sha256(history_path) == second.previous_public_manifest_sha256
    archived_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert archived_payload["public_manifest_digest"] == first.public_manifest_digest


def test_public_publisher_payload_hashes_and_decoder_record(tmp_path):
    _complete_private_run(tmp_path, run_digest="run123", decoder=True)

    manifest = publish_run(tmp_path, "run123")

    assert manifest.published_files == tuple(
        record["path"] for record in manifest.published_file_records
    )
    assert "point_data_chunk_0_amplitudes.hdf5" in manifest.published_files
    assert "residual_chunk_0_amplitudes.hdf5" in manifest.published_files
    assert "decoder_M_test.npz" in manifest.published_files
    for record in manifest.published_file_records:
        path = tmp_path / record["path"]
        assert file_sha256(path) == record["file_sha256"]
        assert path.stat().st_size == record["payload_nbytes"]
