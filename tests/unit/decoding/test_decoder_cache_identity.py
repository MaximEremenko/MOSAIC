from __future__ import annotations

import numpy as np

from core.decoding.decoder_cache import (
    build_decoder_cache_identity,
    build_decoder_cache_path,
    resolve_current_residual_source_identity,
    resolve_public_residual_source_identity,
)
from core.residual_field.commit import (
    create_residual_commit_candidate,
    promote_residual_chunk_commit,
    write_residual_attempt,
    write_residual_stage_commit,
    write_residual_stage_plan,
)
from core.storage.public_manifest import write_public_manifest


IDENTITY = {
    "run_digest": "run123",
    "parameter_digest": "d" * 64,
    "partition_plan_digest": "e" * 64,
    "source_scattering_commit_digest": "a" * 64,
    "source_replacement_digest": None,
    "backend_policy_digest": "b" * 64,
    "expected_output_digest": "0" * 64,
}


def _residual_stage_commit(tmp_path):
    write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=5,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id="try1",
        point_ids=np.array([10], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=9,
        **IDENTITY,
    )
    candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=5,
        expected_partitions={0: (10,)},
        expected_reciprocal_point_count=9,
    )
    write_residual_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={5: (0,)},
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=candidate)
    return write_residual_stage_commit(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_ids=(5,),
    )


def test_decoder_cache_identity_includes_residual_source_and_model_inputs(tmp_path):
    stage_commit = _residual_stage_commit(tmp_path)
    source = resolve_current_residual_source_identity(
        output_dir=tmp_path,
        run_digest="run123",
    )

    identity = build_decoder_cache_identity(
        residual_source_identity=source,
        coordinate_digest="c" * 64,
        vector_digest="v" * 64,
        refnumber_digest="f" * 64,
        feature_mode="single",
        target_parameters={"dog_lambda_reg": 1e-3},
        decoder_architecture_digest="a" * 64,
        code_version="test",
    )
    changed = build_decoder_cache_identity(
        residual_source_identity=source,
        coordinate_digest="c" * 64,
        vector_digest="v" * 64,
        refnumber_digest="f" * 64,
        feature_mode="family",
        target_parameters={"dog_lambda_reg": 1e-3},
        decoder_architecture_digest="a" * 64,
        code_version="test",
    )

    assert source["residual_stage_digest"] == stage_commit.stage_digest
    assert source["upstream_scattering_identity"][0]["source_scattering_commit_digest"] == "a" * 64
    assert identity["decoder_cache_digest"] != changed["decoder_cache_digest"]
    assert build_decoder_cache_path(
        {"supercell": [1], "points": [0]},
        str(tmp_path),
        source_identity=identity,
    ) != build_decoder_cache_path(
        {"supercell": [1], "points": [0]},
        str(tmp_path),
        source_identity=changed,
    )


def test_decoder_source_identity_accepts_validated_public_manifest(tmp_path):
    _residual_stage_commit(tmp_path)
    private_source = resolve_current_residual_source_identity(
        output_dir=tmp_path,
        run_digest="run123",
    )
    public_manifest = write_public_manifest(
        tmp_path / "public_manifest.json",
        output_dir=tmp_path,
        run_digest="run123",
        source_stage="residual_field",
        source_identity=private_source,
    )

    public_source = resolve_public_residual_source_identity(
        output_dir=tmp_path,
        public_manifest_path="public_manifest.json",
    )

    assert public_source["schema"] == "mosaic.decoder.public_residual_source"
    assert public_source["public_manifest_digest"] == public_manifest.public_manifest_digest
    assert public_source["residual_stage_digest"] == private_source["residual_stage_digest"]
    assert public_source["residual_payload_hashes"] == private_source["residual_payload_hashes"]
