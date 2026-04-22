import numpy as np

from core.scattering.contracts import (
    SCATTERING_CHUNK_ARTIFACT_SCHEMA,
    SCATTERING_PARTIAL_RESULT_MERGE_INVARIANTS,
    ScatteringArtifactManifest,
    ScatteringPartialResult,
    ScatteringWorkUnit,
    build_chunk_artifact_refs,
    merge_scattering_partial_results,
    scattering_partial_result_identity,
    validate_scattering_work_unit,
)
from core.residual_field.contracts import (
    RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA,
    RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS,
    ResidualFieldArtifactManifest,
    ResidualFieldPartialResult,
    ResidualFieldWorkUnit,
    build_residual_field_output_artifacts,
    merge_residual_field_partial_results,
    residual_field_partial_result_identity,
    validate_residual_field_work_unit,
)
from core.contracts import CompletionStatus, RetryDisposition


def test_scattering_work_unit_and_manifest_are_deterministic(tmp_path):
    work_unit = ScatteringWorkUnit.interval_chunk(
        interval_id=7,
        chunk_id=3,
        dimension=2,
        output_dir=str(tmp_path),
    )
    validate_scattering_work_unit(work_unit)

    assert work_unit.retry.idempotency_key == "scattering:interval-chunk:7:3"
    assert work_unit.chunk_artifact_prefix.endswith("point_data_chunk_3")
    assert work_unit.interval_artifact is not None
    assert work_unit.interval_artifact.path.endswith("precomputed_intervals/interval_7.npz")

    manifest = ScatteringArtifactManifest.from_work_unit(
        work_unit,
        artifacts=build_chunk_artifact_refs(str(tmp_path), 3),
        completion_status=CompletionStatus.COMMITTED,
        consumer_stage="residual_field",
    )

    assert manifest.completion_status is CompletionStatus.COMMITTED
    assert manifest.retry.replay_disposition is RetryDisposition.NO_OP
    assert manifest.consumer_stage == "residual_field"
    assert len(manifest.artifacts) == 6
    assert manifest.artifact_schema_name == SCATTERING_CHUNK_ARTIFACT_SCHEMA.name


def test_scattering_partial_result_merge_is_additive_and_rejects_duplicates():
    left = ScatteringPartialResult(
        chunk_id=3,
        contributing_interval_ids=(1,),
        point_ids=np.array([10, 11]),
        grid_shape_nd=np.array([[2, 2]]),
        amplitudes_delta=np.array([1.0 + 1.0j, 2.0 + 0.0j]),
        amplitudes_average=np.array([0.25 + 0.0j, 0.5 + 0.0j]),
        reciprocal_point_count=5,
    )
    right = ScatteringPartialResult(
        chunk_id=3,
        contributing_interval_ids=(2,),
        point_ids=np.array([10, 11]),
        grid_shape_nd=np.array([[2, 2]]),
        amplitudes_delta=np.array([3.0 + 0.0j, 4.0 + 1.0j]),
        amplitudes_average=np.array([0.75 + 0.0j, 1.5 + 0.0j]),
        reciprocal_point_count=7,
    )

    merged = merge_scattering_partial_results(left, right)

    assert merged.contributing_interval_ids == (1, 2)
    np.testing.assert_allclose(merged.amplitudes_delta, np.array([4.0 + 1.0j, 6.0 + 1.0j]))
    np.testing.assert_allclose(merged.amplitudes_average, np.array([1.0 + 0.0j, 2.0 + 0.0j]))
    assert merged.reciprocal_point_count == 12

    duplicate = ScatteringPartialResult(
        chunk_id=3,
        contributing_interval_ids=(2,),
        point_ids=np.array([10, 11]),
        grid_shape_nd=np.array([[2, 2]]),
        amplitudes_delta=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j, 0.0 + 0.0j]),
        reciprocal_point_count=0,
    )
    try:
        merge_scattering_partial_results(merged, duplicate)
    except ValueError as exc:
        assert "duplicate interval ids" in str(exc)
    else:
        raise AssertionError("Expected duplicate interval ids to be rejected.")


def test_scattering_partial_result_identity_matches_documented_invariants():
    identity = scattering_partial_result_identity(
        chunk_id=5,
        point_ids=np.array([21, 22, 23]),
        grid_shape_nd=np.array([[3, 3]]),
    )

    assert identity.contributing_interval_ids == ()
    np.testing.assert_allclose(identity.amplitudes_delta, np.zeros(3, dtype=np.complex128))
    np.testing.assert_allclose(identity.amplitudes_average, np.zeros(3, dtype=np.complex128))
    assert identity.reciprocal_point_count == 0
    assert SCATTERING_PARTIAL_RESULT_MERGE_INVARIANTS.associative is True


def test_residual_field_work_unit_and_manifest_keep_seam_neutral(tmp_path):
    work_unit = ResidualFieldWorkUnit.chunk_scope(
        chunk_id=4,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
        patch_scope="chunk",
        window_spec="cheb:100db",
    )
    interval_work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=2,
        chunk_id=4,
        parameter_digest="abc123",
        output_dir=str(tmp_path),
        patch_scope="chunk",
        window_spec="cheb:100db",
    )
    validate_residual_field_work_unit(interval_work_unit)

    assert work_unit.retry.idempotency_key == "residual-field:chunk:4:abc123"
    assert interval_work_unit.retry.idempotency_key == "residual-field:chunk:4:abc123:interval-2"
    assert all(artifact.stage == "scattering" for artifact in interval_work_unit.source_artifacts)
    assert build_residual_field_output_artifacts(str(tmp_path), 4)[0].path.endswith(
        "residual_chunk_4_amplitudes.hdf5"
    )

    manifest = ResidualFieldArtifactManifest.from_work_unit(
        interval_work_unit,
        artifacts=(),
        completion_status=CompletionStatus.PLANNED,
    )

    assert manifest.producer_stage == "residual_field"
    assert manifest.consumer_stage == "decoding"
    assert manifest.retry.replay_disposition is RetryDisposition.NO_OP
    assert manifest.artifact_schema_name == RESIDUAL_FIELD_CHUNK_ARTIFACT_SCHEMA.name


def test_residual_field_partial_result_merge_is_metadata_oriented():
    left = ResidualFieldPartialResult(
        chunk_id=9,
        contributing_interval_ids=(1,),
        parameter_digest="p123",
        output_kind="residual-field-chunk",
        source_artifacts=(),
        output_artifacts=(),
        grid_shape=(4, 4),
        point_ids=(3, 1),
    )
    right = ResidualFieldPartialResult(
        chunk_id=9,
        contributing_interval_ids=(2,),
        parameter_digest="p123",
        output_kind="residual-field-chunk",
        source_artifacts=(),
        output_artifacts=(),
        grid_shape=(4, 4),
        point_ids=(2, 3),
    )

    merged = merge_residual_field_partial_results(left, right)

    assert merged.contributing_interval_ids == (1, 2)
    assert merged.point_ids == (1, 2, 3)
    assert merged.grid_shape == (4, 4)
    assert RESIDUAL_FIELD_PARTIAL_RESULT_MERGE_INVARIANTS.associative is True

    duplicate = ResidualFieldPartialResult(
        chunk_id=9,
        contributing_interval_ids=(2,),
        parameter_digest="p123",
        output_kind="residual-field-chunk",
        source_artifacts=(),
        output_artifacts=(),
        grid_shape=(4, 4),
        point_ids=(2, 3),
    )
    try:
        merge_residual_field_partial_results(merged, duplicate)
    except ValueError as exc:
        assert "duplicate interval ids" in str(exc)
    else:
        raise AssertionError("Expected duplicate interval ids to be rejected.")

    identity = residual_field_partial_result_identity(
        chunk_id=9,
        parameter_digest="p123",
        output_kind="residual-field-chunk",
        grid_shape=(4, 4),
    )
    assert identity.output_artifacts == ()
    assert identity.point_ids == ()
