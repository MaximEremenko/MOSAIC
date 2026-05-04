from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from core.decoding.contracts import DecoderSourceProvenance, DisplacementDecoderSourcePolicy
from core.decoding.displacement_service import prepare_displacement_decoder_inputs
from core.decoding.decoder_cache import (
    build_decoder_cache_path,
    build_decoder_provenance_path,
    load_decoder_cache,
    save_decoder_provenance,
    save_decoder_cache,
)
from core.decoding.decoder_service import (
    DisplacementDecoderKey,
    DisplacementPatchSpec,
    DisplacementDecoderSourceService,
    apply_decoder_family,
    build_displacement_decoder_key,
    ensure_decoder,
    train_decoder_family_from_samples,
    validate_global_displacement_patch_specs,
)
from core.models import WorkflowParameters
from core.decoding.loader import resolve_output_dir


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _patch_validation_parameters(refnumbers):
    return {
        "postprocessing_mode": "displacement",
        "refnumbers": np.asarray(refnumbers, dtype=np.int64),
        "q_window_kind": "cheb",
        "q_window_at_db": 100.0,
        "edge_guard_frac": 0.1,
        "ls_weight_gamma": 0.35,
        "supercell": np.array([4]),
        "vectors": np.array([[1.0]]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
    }


def _point_row(*, central_point_id, dist, step, coordinates=None):
    return {
        "central_point_id": int(central_point_id),
        "coordinates": np.array([0.0] if coordinates is None else coordinates, dtype=np.float64),
        "dist_from_atom_center": np.array(dist, dtype=np.float64),
        "step_in_frac": np.array(step, dtype=np.float64),
    }


def test_decoder_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "decoder.npz"
    decoder = np.eye(2)
    save_decoder_cache(str(cache_path), decoder, 2, logger=_NoopLogger())
    loaded_decoder, feature_dim = load_decoder_cache(str(cache_path), logger=_NoopLogger())
    np.testing.assert_allclose(loaded_decoder, decoder)
    assert feature_dim == 2


def test_validate_global_displacement_patch_specs_accepts_single_spec():
    params = _patch_validation_parameters([1, 1])
    spec = validate_global_displacement_patch_specs(
        params,
        point_data_list=[
            _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
            _point_row(central_point_id=1, dist=[0.2], step=[0.05]),
        ],
    )

    assert spec is not None
    assert spec.dimension == 1
    assert spec.dist_from_atom_center == (0.2,)
    assert spec.step_in_frac == (0.05,)


def test_validate_global_displacement_patch_specs_fails_on_mixed_patch_specs():
    params = _patch_validation_parameters([1, 2])

    with pytest.raises(ValueError, match="single-global-decoder mode requires exactly one patch spec"):
        validate_global_displacement_patch_specs(
            params,
            point_data_list=[
                _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
                _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
            ],
        )


def test_validate_global_displacement_patch_specs_fails_for_same_reference_number_mismatch():
    params = _patch_validation_parameters([7, 7])

    with pytest.raises(ValueError, match="referenceNumber 7 appears with multiple patch specs"):
        validate_global_displacement_patch_specs(
            params,
            point_data_list=[
                _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
                _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
            ],
        )


def test_prepare_displacement_decoder_inputs_fails_before_late_feature_mismatch(monkeypatch, tmp_path):
    processor = SimpleNamespace(
        parameters=_patch_validation_parameters([1, 2]),
        original_coords=np.array([[0.0], [0.0]], dtype=np.float64),
        average_coords=np.array([[0.0], [0.0]], dtype=np.float64),
        u_true_all=None,
        _decoder_M=None,
        _feature_dim=None,
    )

    monkeypatch.setattr(
        "core.decoding.displacement_service.load_chunk_residual_field_and_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("residual-field reload should not happen before patch-spec validation")
        ),
    )
    monkeypatch.setattr(
        "core.decoding.displacement_service.build_feature_sets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("feature building should not happen before patch-spec validation")
        ),
    )

    with pytest.raises(ValueError, match="single-global-decoder mode requires exactly one patch spec"):
        prepare_displacement_decoder_inputs(
            processor,
            chunk_id=3,
            rifft_saver=None,
            point_data_list=[
                _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
                _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
            ],
            output_dir=str(tmp_path),
        )


def test_decoder_source_service_prepare_fails_before_compute_for_mixed_patch_specs(tmp_path):
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters={
            **_patch_validation_parameters([1, 2]),
            "decoder": {
                "source": "compute",
                "compute_output_directory": str(tmp_path / "full_decoder"),
            },
            "original_coords": np.array([[0.0], [0.0]], dtype=np.float64),
            "average_coords": np.array([[0.0], [0.0]], dtype=np.float64),
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="compute",
            compute_output_directory=str(tmp_path / "full_decoder"),
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    artifacts = SimpleNamespace(
        output_dir=str(tmp_path / "masked"),
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [3],
            get_point_data_for_chunk=lambda chunk_id: [
                _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
                _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
            ],
        ),
    )

    service._compute_decoder_cache = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("compute-mode decoder source should not start before patch-spec validation")
    )

    with pytest.raises(ValueError, match="single-global-decoder mode requires exactly one patch spec"):
        service.prepare(
            processor=processor,
            workflow_parameters=SimpleNamespace(),
            structure=SimpleNamespace(),
            artifacts=artifacts,
            client=None,
        )


def test_build_decoder_cache_path_is_stable(tmp_path):
    params = {
        "supercell": np.array([4, 4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0), "k_range": (0.0, 1.0)}],
        "q_window_kind": "cheb",
        "q_window_at_db": 100.0,
        "edge_guard_frac": 0.1,
        "ls_weight_gamma": 0.35,
        "dog_lambda_reg": 1e-3,
    }
    path1 = build_decoder_cache_path(params, str(tmp_path))
    path2 = build_decoder_cache_path(params, str(tmp_path))
    assert path1 == path2


def test_build_decoder_cache_path_includes_patch_window(tmp_path):
    params = {
        "supercell": np.array([4, 4, 4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
        "rspace_info": {
            "points": [
                {
                    "elementSymbol": "O",
                    "referenceNumber": 2,
                    "distFromAtomCenter": [0.3, 0.3, 0.3],
                    "stepInAngstrom": [0.05, 0.05, 0.05],
                }
            ]
        },
    }
    changed = {
        **params,
        "rspace_info": {
            "points": [
                {
                    "elementSymbol": "O",
                    "referenceNumber": 2,
                    "distFromAtomCenter": [0.5, 0.5, 0.5],
                    "stepInAngstrom": [0.05, 0.05, 0.05],
                }
            ]
        },
    }

    assert build_decoder_cache_path(params, str(tmp_path)) != build_decoder_cache_path(
        changed,
        str(tmp_path),
    )


def test_resolve_output_dir_prefers_explicit_path(tmp_path):
    path = resolve_output_dir(None, chunk_id=0, output_dir=str(tmp_path / "out"))
    assert path == str(tmp_path / "out")


def test_decoder_source_policy_defaults_and_validates():
    policy = DisplacementDecoderSourcePolicy.from_mapping(None)
    assert policy.mode == "error"
    assert policy.assignment == "single"
    assert policy.fresh_start is None
    assert DisplacementDecoderSourcePolicy.from_mapping(
        {"source": "compute", "compute_output_directory": "/tmp/full", "fresh_start": "yes"}
    ).fresh_start is True
    current_policy = DisplacementDecoderSourcePolicy.from_mapping(
        {"source": "current", "fresh_start": "yes"}
    )
    assert current_policy.mode == "current"
    assert current_policy.fresh_start is True
    with pytest.raises(ValueError, match="cache_path"):
        DisplacementDecoderSourcePolicy.from_mapping({"source": "cache"})
    with pytest.raises(ValueError, match="compute_output_directory"):
        DisplacementDecoderSourcePolicy.from_mapping({"source": "compute"})
    with pytest.raises(ValueError, match="assignment"):
        DisplacementDecoderSourcePolicy.from_mapping({"source": "error", "assignment": "weird"})


def test_save_decoder_provenance_writes_json(tmp_path):
    save_decoder_provenance(
        str(tmp_path),
        {"mode": "cache", "decoder_cache_path": "/tmp/decoder.npz"},
        logger=_NoopLogger(),
    )
    assert build_decoder_provenance_path(str(tmp_path)).endswith(
        "decoder_source_provenance.json"
    )
    assert (tmp_path / "decoder_source_provenance.json").exists()


def test_decoder_source_service_error_mode_fails_fast(tmp_path):
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters={
            "postprocessing_mode": "displacement",
            "decoder": {"source": "error"},
            "original_coords": np.array([[0.0]]),
            "average_coords": np.array([[0.0]]),
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(mode="error"),
        _decoder_M=None,
        _feature_dim=None,
    )
    artifacts = SimpleNamespace(output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="explicit decoder source"):
        service.prepare(
            processor=processor,
            workflow_parameters=SimpleNamespace(),
            structure=SimpleNamespace(),
            artifacts=artifacts,
            client=None,
        )


def test_prepare_displacement_decoder_inputs_allows_mixed_patch_specs_in_family_mode(monkeypatch, tmp_path):
    processor = SimpleNamespace(
        parameters={
            **_patch_validation_parameters([1, 2]),
            "decoder": {"source": "error", "assignment": "family"},
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(mode="error", assignment="family"),
        original_coords=np.array([[0.0], [0.0]], dtype=np.float64),
        average_coords=np.array([[0.0], [0.0]], dtype=np.float64),
        u_true_all=None,
        _decoder_M=None,
        _feature_dim=None,
        _decoder_family=None,
        _decoder_feature_dims=None,
    )

    monkeypatch.setattr(
        "core.decoding.displacement_service.load_chunk_residual_field_and_grid",
        lambda *args, **kwargs: (
            {},
            np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [0.1, 1.0]], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        "core.decoding.displacement_service.build_feature_sets",
        lambda *args, **kwargs: (
            [np.array([1.0]), np.array([1.0, 0.0])],
            [0, 1],
            [
                build_displacement_decoder_key(processor.parameters, _point_row(central_point_id=0, dist=[0.2], step=[0.05])),
                build_displacement_decoder_key(processor.parameters, _point_row(central_point_id=1, dist=[0.3], step=[0.05])),
            ],
            [],
            [],
            [],
        ),
    )

    prepared = prepare_displacement_decoder_inputs(
        processor,
        chunk_id=3,
        rifft_saver=None,
        point_data_list=[
            _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
            _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
        ],
        output_dir=str(tmp_path),
    )

    assert len(prepared["decoder_keys_all"]) == 2


def test_train_decoder_family_from_samples_and_apply_preserve_decoder_assignment():
    key_a = DisplacementDecoderKey(
        site_class_key="reference_number:1",
        patch_spec=DisplacementPatchSpec(
            dimension=1,
            dist_from_atom_center=(0.2,),
            step_in_frac=(0.05,),
            q_window_kind="cheb",
            q_window_at_db=100.0,
            edge_guard_frac=0.1,
            ls_weight_gamma=0.35,
        ),
    )
    key_b = DisplacementDecoderKey(
        site_class_key="reference_number:2",
        patch_spec=DisplacementPatchSpec(
            dimension=1,
            dist_from_atom_center=(0.3,),
            step_in_frac=(0.05,),
            q_window_kind="cheb",
            q_window_at_db=100.0,
            edge_guard_frac=0.1,
            ls_weight_gamma=0.35,
        ),
    )
    processor = SimpleNamespace(
        decoder_source_policy=DisplacementDecoderSourcePolicy(mode="error", assignment="family"),
        _decoder_M=None,
        _feature_dim=None,
        _decoder_family=None,
        _decoder_feature_dims=None,
    )

    train_decoder_family_from_samples(
        processor,
        training_features=[
            np.array([1.0]),
            np.array([2.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ],
        training_targets=[
            np.array([1.0]),
            np.array([2.0]),
            np.array([10.0]),
            np.array([20.0]),
        ],
        training_decoder_keys=[key_a, key_a, key_b, key_b],
        lam_reg=1e-9,
        logger=_NoopLogger(),
        label="test-family",
    )

    inference_features = [
        np.array([3.0]),
        np.array([0.0, 4.0]),
        np.array([5.0]),
    ]
    inference_keys = [key_a, key_b, key_a]
    ensure_decoder(
        processor,
        features_all=inference_features,
        decoder_keys_all=inference_keys,
        logger=_NoopLogger(),
    )
    result = apply_decoder_family(
        processor,
        inference_features,
        inference_keys,
    )

    np.testing.assert_allclose(result[:, 0], np.array([3.0, 80.0, 5.0]), atol=1e-6)


def test_decoder_source_service_family_mode_accepts_mixed_patch_specs(tmp_path):
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters={
            **_patch_validation_parameters([1, 2]),
            "decoder": {
                "source": "compute",
                "assignment": "family",
                "compute_output_directory": str(tmp_path / "full_decoder"),
            },
            "original_coords": np.array([[0.0], [0.0]], dtype=np.float64),
            "average_coords": np.array([[0.0], [0.0]], dtype=np.float64),
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="compute",
            assignment="family",
            compute_output_directory=str(tmp_path / "full_decoder"),
        ),
        _decoder_M=None,
        _feature_dim=None,
        _decoder_family=None,
        _decoder_feature_dims=None,
        decoder_source_provenance=None,
    )
    key_a = build_displacement_decoder_key(processor.parameters, _point_row(central_point_id=0, dist=[0.2], step=[0.05]))
    key_b = build_displacement_decoder_key(processor.parameters, _point_row(central_point_id=1, dist=[0.3], step=[0.05]))
    artifacts = SimpleNamespace(
        output_dir=str(tmp_path / "masked"),
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [3],
            get_point_data_for_chunk=lambda chunk_id: [
                _point_row(central_point_id=0, dist=[0.2], step=[0.05]),
                _point_row(central_point_id=1, dist=[0.3], step=[0.05]),
            ],
        ),
    )

    def fake_compute_decoder_cache(**kwargs):
        processor._decoder_family = {
            key_a: np.array([[1.0]], dtype=np.float64),
            key_b: np.array([[10.0, 20.0]], dtype=np.float64),
        }
        processor._decoder_feature_dims = {
            key_a: 1,
            key_b: 2,
        }
        processor._decoder_M = None
        processor._feature_dim = None
        return None, DecoderSourceProvenance(
            mode="compute",
            semantics="unmasked-family",
            decoder_cache_path="<stage2-in-memory-family>",
            source_output_directory=str(tmp_path / "full_decoder" / "processed_point_data"),
            compute_output_directory=str(tmp_path / "full_decoder"),
            feature_dim=None,
            loaded_from_cache=False,
            computed=True,
        )

    service._compute_decoder_cache = fake_compute_decoder_cache

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=SimpleNamespace(),
        structure=SimpleNamespace(),
        artifacts=artifacts,
        client=None,
    )

    assert provenance is not None
    assert provenance.semantics == "unmasked-family"
    assert processor._decoder_family is not None


def test_decoder_source_service_cache_mode_loads_existing_decoder(tmp_path):
    cache_path = tmp_path / "decoder.npz"
    save_decoder_cache(str(cache_path), np.eye(2), 2, logger=_NoopLogger())

    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters={
            "postprocessing_mode": "displacement",
            "decoder": {"source": "cache", "cache_path": str(cache_path)},
            "original_coords": np.array([[0.0]]),
            "average_coords": np.array([[0.0]]),
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="cache",
            cache_path=str(cache_path),
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    artifacts = SimpleNamespace(output_dir=str(tmp_path / "masked"))

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=SimpleNamespace(),
        structure=SimpleNamespace(),
        artifacts=artifacts,
        client=None,
    )

    assert provenance is not None
    assert provenance.mode == "cache"
    assert provenance.loaded_from_cache is True
    assert processor._feature_dim == 2
    np.testing.assert_allclose(processor._decoder_M, np.eye(2))


def test_decoder_source_service_cache_mode_accepts_cache_directory(tmp_path):
    cache_dir = tmp_path / "all" / "processed_point_data"
    cache_dir.mkdir(parents=True)
    params = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
        "decoder": {"source": "cache", "cache_path": str(cache_dir)},
        "original_coords": np.array([[0.0]]),
        "average_coords": np.array([[0.0]]),
    }
    cache_path = build_decoder_cache_path(params, str(cache_dir))
    save_decoder_cache(cache_path, np.eye(2), 2, logger=_NoopLogger())
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="cache",
            cache_path=str(cache_dir),
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    artifacts = SimpleNamespace(output_dir=str(tmp_path / "masked"))

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=SimpleNamespace(),
        structure=SimpleNamespace(),
        artifacts=artifacts,
        client=None,
    )

    assert provenance is not None
    assert provenance.decoder_cache_path == cache_path
    assert provenance.source_output_directory == str(cache_dir.resolve())
    assert provenance.loaded_from_cache is True
    np.testing.assert_allclose(processor._decoder_M, np.eye(2))


def test_decoder_source_service_current_mode_loads_existing_current_decoder(tmp_path):
    output_dir = tmp_path / "all" / "processed_point_data"
    output_dir.mkdir(parents=True)
    residual_marker = output_dir / "residual_chunk_0_amplitudes.hdf5"
    residual_marker.write_text("existing residual artifact", encoding="utf-8")
    params = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
        "decoder": {"source": "current"},
        "original_coords": np.array([[0.0]]),
        "average_coords": np.array([[0.0]]),
    }
    cache_path = build_decoder_cache_path(params, str(output_dir))
    save_decoder_cache(cache_path, np.eye(2), 2, logger=_NoopLogger())
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(mode="current"),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=SimpleNamespace(),
        structure=SimpleNamespace(),
        artifacts=SimpleNamespace(output_dir=str(output_dir)),
        client=None,
    )

    assert provenance is not None
    assert provenance.mode == "current"
    assert provenance.semantics == "current-residual"
    assert provenance.decoder_cache_path == cache_path
    assert provenance.loaded_from_cache is True
    assert provenance.computed is False
    assert residual_marker.exists()
    np.testing.assert_allclose(processor._decoder_M, np.eye(2))


def test_decoder_source_service_current_mode_fresh_start_rebuilds_only_decoder(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "all" / "processed_point_data"
    output_dir.mkdir(parents=True)
    residual_marker = output_dir / "residual_chunk_0_amplitudes.hdf5"
    residual_marker.write_text("existing residual artifact", encoding="utf-8")
    point_spec = {
        "elementSymbol": "O",
        "referenceNumber": 1,
        "distFromAtomCenter": [0.2],
        "stepInAngstrom": [0.1],
    }
    params = {
        **_patch_validation_parameters([1]),
        "rspace_info": {
            "mode": "displacement",
            "fresh_start": False,
            "points": [point_spec],
            "decoder": {
                "source": "current",
                "fresh_start": True,
            },
        },
        "decoder": {
            "source": "current",
            "fresh_start": True,
        },
        "original_coords": np.array([[0.0]], dtype=np.float64),
        "average_coords": np.array([[0.0]], dtype=np.float64),
    }
    cache_path = build_decoder_cache_path(params, str(output_dir))
    cached_decoder = np.full((1, 5), 99.0, dtype=np.float64)
    save_decoder_cache(cache_path, cached_decoder, 5, logger=_NoopLogger())
    point_row = _point_row(central_point_id=0, dist=[0.2], step=[0.1])
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(
            select=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("current decoder mode should not reselect points")
            )
        ),
        reciprocal_space_service=SimpleNamespace(
            prepare=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("current decoder mode should not prepare a decoder run")
            )
        ),
        scattering_stage=SimpleNamespace(
            execute=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("current decoder mode should not run scattering")
            )
        ),
        residual_field_stage=SimpleNamespace(
            execute=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("current decoder mode should not run residual fields")
            )
        ),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="current",
            fresh_start=True,
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    monkeypatch.setattr(
        "core.decoding.decoder_service.build_decoder_training_payload",
        lambda *args, **kwargs: {
            "features_train": [np.ones(5, dtype=np.float64)],
            "u_train": [np.array([0.25], dtype=np.float64)],
            "training_decoder_keys": [],
        },
    )

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=SimpleNamespace(),
        structure=SimpleNamespace(),
        artifacts=SimpleNamespace(
            output_dir=str(output_dir),
            saver=SimpleNamespace(),
            db_manager=SimpleNamespace(
                get_pending_chunk_ids=lambda: [0],
                get_point_data_for_chunk=lambda chunk_id: [point_row],
            ),
        ),
        client=None,
    )

    recomputed_decoder, feature_dim = load_decoder_cache(cache_path, logger=_NoopLogger())
    assert provenance is not None
    assert provenance.mode == "current"
    assert provenance.semantics == "current-residual"
    assert provenance.loaded_from_cache is False
    assert provenance.computed is True
    assert feature_dim == 5
    assert processor._feature_dim == 5
    assert residual_marker.exists()
    assert not np.allclose(recomputed_decoder, cached_decoder)


def test_decoder_source_service_compute_mode_reuses_valid_full_decoder(tmp_path, monkeypatch):
    compute_root = tmp_path / "full_decoder"
    processed_dir = compute_root / "processed_point_data"
    processed_dir.mkdir(parents=True)
    stale_marker = processed_dir / "stale.txt"
    stale_marker.write_text("decoder source artifacts", encoding="utf-8")
    params = {
        "supercell": np.array([4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
        "q_window_kind": "cheb",
        "q_window_at_db": 100.0,
        "edge_guard_frac": 0.1,
        "ls_weight_gamma": 0.35,
        "dog_lambda_reg": 1e-3,
    }
    cache_path = build_decoder_cache_path(params, str(processed_dir))
    save_decoder_cache(cache_path, np.eye(2), 2, logger=_NoopLogger())

    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters={
            **params,
            "postprocessing_mode": "displacement",
            "decoder": {
                "source": "compute",
                "compute_output_directory": str(compute_root),
            },
            "original_coords": np.array([[0.0]]),
            "average_coords": np.array([[0.0]]),
        },
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="compute",
            compute_output_directory=str(compute_root),
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    workflow_parameters = WorkflowParameters(
        schema_version=2,
        struct_info={"dimension": 1, "working_directory": str(tmp_path / "masked")},
        peak_info={"reciprocal_space_limits": []},
        rspace_info={"mode": "displacement", "fresh_start": True},
        runtime_info={},
    )
    artifacts = SimpleNamespace(output_dir=str(tmp_path / "masked"))

    monkeypatch.setattr(
        service,
        "_build_unmasked_workflow_parameters",
        lambda workflow_parameters, compute_root, **kwargs: workflow_parameters,
    )
    provenance = service.prepare(
        processor=processor,
        workflow_parameters=workflow_parameters,
        structure=SimpleNamespace(),
        artifacts=artifacts,
        client=None,
    )

    assert provenance is not None
    assert provenance.mode == "compute"
    published_cache_path = build_decoder_cache_path(params, str(tmp_path / "masked"))
    assert provenance.decoder_cache_path == published_cache_path
    assert provenance.loaded_from_cache is True
    assert provenance.computed is False
    assert processor._feature_dim == 2
    assert stale_marker.exists()
    assert Path(published_cache_path).exists()


def test_decoder_source_service_compute_mode_rebuilds_stale_decoder_cache(
    tmp_path,
    monkeypatch,
):
    compute_root = tmp_path / "full_decoder"
    processed_dir = compute_root / "processed_point_data"
    processed_dir.mkdir(parents=True)
    stale_marker = processed_dir / "stale.txt"
    stale_marker.write_text("old artifacts", encoding="utf-8")
    point_spec = {
        "elementSymbol": "O",
        "referenceNumber": 1,
        "distFromAtomCenter": [0.2],
        "stepInAngstrom": [0.1],
    }
    params = {
        **_patch_validation_parameters([1]),
        "rspace_info": {
            "mode": "displacement",
            "fresh_start": True,
            "points": [point_spec],
            "decoder": {
                "source": "compute",
                "compute_output_directory": str(compute_root),
                "fresh_start": True,
            },
        },
        "decoder": {
            "source": "compute",
            "compute_output_directory": str(compute_root),
            "fresh_start": True,
        },
        "original_coords": np.array([[0.0]], dtype=np.float64),
        "average_coords": np.array([[0.0]], dtype=np.float64),
    }
    cache_path = build_decoder_cache_path(params, str(processed_dir))
    save_decoder_cache(cache_path, np.eye(3), 3, logger=_NoopLogger())
    point_row = _point_row(central_point_id=0, dist=[0.2], step=[0.1])

    class _FakeArtifacts(SimpleNamespace):
        def close(self):
            return None

    fake_artifacts = _FakeArtifacts(
        output_dir=str(processed_dir),
        padded_intervals=[{"h_range": (0.0, 1.0)}],
        saver=SimpleNamespace(),
        point_data_processor=SimpleNamespace(),
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [0],
            get_point_data_for_chunk=lambda chunk_id: [point_row],
        ),
    )
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(select=lambda request: SimpleNamespace()),
        reciprocal_space_service=SimpleNamespace(
            prepare=lambda **kwargs: fake_artifacts,
        ),
        scattering_stage=SimpleNamespace(execute=lambda **kwargs: {}),
        residual_field_stage=SimpleNamespace(execute=lambda **kwargs: None),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="compute",
            compute_output_directory=str(compute_root),
            fresh_start=True,
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    workflow_parameters = WorkflowParameters(
        schema_version=2,
        struct_info={"dimension": 1, "working_directory": str(tmp_path / "masked")},
        peak_info={"reciprocal_space_limits": []},
        rspace_info={
            "mode": "displacement",
            "fresh_start": False,
            "points": [point_spec],
        },
        runtime_info={},
    )
    structure = SimpleNamespace(
        supercell=np.array([4], dtype=np.int64),
        original_coords=np.array([[0.0]], dtype=np.float64),
        average_coords=np.array([[0.0]], dtype=np.float64),
        cells_origin=np.array([[0]], dtype=np.int64),
        elements=np.array(["O"], dtype=object),
        refnumbers=np.array([1], dtype=np.int64),
        vectors=np.array([[1.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "core.decoding.decoder_service.build_decoder_training_payload",
        lambda *args, **kwargs: {
            "features_train": [np.ones(5, dtype=np.float64)],
            "u_train": [np.array([0.25], dtype=np.float64)],
            "training_decoder_keys": [],
        },
    )

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=SimpleNamespace(
            output_dir=str(tmp_path / "masked"),
            db_manager=SimpleNamespace(
                get_pending_chunk_ids=lambda: [0],
                get_point_data_for_chunk=lambda chunk_id: [point_row],
            ),
        ),
        client=None,
    )

    assert provenance is not None
    assert provenance.mode == "compute"
    assert provenance.loaded_from_cache is False
    assert provenance.computed is True
    assert processor._feature_dim == 5
    assert not stale_marker.exists()


def test_decoder_source_service_compute_mode_fresh_start_rebuilds_valid_decoder_cache(
    tmp_path,
    monkeypatch,
):
    compute_root = tmp_path / "full_decoder"
    processed_dir = compute_root / "processed_point_data"
    processed_dir.mkdir(parents=True)
    stale_marker = processed_dir / "stale.txt"
    stale_marker.write_text("old decoder artifacts", encoding="utf-8")
    masked_root = tmp_path / "masked"
    masked_root.mkdir()
    keep_marker = masked_root / "keep.txt"
    keep_marker.write_text("main run artifact", encoding="utf-8")
    point_spec = {
        "elementSymbol": "O",
        "referenceNumber": 1,
        "distFromAtomCenter": [0.2],
        "stepInAngstrom": [0.1],
    }
    params = {
        **_patch_validation_parameters([1]),
        "rspace_info": {
            "mode": "displacement",
            "fresh_start": False,
            "points": [point_spec],
            "decoder": {
                "source": "compute",
                "compute_output_directory": str(compute_root),
                "fresh_start": True,
            },
        },
        "decoder": {
            "source": "compute",
            "compute_output_directory": str(compute_root),
            "fresh_start": True,
        },
        "original_coords": np.array([[0.0]], dtype=np.float64),
        "average_coords": np.array([[0.0]], dtype=np.float64),
    }
    cache_path = build_decoder_cache_path(params, str(processed_dir))
    cached_decoder = np.full((1, 5), 99.0, dtype=np.float64)
    save_decoder_cache(cache_path, cached_decoder, 5, logger=_NoopLogger())
    point_row = _point_row(central_point_id=0, dist=[0.2], step=[0.1])

    class _FakeArtifacts(SimpleNamespace):
        def close(self):
            return None

    fake_artifacts = _FakeArtifacts(
        output_dir=str(processed_dir),
        padded_intervals=[{"h_range": (0.0, 1.0)}],
        saver=SimpleNamespace(),
        point_data_processor=SimpleNamespace(),
        db_manager=SimpleNamespace(
            get_pending_chunk_ids=lambda: [0],
            get_point_data_for_chunk=lambda chunk_id: [point_row],
        ),
    )
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(select=lambda request: SimpleNamespace()),
        reciprocal_space_service=SimpleNamespace(
            prepare=lambda **kwargs: fake_artifacts,
        ),
        scattering_stage=SimpleNamespace(execute=lambda **kwargs: {}),
        residual_field_stage=SimpleNamespace(execute=lambda **kwargs: None),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="compute",
            compute_output_directory=str(compute_root),
            fresh_start=True,
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )
    workflow_parameters = WorkflowParameters(
        schema_version=2,
        struct_info={"dimension": 1, "working_directory": str(masked_root)},
        peak_info={"reciprocal_space_limits": []},
        rspace_info={
            "mode": "displacement",
            "fresh_start": False,
            "points": [point_spec],
        },
        runtime_info={},
    )
    structure = SimpleNamespace(
        supercell=np.array([4], dtype=np.int64),
        original_coords=np.array([[0.0]], dtype=np.float64),
        average_coords=np.array([[0.0]], dtype=np.float64),
        cells_origin=np.array([[0]], dtype=np.int64),
        elements=np.array(["O"], dtype=object),
        refnumbers=np.array([1], dtype=np.int64),
        vectors=np.array([[1.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "core.decoding.decoder_service.build_decoder_training_payload",
        lambda *args, **kwargs: {
            "features_train": [np.ones(5, dtype=np.float64)],
            "u_train": [np.array([0.25], dtype=np.float64)],
            "training_decoder_keys": [],
        },
    )

    provenance = service.prepare(
        processor=processor,
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=SimpleNamespace(
            output_dir=str(masked_root),
            db_manager=SimpleNamespace(
                get_pending_chunk_ids=lambda: [0],
                get_point_data_for_chunk=lambda chunk_id: [point_row],
            ),
        ),
        client=None,
    )

    recomputed_decoder, feature_dim = load_decoder_cache(cache_path, logger=_NoopLogger())
    assert provenance is not None
    assert provenance.mode == "compute"
    assert provenance.loaded_from_cache is False
    assert provenance.computed is True
    assert feature_dim == 5
    assert processor._feature_dim == 5
    assert not np.allclose(recomputed_decoder, cached_decoder)
    assert not stale_marker.exists()
    assert processed_dir.exists()
    assert keep_marker.exists()
