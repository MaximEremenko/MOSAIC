from types import SimpleNamespace

import numpy as np
import pytest

from core.decoding.contracts import DecoderSourceProvenance, DisplacementDecoderSourcePolicy
from core.decoding.decoder_cache import (
    build_decoder_cache_path,
    build_decoder_provenance_path,
    load_decoder_cache,
    save_decoder_provenance,
    save_decoder_cache,
)
from core.decoding.decoder_service import DisplacementDecoderSourceService
from core.models import WorkflowParameters
from core.decoding.loader import resolve_output_dir


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_decoder_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "decoder.npz"
    decoder = np.eye(2)
    save_decoder_cache(str(cache_path), decoder, 2, logger=_NoopLogger())
    loaded_decoder, feature_dim = load_decoder_cache(str(cache_path), logger=_NoopLogger())
    np.testing.assert_allclose(loaded_decoder, decoder)
    assert feature_dim == 2


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


def test_resolve_output_dir_prefers_explicit_path(tmp_path):
    path = resolve_output_dir(None, chunk_id=0, output_dir=str(tmp_path / "out"))
    assert path == str(tmp_path / "out")


def test_decoder_source_policy_defaults_and_validates():
    assert DisplacementDecoderSourcePolicy.from_mapping(None).mode == "error"
    with pytest.raises(ValueError, match="cache_path"):
        DisplacementDecoderSourcePolicy.from_mapping({"source": "cache"})
    with pytest.raises(ValueError, match="compute_output_directory"):
        DisplacementDecoderSourcePolicy.from_mapping({"source": "compute"})


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


def test_decoder_source_service_compute_mode_reuses_valid_full_decoder(tmp_path, monkeypatch):
    compute_root = tmp_path / "full_decoder"
    processed_dir = compute_root / "processed_point_data"
    processed_dir.mkdir(parents=True)
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
        rspace_info={"mode": "displacement"},
        runtime_info={},
    )
    artifacts = SimpleNamespace(output_dir=str(tmp_path / "masked"))

    monkeypatch.setattr(
        service,
        "_build_unmasked_workflow_parameters",
        lambda workflow_parameters, compute_root: workflow_parameters,
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
    assert provenance.loaded_from_cache is True
    assert provenance.computed is False
    assert processor._feature_dim == 2
