from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.decoding.contracts import DisplacementDecoderSourcePolicy
from core.decoding.decoder_cache import (
    build_decoder_cache_path,
    resolve_current_residual_source_identity,
    save_decoder_cache,
)
from core.decoding.decoder_service import DisplacementDecoderSourceService
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


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _write_public_residual_manifest(tmp_path):
    write_residual_attempt(
        output_dir=tmp_path,
        chunk_id=0,
        partition_id=0,
        point_start=0,
        point_stop=1,
        interval_ids=(1,),
        attempt_id="try1",
        point_ids=np.array([0], dtype=np.int64),
        grid_shape_nd=np.array([[1]], dtype=np.int64),
        amplitudes_delta=np.array([1.0 + 0.0j]),
        amplitudes_average=np.array([0.0 + 0.0j]),
        contribution_reciprocal_points=1,
        **IDENTITY,
    )
    candidate = create_residual_commit_candidate(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_id=0,
        expected_partitions={0: (0,)},
        expected_reciprocal_point_count=1,
    )
    write_residual_stage_plan(
        output_dir=tmp_path,
        run_digest="run123",
        expected_by_chunk={0: (0,)},
    )
    promote_residual_chunk_commit(output_dir=tmp_path, candidate=candidate)
    write_residual_stage_commit(
        output_dir=tmp_path,
        run_digest="run123",
        chunk_ids=(0,),
    )
    source_identity = resolve_current_residual_source_identity(
        output_dir=tmp_path,
        run_digest="run123",
    )
    return write_public_manifest(
        tmp_path / "public_manifest.json",
        output_dir=tmp_path,
        run_digest="run123",
        source_stage="residual_field",
        source_identity=source_identity,
    )


def test_decoder_current_source_requires_residual_stage_commit(tmp_path):
    with pytest.raises(RuntimeError, match="stage_commit"):
        resolve_current_residual_source_identity(
            output_dir=tmp_path,
            run_digest="run123",
        )


def test_decoder_current_source_rejects_loose_residual_chunk_files(tmp_path):
    (tmp_path / "residual_chunk_0_amplitudes.hdf5").write_text(
        "stale residual data",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Loose residual chunk"):
        resolve_current_residual_source_identity(
            output_dir=tmp_path,
            run_digest="run123",
        )


def test_current_source_service_validates_stage_commit_before_cache_lookup(tmp_path):
    params = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "points": [0],
        "original_coords": np.array([[0.0]]),
        "average_coords": np.array([[0.0]]),
        "scattering_run_digest": "run123",
        "decoder": {"source": "current"},
    }
    cache_path = build_decoder_cache_path(params, str(tmp_path))
    save_decoder_cache(cache_path, np.eye(1), 1, logger=_NoopLogger())
    (tmp_path / "residual_chunk_0_amplitudes.hdf5").write_text(
        "stale residual data",
        encoding="utf-8",
    )
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

    with pytest.raises(RuntimeError, match="Loose residual chunk"):
        service.prepare(
            processor=processor,
            workflow_parameters=SimpleNamespace(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path)),
            client=None,
        )


def test_current_source_service_accepts_explicit_public_manifest_before_cache_lookup(tmp_path):
    _write_public_residual_manifest(tmp_path)
    params = {
        "postprocessing_mode": "displacement",
        "supercell": np.array([4]),
        "points": [0],
        "original_coords": np.array([[0.0]]),
        "average_coords": np.array([[0.0]]),
        "decoder": {
            "source": "current",
            "public_manifest_path": "public_manifest.json",
        },
    }
    service = DisplacementDecoderSourceService(
        point_selection_service=SimpleNamespace(),
        reciprocal_space_service=SimpleNamespace(),
        scattering_stage=SimpleNamespace(),
        residual_field_stage=SimpleNamespace(),
    )
    processor = SimpleNamespace(
        parameters=params,
        decoder_source_policy=DisplacementDecoderSourcePolicy(
            mode="current",
            public_manifest_path="public_manifest.json",
        ),
        _decoder_M=None,
        _feature_dim=None,
        decoder_source_provenance=None,
    )

    with pytest.raises(RuntimeError, match="requires current-run residual artifacts"):
        service.prepare(
            processor=processor,
            workflow_parameters=SimpleNamespace(),
            structure=SimpleNamespace(),
            artifacts=SimpleNamespace(output_dir=str(tmp_path)),
            client=None,
        )
