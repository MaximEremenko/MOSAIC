from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from core.runtime.log_utils import short_path
from core.storage.attempt_store import relative_to_output, stage_commit_path
from core.storage.digests import digest_dict
from core.storage.manifest import read_manifest
from core.storage.public_manifest import read_public_manifest
from core.residual_field.commit import (
    RESIDUAL_FIELD_STAGE,
    ResidualChunkCommitManifest,
    ResidualCommitCandidateManifest,
    ResidualStageCommitManifest,
    validate_residual_commit_candidate,
)


def _to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def build_decoder_cache_path(
    parameters: dict,
    output_dir: str,
    *,
    source_identity: dict | None = None,
) -> str:
    intervals = parameters.get("reciprocal_space_intervals_all", [])
    rspace_info = parameters.get("rspace_info", {}) or {}
    points = parameters.get("points", rspace_info.get("points", []))
    key_obj = {
        "supercell": _to_plain(np.asarray(parameters["supercell"], int)),
        "intervals": _to_plain(intervals),
        "points": _to_plain(points),
        "q_window_kind": parameters.get("q_window_kind", "cheb"),
        "q_window_at_db": float(parameters.get("q_window_at_db", 100.0)),
        "edge_guard_frac": float(parameters.get("edge_guard_frac", 0.10)),
        "ls_weight_gamma": float(parameters.get("ls_weight_gamma", 0.35)),
        "dog_lambda_reg": float(parameters.get("dog_lambda_reg", 1e-3)),
    }
    if source_identity is not None:
        key_obj["source_identity"] = _to_plain(source_identity)
    key_json = json.dumps(key_obj, sort_keys=True)
    digest = hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]
    return os.path.join(output_dir, f"decoder_M_{digest}.npz")


def build_decoder_cache_identity(
    *,
    residual_source_identity: dict,
    coordinate_digest: str,
    average_coordinate_digest: str | None = None,
    vector_digest: str,
    refnumber_digest: str | None,
    feature_mode: str,
    target_parameters: dict,
    decoder_architecture_digest: str,
    code_version: str,
    schema_version: int = 1,
) -> dict:
    payload = {
        "schema": "mosaic.decoder.cache_identity",
        "schema_version": int(schema_version),
        "residual_source_identity": _to_plain(residual_source_identity),
        "coordinate_digest": str(coordinate_digest),
        "average_coordinate_digest": (
            None if average_coordinate_digest is None else str(average_coordinate_digest)
        ),
        "vector_digest": str(vector_digest),
        "refnumber_digest": None if refnumber_digest is None else str(refnumber_digest),
        "feature_mode": str(feature_mode),
        "target_parameters": _to_plain(target_parameters),
        "decoder_architecture_digest": str(decoder_architecture_digest),
        "code_version": str(code_version),
    }
    payload["decoder_cache_digest"] = digest_dict(
        payload,
        domain="mosaic.decoder.cache_identity.v1",
    )
    return payload


def resolve_current_residual_source_identity(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> dict:
    output_root = Path(output_dir)
    stale_prefix = "residual" "_chunk_"
    loose_residual_files = sorted(output_root.glob(stale_prefix + "*"))
    if loose_residual_files:
        raise RuntimeError(
            "Loose residual chunk files are not valid current decoder source evidence. "
            "Use residual_field/stage_commit.json for private current runs."
        )
    path = stage_commit_path(output_root, run_digest, RESIDUAL_FIELD_STAGE)
    if not path.exists():
        raise RuntimeError(
            "Current decoder source requires residual_field/stage_commit.json for "
            f"run {run_digest!r}."
        )
    stage_commit = read_manifest(
        path,
        codec=ResidualStageCommitManifest,
        output_dir=output_root,
    )
    payload_hashes = []
    upstream_identities = []
    for chunk_commit_rel in stage_commit.chunk_commit_paths:
        chunk_commit = read_manifest(
            output_root / chunk_commit_rel,
            codec=ResidualChunkCommitManifest,
            output_dir=output_root,
        )
        candidate = read_manifest(
            output_root / chunk_commit.candidate_manifest_path,
            codec=ResidualCommitCandidateManifest,
            output_dir=output_root,
        )
        validate_residual_commit_candidate(candidate, output_dir=output_root)
        payload_hashes.append(
            {
                "chunk_id": int(chunk_commit.chunk_id),
                "candidate_id": chunk_commit.selected_candidate_id,
                "payload_sha256": chunk_commit.payload_sha256,
                "file_sha256": chunk_commit.file_sha256,
                "payload_nbytes": int(chunk_commit.payload_nbytes),
                "candidate_payload_path": chunk_commit.candidate_payload_path,
            }
        )
        if candidate.source_identity is not None:
            upstream_identities.append(candidate.source_identity)
    source_identity = {
        "schema": "mosaic.decoder.current_residual_source",
        "schema_version": 1,
        "run_digest": str(run_digest),
        "stage_commit_path": relative_to_output(path, output_dir=output_root),
        "residual_stage_digest": stage_commit.stage_digest,
        "residual_stage_plan_digest": stage_commit.stage_plan_digest,
        "residual_payload_hashes": payload_hashes,
        "upstream_scattering_identity": upstream_identities,
    }
    source_identity["source_identity_digest"] = digest_dict(
        source_identity,
        domain="mosaic.decoder.current_residual_source.v1",
    )
    return source_identity


def _require_residual_public_source_fields(source_identity: dict) -> None:
    if not isinstance(source_identity.get("residual_stage_digest"), str):
        raise RuntimeError("public_manifest.json is missing residual_stage_digest source identity.")
    if not isinstance(source_identity.get("residual_payload_hashes"), list):
        raise RuntimeError("public_manifest.json is missing residual payload hash identity.")
    if not isinstance(source_identity.get("upstream_scattering_identity"), list):
        raise RuntimeError("public_manifest.json is missing upstream scattering identity.")


def resolve_public_residual_source_identity(
    *,
    output_dir: str | Path,
    public_manifest_path: str | Path,
) -> dict:
    output_root = Path(output_dir)
    public_manifest = read_public_manifest(
        public_manifest_path,
        output_dir=output_root,
    )
    if public_manifest.source_stage != RESIDUAL_FIELD_STAGE:
        raise RuntimeError(
            "processing.decoder.source='current' public manifest must describe "
            "a residual_field source."
        )
    public_source_identity = dict(public_manifest.source_identity)
    _require_residual_public_source_fields(public_source_identity)
    manifest_path = Path(public_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = output_root / manifest_path
    source_identity = {
        "schema": "mosaic.decoder.public_residual_source",
        "schema_version": 1,
        "run_digest": public_manifest.run_digest,
        "public_manifest_path": relative_to_output(manifest_path, output_dir=output_root),
        "public_manifest_digest": public_manifest.public_manifest_digest,
        "residual_stage_digest": public_source_identity["residual_stage_digest"],
        "residual_stage_plan_digest": public_source_identity.get("residual_stage_plan_digest"),
        "residual_payload_hashes": list(public_source_identity["residual_payload_hashes"]),
        "upstream_scattering_identity": list(
            public_source_identity["upstream_scattering_identity"]
        ),
        "public_source_identity": public_source_identity,
    }
    source_identity["source_identity_digest"] = digest_dict(
        source_identity,
        domain="mosaic.decoder.public_residual_source.v1",
    )
    return source_identity


def build_decoder_provenance_path(output_dir: str) -> str:
    return os.path.join(output_dir, "decoder_source_provenance.json")


def load_decoder_cache(cache_path: str, logger):
    if not os.path.isfile(cache_path):
        return None, None
    try:
        cache = np.load(cache_path, allow_pickle=False)
        decoder = np.asarray(cache["M"], float)
        feature_dim = cache.get("feature_dim", None)
        if feature_dim is not None:
            feature_dim = int(np.ravel(feature_dim)[0])
        else:
            feature_dim = decoder.shape[1]
        logger.info(
            "Loaded decoder M from '%s' (shape %s, feature_dim=%d).",
            short_path(cache_path),
            decoder.shape,
            feature_dim,
        )
        return decoder, feature_dim
    except Exception as exc:
        logger.warning(
            "Failed to load decoder M from '%s': %s. Will retrain.",
            short_path(cache_path),
            exc,
        )
        return None, None


def save_decoder_cache(cache_path: str, decoder_M, feature_dim: int, logger) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as handle:
            np.savez(
                handle,
                M=decoder_M,
                feature_dim=np.array(feature_dim, dtype=np.int64),
            )
        os.replace(temp_path, path)
        logger.info("Decoder M saved to '%s'.", short_path(path))
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.warning("Failed to save decoder M to '%s': %s", short_path(path), exc)
        raise


def save_decoder_provenance(output_dir: str, provenance: dict, logger) -> None:
    path = Path(build_decoder_provenance_path(output_dir))
    try:
        cache_path = provenance.get("decoder_cache_path")
        if isinstance(cache_path, str) and cache_path and not cache_path.startswith("<"):
            resolved_cache = Path(cache_path)
            if not resolved_cache.is_absolute():
                resolved_cache = Path(output_dir) / resolved_cache
            if resolved_cache.exists():
                decoder, feature_dim = load_decoder_cache(str(resolved_cache), logger)
                if decoder is None or feature_dim is None:
                    raise RuntimeError(
                        f"Decoder provenance references an unreadable cache: {resolved_cache}"
                    )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Decoder source provenance saved to '%s'.", short_path(path))
    except Exception as exc:
        logger.warning("Failed to save decoder provenance to '%s': %s", short_path(path), exc)
        raise
