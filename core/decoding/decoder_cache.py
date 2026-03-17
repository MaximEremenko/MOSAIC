from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from core.runtime.log_utils import short_path


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


def build_decoder_cache_path(parameters: dict, output_dir: str) -> str:
    intervals = parameters.get("reciprocal_space_intervals_all", [])
    key_obj = {
        "supercell": _to_plain(np.asarray(parameters["supercell"], int)),
        "intervals": _to_plain(intervals),
        "q_window_kind": parameters.get("q_window_kind", "cheb"),
        "q_window_at_db": float(parameters.get("q_window_at_db", 100.0)),
        "edge_guard_frac": float(parameters.get("edge_guard_frac", 0.10)),
        "ls_weight_gamma": float(parameters.get("ls_weight_gamma", 0.35)),
        "dog_lambda_reg": float(parameters.get("dog_lambda_reg", 1e-3)),
    }
    key_json = json.dumps(key_obj, sort_keys=True)
    digest = hashlib.sha1(key_json.encode("utf-8")).hexdigest()[:16]
    return os.path.join(output_dir, f"decoder_M_{digest}.npz")


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
    try:
        np.savez(
            cache_path,
            M=decoder_M,
            feature_dim=np.array(feature_dim, dtype=np.int64),
        )
        logger.info("Decoder M saved to '%s'.", short_path(cache_path))
    except Exception as exc:
        logger.warning("Failed to save decoder M to '%s': %s", short_path(cache_path), exc)


def save_decoder_provenance(output_dir: str, provenance: dict, logger) -> None:
    path = Path(build_decoder_provenance_path(output_dir))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Decoder source provenance saved to '%s'.", short_path(path))
    except Exception as exc:
        logger.warning("Failed to save decoder provenance to '%s': %s", short_path(path), exc)
