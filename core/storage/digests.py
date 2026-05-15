from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def normalize_digest_input(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return normalize_digest_input(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): normalize_digest_input(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, tuple):
        return [normalize_digest_input(item) for item in value]
    if isinstance(value, list):
        return [normalize_digest_input(item) for item in value]
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        normalize_digest_input(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def digest_dict(payload: Mapping[str, Any], *, domain: str) -> str:
    encoded = canonical_json({"domain": domain, "payload": payload}).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256_hex(value: str, *, field_name: str = "digest") -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex digest.")
    return text


__all__ = [
    "canonical_json",
    "digest_dict",
    "normalize_digest_input",
    "require_sha256_hex",
]
