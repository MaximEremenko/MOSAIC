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


def build_execution_digest(
    *,
    scientific_digest: str,
    backend: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    backend_policy_digest: str | None = None,
    schema_version: int,
    domain: str,
) -> str:
    """Generic execution-identity algebra shared by every Map-Reduce stage.

    ``execution_digest`` is ``scientific_digest`` plus the numerical execution
    contract (backend, epsilon, dtype, pre-sum/reducer strategy, backend
    policy). The scientific-identity *content* stays domain-specific in each
    stage's planning module; only this stage-agnostic combination lives here.
    """
    require_sha256_hex(scientific_digest, field_name="scientific_digest")
    payload = {
        "schema_version": schema_version,
        "scientific_digest": scientific_digest,
        "backend": str(backend),
        "eps": float(eps),
        "dtype": str(dtype),
        "pre_sum_mode": str(pre_sum_mode),
        "reducer_strategy": str(reducer_strategy),
        "backend_policy_digest": backend_policy_digest,
    }
    return digest_dict(payload, domain=domain)


def build_run_digest(
    execution_digest: str,
    *,
    schema_version: int,
    domain: str = "mosaic.run.v1",
    length: int = 32,
) -> str:
    """Derive the path-layout run digest from an ``execution_digest``.

    NOTE: ``execution_digest`` is device-bound (it includes ``backend``),
    so this run digest separates CPU and GPU runs into different trees. The current
    identity model re-anchors the run/checkpoint identity onto the device-independent
    :func:`build_run_identity_digest`; this function is retained for callers and
    tests that still address the device-bound layout during the migration.
    """
    require_sha256_hex(execution_digest, field_name="execution_digest")
    return digest_dict(
        {"schema_version": schema_version, "execution_digest": execution_digest},
        domain=domain,
    )[:length]


def build_run_identity_digest(
    *,
    scientific_digest: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    schema_version: int,
    domain: str = "mosaic.run_identity.v1",
    length: int = 32,
) -> str:
    """Device-INDEPENDENT run / checkpoint identity .

    Unlike :func:`build_execution_digest`, this deliberately EXCLUDES the
    ``backend`` (cpu/cuda) and any device-bound policy. Under a durable
    checkpoint is addressed by the *science* plus the *numerical contract*
    (eps/dtype/pre-sum/reducer) — never by the device that computed it — so a CPU
    run and a GPU run of the same science share the same run tree and the same
    checkpoint addresses. Cross-device promotion is then decided by
    scientific-invariant validation, not by output-byte equality (which is
    impossible across CPU/GPU and even across GPU launches). The realized backend
    is recorded as *attempt metadata* (see ``NufftExecutionSettings.backend``),
    never as identity.

    eps/dtype/pre-sum/reducer ARE part of the identity: two runs that disagree on
    the numerical contract are different checkpoints (and need not agree).
    """
    require_sha256_hex(scientific_digest, field_name="scientific_digest")
    payload = {
        "schema_version": schema_version,
        "scientific_digest": scientific_digest,
        "eps": float(eps),
        "dtype": str(dtype),
        "pre_sum_mode": str(pre_sum_mode),
        "reducer_strategy": str(reducer_strategy),
    }
    return digest_dict(payload, domain=domain)[:length]


__all__ = [
    "build_execution_digest",
    "build_run_digest",
    "build_run_identity_digest",
    "canonical_json",
    "digest_dict",
    "normalize_digest_input",
    "require_sha256_hex",
]
