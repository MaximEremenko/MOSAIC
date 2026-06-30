from __future__ import annotations

import json
import os
import re
from dataclasses import is_dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence, TypeVar

from core.storage.atomic import (
    assert_path_contained,
    assert_relative_path_contained,
    atomic_create_no_overwrite,
    atomic_write_json,
    serialize_json_payload,
)


class ManifestError(ValueError):
    """Raised when a manifest is malformed or forward-incompatible."""


class ManifestCodec(Protocol):
    schema: ClassVar[str]
    schema_version: ClassVar[int]

    def to_payload(self) -> dict[str, Any]:
        ...

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]):
        ...


T = TypeVar("T")


_SCHEMA_FILE_BY_SCHEMA: dict[str, str] = {
    "mosaic.run_manifest": "run_manifest.schema.json",
    "mosaic.qspace_plan": "qspace_plan.schema.json",
    "mosaic.scattering.attempt": "scattering_attempt.schema.json",
    "mosaic.scattering.commit_candidate": "scattering_commit_candidate.schema.json",
    "mosaic.scattering.chunk_commit": "scattering_chunk_commit.schema.json",
    "mosaic.scattering.stage_plan": "scattering_stage_plan.schema.json",
    "mosaic.scattering.stage_commit": "scattering_stage_commit.schema.json",
    "mosaic.residual_field.attempt": "residual_attempt.schema.json",
    "mosaic.residual_field.commit_candidate": "residual_commit_candidate.schema.json",
    "mosaic.residual_field.chunk_commit": "residual_chunk_commit.schema.json",
    "mosaic.residual_field.stage_plan": "residual_stage_plan.schema.json",
    "mosaic.residual_field.stage_commit": "residual_stage_commit.schema.json",
    "mosaic.residual_field.no_output": "residual_no_output.schema.json",
    "mosaic.decoder.commit": "decoder_commit.schema.json",
    "mosaic.fs_capability": "fs_capability.schema.json",
    "mosaic.performance_metrics": "performance_metrics.schema.json",
    "mosaic.public_manifest": "public_manifest.schema.json",
}


def _windows_drive(path_text: str) -> bool:
    return len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha()


def validate_relative_manifest_path(path: str) -> str:
    path_text = os.fspath(path)
    if not isinstance(path_text, str) or not path_text:
        raise ManifestError("Manifest paths must be non-empty strings.")
    if "\\" in path_text:
        raise ManifestError(f"Manifest paths must use POSIX separators: {path_text!r}")
    if _windows_drive(path_text):
        raise ManifestError(f"Manifest path has a Windows drive root: {path_text!r}")
    candidate = Path(path_text)
    if candidate.is_absolute():
        raise ManifestError(f"Manifest path must be relative: {path_text!r}")
    parts = path_text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ManifestError(f"Manifest path contains an invalid segment: {path_text!r}")
    return "/".join(parts)


def _validate_embedded_paths(payload: Any, *, output_dir: Path, key: str | None = None) -> None:
    if isinstance(payload, Mapping):
        for child_key, child_value in payload.items():
            _validate_embedded_paths(child_value, output_dir=output_dir, key=str(child_key))
        return
    if isinstance(payload, list):
        for child in payload:
            _validate_embedded_paths(child, output_dir=output_dir, key=key)
        return
    if isinstance(payload, str) and key is not None and (key == "path" or key.endswith("_path")):
        relative = validate_relative_manifest_path(payload)
        assert_relative_path_contained(relative, output_dir=output_dir)


@lru_cache(maxsize=None)
def _load_schema_for_name(schema_name: str) -> Mapping[str, Any] | None:
    filename = _SCHEMA_FILE_BY_SCHEMA.get(schema_name)
    if filename is None:
        if schema_name.startswith("mosaic."):
            raise ManifestError(f"Unknown MOSAIC manifest schema {schema_name!r}.")
        return None
    schema_path = Path(__file__).with_name("schemas") / filename
    if not schema_path.exists():
        raise ManifestError(f"Missing manifest schema file for {schema_name!r}: {filename}")
    with schema_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, Mapping):
        raise ManifestError(f"Manifest schema file {filename!r} must contain a JSON object.")
    return loaded


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ManifestError(f"Unsupported manifest schema type {expected!r}.")


def _validate_schema_node(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if "anyOf" in schema:
        errors: list[str] = []
        variants = schema["anyOf"]
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
            raise ManifestError(f"Schema node {path} has invalid anyOf.")
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise ManifestError(f"Schema node {path} has non-object anyOf variant.")
            try:
                _validate_schema_node(value, variant, path=path)
                return
            except ManifestError as exc:
                errors.append(str(exc))
        raise ManifestError(f"{path} does not match any allowed schema variant: {errors[0]}")

    if "const" in schema and value != schema["const"]:
        raise ManifestError(f"{path} must be {schema['const']!r}.")
    if "enum" in schema:
        choices = schema["enum"]
        if value not in choices:
            raise ManifestError(f"{path} must be one of {choices!r}.")

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_type_matches(value, str(item)) for item in expected_types):
            raise ManifestError(f"{path} has invalid type.")

    if isinstance(value, str) and "pattern" in schema:
        pattern = str(schema["pattern"])
        if re.fullmatch(pattern, value) is None:
            raise ManifestError(f"{path} does not match required pattern.")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if properties is None:
            properties = {}
        if not isinstance(properties, Mapping):
            raise ManifestError(f"Schema node {path} has invalid properties.")
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ManifestError(f"Schema node {path} has invalid required list.")
        missing = [str(key) for key in required if str(key) not in value]
        if missing:
            raise ManifestError(f"{path} is missing required field(s): {', '.join(missing)}.")
        if schema.get("additionalProperties") is False:
            extras = sorted(str(key) for key in value if str(key) not in properties)
            if extras:
                raise ManifestError(f"{path} has unexpected field(s): {', '.join(extras)}.")
        for key, child_schema in properties.items():
            if key not in value:
                continue
            if not isinstance(child_schema, Mapping):
                raise ManifestError(f"Schema node {path}.{key} must be an object.")
            _validate_schema_node(value[key], child_schema, path=f"{path}.{key}")

    if isinstance(value, list) and "items" in schema:
        item_schema = schema["items"]
        if not isinstance(item_schema, Mapping):
            raise ManifestError(f"Schema node {path}.items must be an object.")
        for index, item in enumerate(value):
            _validate_schema_node(item, item_schema, path=f"{path}[{index}]")


def _validate_against_schema_file(payload: Mapping[str, Any], *, schema_name: str) -> None:
    schema = _load_schema_for_name(schema_name)
    if schema is None:
        return
    _validate_schema_node(dict(payload), schema, path="$")


def validate_manifest_payload(
    payload: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
    expected_version: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ManifestError("Manifest payload must be a JSON object.")
    normalized = dict(payload)
    schema = normalized.get("schema")
    schema_version = normalized.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ManifestError("Manifest field 'schema' must be a non-empty string.")
    if not isinstance(schema_version, int):
        raise ManifestError("Manifest field 'schema_version' must be an integer.")
    if expected_schema is not None and schema != expected_schema:
        raise ManifestError(f"Expected schema {expected_schema!r}, got {schema!r}.")
    if expected_version is not None and schema_version > int(expected_version):
        raise ManifestError(
            f"Manifest schema_version {schema_version} is newer than supported {expected_version}."
        )
    _validate_against_schema_file(normalized, schema_name=schema)
    if output_dir is not None:
        _validate_embedded_paths(normalized, output_dir=Path(output_dir).resolve())
    return normalized


def _payload_from_manifest(manifest: Any) -> dict[str, Any]:
    if hasattr(manifest, "to_payload"):
        return dict(manifest.to_payload())
    if is_dataclass(manifest):
        return {field.name: getattr(manifest, field.name) for field in fields(manifest)}
    if isinstance(manifest, Mapping):
        return dict(manifest)
    raise TypeError("Manifest must be a mapping, dataclass, or expose to_payload().")


def read_manifest(path: str | Path, *, codec: type[T], output_dir: str | Path) -> T:
    manifest_path = assert_path_contained(path, output_dir=output_dir)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized = validate_manifest_payload(
        payload,
        expected_schema=getattr(codec, "schema", None),
        expected_version=getattr(codec, "schema_version", None),
        output_dir=output_dir,
    )
    return codec.from_payload(normalized)


def write_manifest(path: str | Path, manifest: Any, *, output_dir: str | Path) -> None:
    payload = _payload_from_manifest(manifest)
    expected_schema = getattr(manifest, "schema", None)
    expected_version = getattr(manifest, "schema_version", None)

    def _validate_manifest(payload_to_validate: Mapping[str, Any]) -> None:
        validate_manifest_payload(
            payload_to_validate,
            expected_schema=expected_schema,
            expected_version=expected_version,
            output_dir=output_dir,
        )

    atomic_write_json(path, payload, output_dir=output_dir, validator=_validate_manifest)


def try_commit_manifest(
    path: str | Path,
    manifest: Any,
    *,
    codec: type[T],
    output_dir: str | Path,
    conflict_error: Callable[[T], Exception] | None = None,
) -> tuple[T, bool]:
    """First-writer-wins commit of ``manifest`` to ``path`` (no overwrite, no TOCTOU).

    Serializes ``manifest`` to EXACTLY the same bytes ``write_manifest`` would
    write (both route through ``serialize_json_payload``) and validates the
    payload identically, then commits via ``atomic_create_no_overwrite``:

    * If this caller created the file, returns ``(manifest, True)``.
    * Otherwise the file already exists; the committed manifest is read back via
      ``read_manifest``. If it equals ``manifest`` (the idempotent / concurrent
      identical-writer case), returns ``(existing, False)``. If it differs,
      raises — by default a ``RuntimeError`` describing the conflict, or the
      exception built by ``conflict_error(existing)`` when the caller supplies one
      (so each promote path can preserve its existing error wording).

    This replaces the ``if target.exists(): match-or-raise else write_manifest``
    pattern, which had both an atomic-overwrite (last-writer-wins) write and a
    check-then-write TOCTOU window.
    """
    payload = _payload_from_manifest(manifest)
    expected_schema = getattr(manifest, "schema", None)
    expected_version = getattr(manifest, "schema_version", None)
    validate_manifest_payload(
        payload,
        expected_schema=expected_schema,
        expected_version=expected_version,
        output_dir=output_dir,
    )
    target = assert_path_contained(path, output_dir=output_dir)
    data = serialize_json_payload(payload)
    created = atomic_create_no_overwrite(target, data)
    if created:
        return manifest, True
    existing = read_manifest(target, codec=codec, output_dir=output_dir)
    if existing == manifest:
        return existing, False
    if conflict_error is not None:
        raise conflict_error(existing)
    raise RuntimeError(
        f"Manifest at {target!s} already committed with different content."
    )


__all__ = [
    "ManifestCodec",
    "ManifestError",
    "read_manifest",
    "try_commit_manifest",
    "validate_manifest_payload",
    "validate_relative_manifest_path",
    "write_manifest",
]
