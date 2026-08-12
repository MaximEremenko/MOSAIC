from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import pytest

from core.storage.manifest import try_commit_manifest, write_manifest


@dataclass(frozen=True)
class _ToyManifest:
    # A non-"mosaic." schema name skips the JSON-schema-file validation while
    # still exercising the basic schema/schema_version checks.
    schema: ClassVar[str] = "toy.manifest"
    schema_version: ClassVar[int] = 1

    name: str
    value: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "name": self.name,
            "value": self.value,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "_ToyManifest":
        return cls(name=str(payload["name"]), value=int(payload["value"]))


def test_try_commit_serialized_bytes_match_write_manifest(tmp_path):
    """The CAS commit must produce byte-identical files to write_manifest."""
    manifest = _ToyManifest(name="alpha", value=7)

    via_write = tmp_path / "via_write.json"
    write_manifest(via_write, manifest, output_dir=tmp_path)

    via_cas = tmp_path / "via_cas.json"
    committed, created = try_commit_manifest(
        via_cas, manifest, codec=_ToyManifest, output_dir=tmp_path
    )

    assert created is True
    assert committed == manifest
    assert via_cas.read_bytes() == via_write.read_bytes()


def test_try_commit_is_idempotent(tmp_path):
    manifest = _ToyManifest(name="beta", value=3)
    target = tmp_path / "commit.json"

    first, created_first = try_commit_manifest(
        target, manifest, codec=_ToyManifest, output_dir=tmp_path
    )
    bytes_after_first = target.read_bytes()

    second, created_second = try_commit_manifest(
        target, manifest, codec=_ToyManifest, output_dir=tmp_path
    )

    assert created_first is True
    assert created_second is False
    assert first == second == manifest
    # Idempotent skip leaves the committed bytes untouched.
    assert target.read_bytes() == bytes_after_first


def test_try_commit_conflict_raises_and_keeps_first_bytes(tmp_path):
    first = _ToyManifest(name="gamma", value=1)
    different = _ToyManifest(name="gamma", value=2)
    target = tmp_path / "commit.json"

    try_commit_manifest(target, first, codec=_ToyManifest, output_dir=tmp_path)
    first_bytes = target.read_bytes()

    with pytest.raises(RuntimeError):
        try_commit_manifest(
            target, different, codec=_ToyManifest, output_dir=tmp_path
        )

    # The first writer's committed bytes survive the conflicting attempt.
    assert target.read_bytes() == first_bytes


def test_try_commit_uses_custom_conflict_error(tmp_path):
    first = _ToyManifest(name="delta", value=1)
    different = _ToyManifest(name="delta", value=9)
    target = tmp_path / "commit.json"

    try_commit_manifest(target, first, codec=_ToyManifest, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="custom conflict wording"):
        try_commit_manifest(
            target,
            different,
            codec=_ToyManifest,
            output_dir=tmp_path,
            conflict_error=lambda _existing: RuntimeError("custom conflict wording"),
        )
