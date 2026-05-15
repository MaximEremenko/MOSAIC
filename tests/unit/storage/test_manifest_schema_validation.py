from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping

import pytest

from core.storage.manifest import ManifestError, read_manifest, validate_manifest_payload, write_manifest


HEX_A = "a" * 64
HEX_B = "b" * 64


@dataclass(frozen=True)
class RunManifest:
    schema: ClassVar[str] = "mosaic.run_manifest"
    schema_version: ClassVar[int] = 1
    scientific_digest: str = HEX_A
    execution_digest: str = HEX_B
    run_digest: str = "run123"

    def to_payload(self):
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "scientific_digest": self.scientific_digest,
            "execution_digest": self.execution_digest,
            "run_digest": self.run_digest,
            "execution_contract": {},
        }

    @classmethod
    def from_payload(cls, payload: Mapping):
        return cls(
            scientific_digest=str(payload["scientific_digest"]),
            execution_digest=str(payload["execution_digest"]),
            run_digest=str(payload["run_digest"]),
        )


def _run_payload(**overrides):
    payload = RunManifest().to_payload()
    payload.update(overrides)
    return payload


def _path_payload(path: str):
    return {
        "schema": "example.path_manifest",
        "schema_version": 1,
        "path": path,
    }


def test_manifest_read_write_roundtrip_validates_schema_and_paths(tmp_path):
    target = tmp_path / "manifest.json"

    write_manifest(target, RunManifest(), output_dir=tmp_path)
    loaded = read_manifest(target, codec=RunManifest, output_dir=tmp_path)

    assert loaded == RunManifest()


def test_manifest_rejects_forward_incompatible_version():
    with pytest.raises(ManifestError, match="newer"):
        validate_manifest_payload(
            {"schema": "example.manifest", "schema_version": 2},
            expected_schema="example.manifest",
            expected_version=1,
        )


def test_manifest_rejects_known_schema_forward_version_without_expected_version():
    with pytest.raises(ManifestError):
        validate_manifest_payload(_run_payload(schema_version=999))


def test_manifest_rejects_unknown_mosaic_schema():
    with pytest.raises(ManifestError, match="Unknown MOSAIC"):
        validate_manifest_payload(
            {"schema": "mosaic.unknown_manifest", "schema_version": 1}
        )


def test_manifest_rejects_non_hex_digest_characters():
    with pytest.raises(ManifestError, match="pattern"):
        validate_manifest_payload(_run_payload(scientific_digest="g" * 64))


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/payload.hdf5",
        "../payload.hdf5",
        "a//payload.hdf5",
        "a/./payload.hdf5",
        "C:/payload.hdf5",
        r"a\payload.hdf5",
    ],
)
def test_manifest_rejects_invalid_embedded_paths(tmp_path, bad_path):
    with pytest.raises(ManifestError):
        validate_manifest_payload(
            _path_payload(bad_path),
            expected_schema="example.path_manifest",
            expected_version=1,
            output_dir=tmp_path,
        )
