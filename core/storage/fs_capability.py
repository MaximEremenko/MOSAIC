from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from core.storage.attempt_store import fs_capability_path
from core.storage.digests import digest_dict, normalize_digest_input
from core.storage.manifest import ManifestError, read_manifest, write_manifest


FS_CAPABILITY_SCHEMA = "mosaic.fs_capability"
FS_CAPABILITY_SCHEMA_VERSION = 1


def build_fs_capability_digest(
    *,
    run_digest: str,
    capabilities: Mapping[str, Any],
) -> str:
    return digest_dict(
        {
            "schema_version": FS_CAPABILITY_SCHEMA_VERSION,
            "run_digest": str(run_digest),
            "capabilities": normalize_digest_input(dict(capabilities)),
        },
        domain="mosaic.fs_capability.v1",
    )


@dataclass(frozen=True)
class FSCapabilityManifest:
    run_digest: str
    capabilities: dict[str, Any]
    capability_digest: str

    schema: ClassVar[str] = FS_CAPABILITY_SCHEMA
    schema_version: ClassVar[int] = FS_CAPABILITY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "capabilities": normalize_digest_input(dict(self.capabilities)),
            "capability_digest": self.capability_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FSCapabilityManifest":
        manifest = cls(
            run_digest=str(payload["run_digest"]),
            capabilities=dict(payload["capabilities"]),
            capability_digest=str(payload["capability_digest"]),
        )
        expected = build_fs_capability_digest(
            run_digest=manifest.run_digest,
            capabilities=manifest.capabilities,
        )
        if manifest.capability_digest != expected:
            raise ManifestError("fs_capability.json digest does not match its payload.")
        return manifest


def write_fs_capability_manifest(
    *,
    output_dir: str | Path,
    run_digest: str,
    capabilities: Mapping[str, Any],
) -> FSCapabilityManifest:
    manifest = FSCapabilityManifest(
        run_digest=str(run_digest),
        capabilities=dict(capabilities),
        capability_digest=build_fs_capability_digest(
            run_digest=str(run_digest),
            capabilities=capabilities,
        ),
    )
    write_manifest(
        fs_capability_path(output_dir, run_digest),
        manifest,
        output_dir=output_dir,
    )
    return manifest


def read_fs_capability_manifest(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> FSCapabilityManifest:
    return read_manifest(
        fs_capability_path(output_dir, run_digest),
        codec=FSCapabilityManifest,
        output_dir=output_dir,
    )


__all__ = [
    "FS_CAPABILITY_SCHEMA",
    "FSCapabilityManifest",
    "build_fs_capability_digest",
    "read_fs_capability_manifest",
    "write_fs_capability_manifest",
]
