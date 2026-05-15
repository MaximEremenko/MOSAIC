from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from core.storage.digests import digest_dict, normalize_digest_input
from core.storage.manifest import ManifestError, read_manifest, write_manifest


PUBLIC_MANIFEST_SCHEMA = "mosaic.public_manifest"
PUBLIC_MANIFEST_SCHEMA_VERSION = 1


def build_public_manifest_digest(
    *,
    run_digest: str,
    source_stage: str,
    source_identity: Mapping[str, Any],
    published_files: Sequence[str] = (),
    published_file_records: Sequence[Mapping[str, Any]] = (),
    previous_run_digest: str | None = None,
    previous_public_manifest_sha256: str | None = None,
    previous_public_manifest_path: str | None = None,
) -> str:
    return digest_dict(
        {
            "schema_version": PUBLIC_MANIFEST_SCHEMA_VERSION,
            "run_digest": str(run_digest),
            "source_stage": str(source_stage),
            "source_identity": normalize_digest_input(dict(source_identity)),
            "published_files": [str(item) for item in published_files],
            "published_file_records": [
                normalize_digest_input(dict(item)) for item in published_file_records
            ],
            "previous_run_digest": previous_run_digest,
            "previous_public_manifest_sha256": previous_public_manifest_sha256,
            "previous_public_manifest_path": previous_public_manifest_path,
        },
        domain="mosaic.public_manifest.v1",
    )


@dataclass(frozen=True)
class PublicManifest:
    run_digest: str
    source_stage: str
    source_identity: dict[str, Any]
    published_files: tuple[str, ...]
    published_file_records: tuple[dict[str, Any], ...]
    public_manifest_digest: str
    previous_run_digest: str | None = None
    previous_public_manifest_sha256: str | None = None
    previous_public_manifest_path: str | None = None

    schema: ClassVar[str] = PUBLIC_MANIFEST_SCHEMA
    schema_version: ClassVar[int] = PUBLIC_MANIFEST_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "source_stage": self.source_stage,
            "source_identity": normalize_digest_input(dict(self.source_identity)),
            "published_files": list(self.published_files),
            "published_file_records": [
                normalize_digest_input(dict(item)) for item in self.published_file_records
            ],
            "public_manifest_digest": self.public_manifest_digest,
            "previous_run_digest": self.previous_run_digest,
            "previous_public_manifest_sha256": self.previous_public_manifest_sha256,
            "previous_public_manifest_path": self.previous_public_manifest_path,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PublicManifest":
        published_files = tuple(str(item) for item in payload.get("published_files", ()))
        published_file_records = tuple(
            dict(item) for item in payload.get("published_file_records", ())
        )
        record_paths = tuple(str(item.get("path", "")) for item in published_file_records)
        if record_paths != published_files:
            raise ManifestError(
                "public_manifest.json published_files must exactly match published_file_records paths."
            )
        if len(set(record_paths)) != len(record_paths):
            raise ManifestError("public_manifest.json contains duplicate published file paths.")
        manifest = cls(
            run_digest=str(payload["run_digest"]),
            source_stage=str(payload["source_stage"]),
            source_identity=dict(payload["source_identity"]),
            published_files=published_files,
            published_file_records=published_file_records,
            public_manifest_digest=str(payload["public_manifest_digest"]),
            previous_run_digest=(
                None
                if payload.get("previous_run_digest") is None
                else str(payload["previous_run_digest"])
            ),
            previous_public_manifest_sha256=(
                None
                if payload.get("previous_public_manifest_sha256") is None
                else str(payload["previous_public_manifest_sha256"])
            ),
            previous_public_manifest_path=(
                None
                if payload.get("previous_public_manifest_path") is None
                else str(payload["previous_public_manifest_path"])
            ),
        )
        expected = build_public_manifest_digest(
            run_digest=manifest.run_digest,
            source_stage=manifest.source_stage,
            source_identity=manifest.source_identity,
            published_files=manifest.published_files,
            published_file_records=manifest.published_file_records,
            previous_run_digest=manifest.previous_run_digest,
            previous_public_manifest_sha256=manifest.previous_public_manifest_sha256,
            previous_public_manifest_path=manifest.previous_public_manifest_path,
        )
        if manifest.public_manifest_digest != expected:
            raise ManifestError("public_manifest.json digest does not match its payload.")
        return manifest


def write_public_manifest(
    path: str | Path,
    *,
    output_dir: str | Path,
    run_digest: str,
    source_stage: str,
    source_identity: Mapping[str, Any],
    published_files: Sequence[str] = (),
    published_file_records: Sequence[Mapping[str, Any]] = (),
    previous_run_digest: str | None = None,
    previous_public_manifest_sha256: str | None = None,
    previous_public_manifest_path: str | None = None,
) -> PublicManifest:
    records = tuple(dict(item) for item in published_file_records)
    try:
        record_paths = tuple(str(item["path"]) for item in records)
    except KeyError as exc:
        raise ManifestError("published_file_records entries must include path.") from exc
    normalized_published_files = tuple(str(item) for item in published_files)
    if record_paths != normalized_published_files:
        raise ManifestError(
            "published_files must exactly match published_file_records paths."
        )
    manifest = PublicManifest(
        run_digest=str(run_digest),
        source_stage=str(source_stage),
        source_identity=dict(source_identity),
        published_files=normalized_published_files,
        published_file_records=records,
        public_manifest_digest=build_public_manifest_digest(
            run_digest=str(run_digest),
            source_stage=str(source_stage),
            source_identity=source_identity,
            published_files=published_files,
            published_file_records=records,
            previous_run_digest=previous_run_digest,
            previous_public_manifest_sha256=previous_public_manifest_sha256,
            previous_public_manifest_path=previous_public_manifest_path,
        ),
        previous_run_digest=previous_run_digest,
        previous_public_manifest_sha256=previous_public_manifest_sha256,
        previous_public_manifest_path=previous_public_manifest_path,
    )
    write_manifest(path, manifest, output_dir=output_dir)
    return manifest


def read_public_manifest(
    path: str | Path,
    *,
    output_dir: str | Path,
) -> PublicManifest:
    return read_manifest(path, codec=PublicManifest, output_dir=output_dir)


__all__ = [
    "PUBLIC_MANIFEST_SCHEMA",
    "PublicManifest",
    "build_public_manifest_digest",
    "read_public_manifest",
    "write_public_manifest",
]
