from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from core.storage.atomic import assert_path_contained
from core.storage.attempt_store import stage_root
from core.storage.digests import digest_dict
from core.storage.fingerprint import file_sha256
from core.storage.manifest import read_manifest, write_manifest


DECODER_COMMIT_SCHEMA = "mosaic.decoder.commit"
DECODER_COMMIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DecoderCommitManifest:
    run_digest: str
    decoder_cache_path: str
    decoder_cache_file_sha256: str
    decoder_cache_nbytes: int
    decoder_cache_identity: dict[str, Any]
    decoder_commit_digest: str

    schema: ClassVar[str] = DECODER_COMMIT_SCHEMA
    schema_version: ClassVar[int] = DECODER_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "decoder_cache_path": self.decoder_cache_path,
            "decoder_cache_file_sha256": self.decoder_cache_file_sha256,
            "decoder_cache_nbytes": int(self.decoder_cache_nbytes),
            "decoder_cache_identity": dict(self.decoder_cache_identity),
            "decoder_commit_digest": self.decoder_commit_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DecoderCommitManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            decoder_cache_path=str(payload["decoder_cache_path"]),
            decoder_cache_file_sha256=str(payload["decoder_cache_file_sha256"]),
            decoder_cache_nbytes=int(payload["decoder_cache_nbytes"]),
            decoder_cache_identity=dict(payload["decoder_cache_identity"]),
            decoder_commit_digest=str(payload["decoder_commit_digest"]),
        )


def decoder_commit_path(output_dir: str | Path, run_digest: str) -> Path:
    return stage_root(output_dir, run_digest, "decoding") / "decoder_commit.json"


def write_decoder_commit(
    *,
    output_dir: str | Path,
    run_digest: str,
    decoder_cache_path: str | Path,
    decoder_cache_identity: Mapping[str, Any],
) -> DecoderCommitManifest:
    cache_path = assert_path_contained(decoder_cache_path, output_dir=output_dir)
    relative_cache_path = cache_path.relative_to(Path(output_dir).resolve()).as_posix()
    file_digest = file_sha256(cache_path)
    cache_nbytes = int(cache_path.stat().st_size)
    commit_digest = digest_dict(
        {
            "schema_version": DECODER_COMMIT_SCHEMA_VERSION,
            "run_digest": str(run_digest),
            "decoder_cache_path": relative_cache_path,
            "decoder_cache_file_sha256": file_digest,
            "decoder_cache_nbytes": cache_nbytes,
            "decoder_cache_identity": dict(decoder_cache_identity),
        },
        domain="mosaic.decoder.commit.v1",
    )
    manifest = DecoderCommitManifest(
        run_digest=str(run_digest),
        decoder_cache_path=relative_cache_path,
        decoder_cache_file_sha256=file_digest,
        decoder_cache_nbytes=cache_nbytes,
        decoder_cache_identity=dict(decoder_cache_identity),
        decoder_commit_digest=commit_digest,
    )
    target = decoder_commit_path(output_dir, run_digest)
    if target.exists():
        existing = read_manifest(target, codec=DecoderCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError("decoder_commit.json already exists with different identity.")
        return existing
    write_manifest(target, manifest, output_dir=output_dir)
    return manifest


__all__ = [
    "DECODER_COMMIT_SCHEMA",
    "DecoderCommitManifest",
    "decoder_commit_path",
    "write_decoder_commit",
]
