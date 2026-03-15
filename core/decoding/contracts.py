from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecodingRequest:
    output_dir: str
    parameters: dict[str, Any]
    client: Any


@dataclass(frozen=True)
class DisplacementDecoderSourcePolicy:
    mode: str = "error"
    assignment: str = "single"
    cache_path: str | None = None
    compute_output_directory: str | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any] | None,
    ) -> "DisplacementDecoderSourcePolicy":
        mapping = dict(payload or {})
        mode = str(
            mapping.get("source")
            or mapping.get("mode")
            or "error"
        ).strip().lower()
        assignment = str(
            mapping.get("assignment")
            or mapping.get("decoder_assignment")
            or "single"
        ).strip().lower()
        if mode not in {"error", "cache", "compute"}:
            raise ValueError(
                "processing.decoder.source must be one of: error, cache, compute."
            )
        if assignment not in {"single", "family"}:
            raise ValueError(
                "processing.decoder.assignment must be one of: single, family."
            )
        cache_path = mapping.get("cache_path") or mapping.get("path")
        compute_output_directory = (
            mapping.get("compute_output_directory")
            or mapping.get("output_directory")
            or mapping.get("working_directory")
        )
        if mode == "cache" and not cache_path:
            raise ValueError(
                "processing.decoder.cache_path is required when processing.decoder.source='cache'."
            )
        if mode == "compute" and not compute_output_directory:
            raise ValueError(
                "processing.decoder.compute_output_directory is required when processing.decoder.source='compute'."
            )
        return cls(
            mode=mode,
            assignment=assignment,
            cache_path=str(cache_path) if cache_path else None,
            compute_output_directory=(
                str(compute_output_directory) if compute_output_directory else None
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {"source": self.mode, "assignment": self.assignment}
        if self.cache_path is not None:
            payload["cache_path"] = self.cache_path
        if self.compute_output_directory is not None:
            payload["compute_output_directory"] = self.compute_output_directory
        return payload


@dataclass(frozen=True)
class DecoderSourceProvenance:
    mode: str
    semantics: str
    decoder_cache_path: str
    source_output_directory: str | None = None
    compute_output_directory: str | None = None
    feature_dim: int | None = None
    loaded_from_cache: bool = False
    computed: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "DecodingRequest",
    "DecoderSourceProvenance",
    "DisplacementDecoderSourcePolicy",
]
