from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompletionStatus(str, Enum):
    """Minimal artifact lifecycle states for Phase 1B scaffolding."""

    PLANNED = "planned"
    MATERIALIZED = "materialized"
    COMMITTED = "committed"
    SUPERSEDED = "superseded"


class RetryDisposition(str, Enum):
    """How a replayed work unit interacts with existing artifacts."""

    NO_OP = "no-op"
    OVERWRITE = "overwrite"
    MERGE = "merge"


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable reference to a stage artifact."""

    stage: str
    kind: str
    key: str
    path: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class RetryIdempotencySemantics:
    """Typed retry/idempotency semantics for a stage work unit or artifact."""

    failure_unit: str
    retry_unit: str
    idempotency_key: str
    replay_disposition: RetryDisposition
    crash_recovery_rule: str


@dataclass(frozen=True)
class MergeInvariantSpec:
    """Close-by documentation for partial-result merge requirements."""

    identity: str
    associative: bool
    compatibility_checks: tuple[str, ...]
    deterministic_serialization_boundary: str
    duplicate_handling: str
    ordering: str


@dataclass(frozen=True)
class ArtifactSchemaSpec:
    """Durable schema contract for a stage-owned artifact manifest."""

    stage: str
    name: str
    schema_version: int
    required_artifact_kinds: tuple[str, ...]
    completeness_rule: str
    resume_rule: str


@dataclass(frozen=True)
class ArtifactManifestAssessment:
    """Materialized view of completeness and replay/resume readiness."""

    schema: ArtifactSchemaSpec
    artifact_key: str
    completion_status: CompletionStatus
    missing_artifact_kinds: tuple[str, ...]
    missing_artifact_paths: tuple[str, ...]
    all_required_artifacts_present: bool
    committed_state_consistent: bool
    is_complete: bool
    can_resume: bool
    detail: str


__all__ = [
    "ArtifactRef",
    "ArtifactManifestAssessment",
    "ArtifactSchemaSpec",
    "CompletionStatus",
    "MergeInvariantSpec",
    "RetryDisposition",
    "RetryIdempotencySemantics",
]
