from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class CompletionStatus(str, Enum):
    """Minimal artifact lifecycle states for staged artifacts."""

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


@dataclass(frozen=True)
class ScatteringHandoff:
    """Typed scattering -> residual-field inter-stage handoff.

    Carries the handful of stringly-typed keys that flow from
    :meth:`core.scattering.stage.ScatteringStage.execute` into
    :meth:`core.residual_field.stage.ResidualFieldStage.execute`; those keys are
    the fields below.

    A mapping bridge is provided for callers that exchange a plain dict:
    :meth:`from_mapping` tolerates a plain ``dict`` (or ``None``) via ``.get()``
    defaults, and :meth:`to_mapping` reproduces the read-relevant key shape so a
    caller consuming a mapping keeps working.
    """

    scattering_run_digest: str | None = None
    source_scattering_commit_digest: str | None = None
    residual_parameter_digest: str | None = None
    # Alias the residual stage falls back to when ``scattering_run_digest``
    # is absent (residual_field/stage.py read of ``run_digest``).
    run_digest: str | None = None
    # ``True`` when the originating handoff carried no payload at all (an empty
    # or ``None`` source mapping).
    is_empty: bool = field(default=False)

    def to_mapping(self) -> dict[str, object]:
        """Reproduce the read-relevant dict shape for mapping consumers.

        Only keys that are actually present in the source are emitted, so a
        round-trip through :meth:`from_mapping` preserves key presence (and the
        ``is_empty`` flag) for the residual stage's membership checks.
        """
        if self.is_empty:
            return {}
        payload: dict[str, object] = {}
        if self.scattering_run_digest is not None:
            payload["scattering_run_digest"] = self.scattering_run_digest
        if self.run_digest is not None:
            payload["run_digest"] = self.run_digest
        if self.source_scattering_commit_digest is not None:
            payload["source_scattering_commit_digest"] = (
                self.source_scattering_commit_digest
            )
        if self.residual_parameter_digest is not None:
            payload["residual_parameter_digest"] = self.residual_parameter_digest
        return payload

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object] | None
    ) -> "ScatteringHandoff":
        """Build a handoff from a mapping, tolerant of missing keys.

        Absent keys become ``None``, matching ``.get()`` default semantics.
        An empty or ``None`` payload yields ``is_empty=True``. Unknown keys
        (including ones written by older codebase generations) are ignored.
        """
        if payload is None or not payload:
            return cls(is_empty=True)

        def _opt_str(key: str) -> str | None:
            value = payload.get(key)
            return None if value is None else str(value)

        return cls(
            scattering_run_digest=_opt_str("scattering_run_digest"),
            source_scattering_commit_digest=_opt_str("source_scattering_commit_digest"),
            residual_parameter_digest=_opt_str("residual_parameter_digest"),
            run_digest=_opt_str("run_digest"),
            is_empty=False,
        )


__all__ = [
    "ArtifactRef",
    "ArtifactManifestAssessment",
    "ArtifactSchemaSpec",
    "CompletionStatus",
    "MergeInvariantSpec",
    "RetryDisposition",
    "RetryIdempotencySemantics",
    "ScatteringHandoff",
]
