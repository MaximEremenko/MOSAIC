from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from core.storage.attempt_store import (
    attempt_manifest_path,
    attempt_payload_path,
    chunk_commit_path,
    chunk_root,
    commit_attempts_root,
    commit_candidate_manifest_path,
    commit_candidate_payload_path,
    relative_to_output,
    stage_commit_path,
    stage_plan_path,
    stage_root,
)
from core.storage.agreement import (
    DEFAULT_NUFFT_EPS,
    cancellation_kappa,
    predict_agreement_rtol,
    relative_l2,
)
from core.storage.commit_payloads import (
    _payload_datasets,
    _payload_digest,
    _read_payload,
    _resolve_runtime_provenance,
)
from core.storage.digests import digest_dict
from core.storage.fingerprint import file_sha256
from core.storage.hdf5_atomic import atomic_hdf5_write
from core.storage.manifest import read_manifest, try_commit_manifest, write_manifest
from core.storage.performance import write_performance_metrics
from core.storage.work_identity import assert_device_independent


RESIDUAL_FIELD_ATTEMPT_SCHEMA = "mosaic.residual_field.attempt"
RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA = "mosaic.residual_field.commit_candidate"
RESIDUAL_FIELD_CHUNK_COMMIT_SCHEMA = "mosaic.residual_field.chunk_commit"
RESIDUAL_FIELD_STAGE_PLAN_SCHEMA = "mosaic.residual_field.stage_plan"
RESIDUAL_FIELD_STAGE_COMMIT_SCHEMA = "mosaic.residual_field.stage_commit"
RESIDUAL_FIELD_NO_OUTPUT_SCHEMA = "mosaic.residual_field.no_output"
RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION = 1
RESIDUAL_FIELD_STAGE = "residual_field"


def build_residual_work_unit_digest(
    *,
    run_digest: str,
    chunk_id: int,
    partition_id: int,
    point_start: int,
    point_stop: int,
    interval_ids: Sequence[int],
    parameter_digest: str,
    partition_plan_digest: str,
    source_scattering_commit_digest: str,
    source_replacement_digest: str | None,
    expected_output_digest: str,
) -> str:
    """Device-independent residual checkpoint/work address.

    Mirrors the scattering work-unit identity rule: the address is the run + structural partition
    keys (chunk, partition, point range, interval ids), the parameter / partition-
    plan digests, the upstream source-commit identities, and the *structural*
    ``expected_output_digest`` (the planned output SHAPE -- counts/intervals/params --
    NOT output bytes). It deliberately EXCLUDES the device-bound
    ``backend_policy_digest`` (which carries ``backend_kind``) so a CPU- and a
    GPU-computed partition for the same science map to the SAME address and promote
    to one checkpoint. The realized backend policy stays on the attempt manifest as
    METADATA. ``run_digest`` is already device-independent (it propagates the
    scattering run identity). Domain bumped to ``v2`` (no released data to migrate).
    """
    identity_payload = {
        "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
        "stage": RESIDUAL_FIELD_STAGE,
        "run_digest": str(run_digest),
        "chunk_id": int(chunk_id),
        "partition_id": int(partition_id),
        "point_start": int(point_start),
        "point_stop": int(point_stop),
        "interval_ids": [int(item) for item in sorted(interval_ids)],
        "parameter_digest": str(parameter_digest),
        "partition_plan_digest": str(partition_plan_digest),
        "source_scattering_commit_digest": str(source_scattering_commit_digest),
        "source_replacement_digest": (
            None if source_replacement_digest is None else str(source_replacement_digest)
        ),
        "expected_output_digest": str(expected_output_digest),
    }
    # Device-independence tripwire: pin the device-INDEPENDENCE invariant immediately before
    # hashing. This payload already excludes backend_policy_digest, so this is a
    # no-op today -- it fails loudly only if a future edit folds a device/runtime
    # field into the residual work-unit identity.
    assert_device_independent(
        identity_payload, context="residual work-unit digest"
    )
    return digest_dict(
        identity_payload,
        domain="mosaic.residual_field.work_unit.v2",
    )



def _payload_attrs(
    *,
    schema: str,
    run_digest: str,
    work_unit_digest: str,
    attempt_id: str | None,
    chunk_id: int,
    partition_id: int,
    point_start: int,
    point_stop: int,
    reciprocal_point_count: int,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "schema": schema,
        "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
        "run_digest": str(run_digest),
        "stage": RESIDUAL_FIELD_STAGE,
        "work_unit_digest": str(work_unit_digest),
        "chunk_id": int(chunk_id),
        "partition_id": int(partition_id),
        "point_start": int(point_start),
        "point_stop": int(point_stop),
        "reciprocal_point_count": int(reciprocal_point_count),
    }
    if attempt_id is not None:
        attrs["attempt_id"] = str(attempt_id)
    return attrs



@dataclass(frozen=True)
class ResidualAttemptManifest:
    run_digest: str
    work_unit_digest: str
    attempt_id: str
    chunk_id: int
    partition_id: int
    point_start: int
    point_stop: int
    interval_ids: tuple[int, ...]
    parameter_digest: str
    partition_plan_digest: str
    source_scattering_commit_digest: str
    source_replacement_digest: str | None
    backend_policy_digest: str
    expected_output_digest: str
    contribution_reciprocal_points: int
    runtime_provenance: dict[str, Any]
    payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int

    schema: ClassVar[str] = RESIDUAL_FIELD_ATTEMPT_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "work_unit_digest": self.work_unit_digest,
            "attempt_id": self.attempt_id,
            "chunk_id": int(self.chunk_id),
            "partition_id": int(self.partition_id),
            "point_start": int(self.point_start),
            "point_stop": int(self.point_stop),
            "interval_ids": [int(item) for item in self.interval_ids],
            "parameter_digest": self.parameter_digest,
            "partition_plan_digest": self.partition_plan_digest,
            "source_scattering_commit_digest": self.source_scattering_commit_digest,
            "source_replacement_digest": self.source_replacement_digest,
            "backend_policy_digest": self.backend_policy_digest,
            "expected_output_digest": self.expected_output_digest,
            "contribution_reciprocal_points": int(self.contribution_reciprocal_points),
            "runtime_provenance": dict(self.runtime_provenance),
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualAttemptManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            work_unit_digest=str(payload["work_unit_digest"]),
            attempt_id=str(payload["attempt_id"]),
            chunk_id=int(payload["chunk_id"]),
            partition_id=int(payload["partition_id"]),
            point_start=int(payload["point_start"]),
            point_stop=int(payload["point_stop"]),
            interval_ids=tuple(int(item) for item in payload["interval_ids"]),
            parameter_digest=str(payload["parameter_digest"]),
            partition_plan_digest=str(payload["partition_plan_digest"]),
            source_scattering_commit_digest=str(payload["source_scattering_commit_digest"]),
            source_replacement_digest=(
                None
                if payload.get("source_replacement_digest") is None
                else str(payload["source_replacement_digest"])
            ),
            backend_policy_digest=str(payload["backend_policy_digest"]),
            expected_output_digest=str(payload["expected_output_digest"]),
            contribution_reciprocal_points=int(payload["contribution_reciprocal_points"]),
            runtime_provenance=dict(payload["runtime_provenance"]),
            payload_path=str(payload["payload_path"]),
            payload_sha256=str(payload["payload_sha256"]),
            file_sha256=str(payload["file_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
        )


@dataclass(frozen=True)
class ResidualCommitCandidateManifest:
    run_digest: str
    candidate_id: str
    chunk_id: int
    partition_ids: tuple[int, ...]
    selected_attempts: tuple[dict[str, Any], ...]
    payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int
    reciprocal_point_count: int
    source_identity: dict[str, Any] | None = None

    schema: ClassVar[str] = RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "candidate_id": self.candidate_id,
            "chunk_id": int(self.chunk_id),
            "partition_ids": [int(item) for item in self.partition_ids],
            "selected_attempts": [dict(item) for item in self.selected_attempts],
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
            "reciprocal_point_count": int(self.reciprocal_point_count),
            "source_identity": self.source_identity,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualCommitCandidateManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            candidate_id=str(payload["candidate_id"]),
            chunk_id=int(payload["chunk_id"]),
            partition_ids=tuple(int(item) for item in payload["partition_ids"]),
            selected_attempts=tuple(dict(item) for item in payload["selected_attempts"]),
            payload_path=str(payload["payload_path"]),
            payload_sha256=str(payload["payload_sha256"]),
            file_sha256=str(payload["file_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
            reciprocal_point_count=int(payload["reciprocal_point_count"]),
            source_identity=(
                None
                if payload.get("source_identity") is None
                else dict(payload["source_identity"])
            ),
        )


@dataclass(frozen=True)
class ResidualChunkCommitManifest:
    run_digest: str
    chunk_id: int
    selected_candidate_id: str
    candidate_manifest_path: str
    candidate_payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int

    schema: ClassVar[str] = RESIDUAL_FIELD_CHUNK_COMMIT_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "chunk_id": int(self.chunk_id),
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_manifest_path": self.candidate_manifest_path,
            "candidate_payload_path": self.candidate_payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualChunkCommitManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            chunk_id=int(payload["chunk_id"]),
            selected_candidate_id=str(payload["selected_candidate_id"]),
            candidate_manifest_path=str(payload["candidate_manifest_path"]),
            candidate_payload_path=str(payload["candidate_payload_path"]),
            payload_sha256=str(payload["payload_sha256"]),
            file_sha256=str(payload["file_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
        )


@dataclass(frozen=True)
class ResidualStagePlanManifest:
    run_digest: str
    expected_by_chunk: tuple[dict[str, Any], ...]
    stage_plan_digest: str

    schema: ClassVar[str] = RESIDUAL_FIELD_STAGE_PLAN_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "expected_by_chunk": [dict(item) for item in self.expected_by_chunk],
            "stage_plan_digest": self.stage_plan_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualStagePlanManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            expected_by_chunk=tuple(
                {
                    "chunk_id": int(item["chunk_id"]),
                    "partition_ids": [int(partition_id) for partition_id in item["partition_ids"]],
                }
                for item in payload["expected_by_chunk"]
            ),
            stage_plan_digest=str(payload["stage_plan_digest"]),
        )


@dataclass(frozen=True)
class ResidualStageCommitManifest:
    run_digest: str
    chunk_ids: tuple[int, ...]
    chunk_commit_paths: tuple[str, ...]
    stage_digest: str
    stage_plan_digest: str | None = None

    schema: ClassVar[str] = RESIDUAL_FIELD_STAGE_COMMIT_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "chunk_ids": [int(item) for item in self.chunk_ids],
            "chunk_commit_paths": list(self.chunk_commit_paths),
            "stage_digest": self.stage_digest,
            "stage_plan_digest": self.stage_plan_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualStageCommitManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            chunk_ids=tuple(int(item) for item in payload["chunk_ids"]),
            chunk_commit_paths=tuple(str(item) for item in payload["chunk_commit_paths"]),
            stage_digest=str(payload["stage_digest"]),
            stage_plan_digest=(
                None
                if payload.get("stage_plan_digest") is None
                else str(payload["stage_plan_digest"])
            ),
        )


@dataclass(frozen=True)
class ResidualNoOutputManifest:
    run_digest: str
    source_scattering_commit_digest: str
    reason: str
    no_output_digest: str

    schema: ClassVar[str] = RESIDUAL_FIELD_NO_OUTPUT_SCHEMA
    schema_version: ClassVar[int] = RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": RESIDUAL_FIELD_STAGE,
            "source_scattering_commit_digest": self.source_scattering_commit_digest,
            "reason": self.reason,
            "no_output_digest": self.no_output_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualNoOutputManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            source_scattering_commit_digest=str(payload["source_scattering_commit_digest"]),
            reason=str(payload["reason"]),
            no_output_digest=str(payload["no_output_digest"]),
        )


def write_residual_attempt(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    partition_id: int,
    point_start: int,
    point_stop: int,
    interval_ids: Sequence[int],
    attempt_id: str,
    parameter_digest: str,
    partition_plan_digest: str,
    source_scattering_commit_digest: str,
    source_replacement_digest: str | None = None,
    backend_policy_digest: str,
    expected_output_digest: str,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    contribution_reciprocal_points: int,
    point_ids: np.ndarray,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> ResidualAttemptManifest:
    normalized_interval_ids = tuple(sorted(int(item) for item in interval_ids))
    # Device-independent checkpoint address. backend_policy_digest is still
    # received and recorded on the manifest as metadata, but NOT folded into the
    # address (so CPU and GPU partitions share one checkpoint).
    work_unit_digest = build_residual_work_unit_digest(
        run_digest=run_digest,
        chunk_id=chunk_id,
        partition_id=partition_id,
        point_start=point_start,
        point_stop=point_stop,
        interval_ids=normalized_interval_ids,
        parameter_digest=parameter_digest,
        partition_plan_digest=partition_plan_digest,
        source_scattering_commit_digest=source_scattering_commit_digest,
        source_replacement_digest=source_replacement_digest,
        expected_output_digest=expected_output_digest,
    )
    datasets = _payload_datasets(
        point_ids=point_ids,
        grid_shape_nd=grid_shape_nd,
        amplitudes_delta=amplitudes_delta,
        amplitudes_average=amplitudes_average,
    )
    attrs = _payload_attrs(
        schema=RESIDUAL_FIELD_ATTEMPT_SCHEMA,
        run_digest=run_digest,
        work_unit_digest=work_unit_digest,
        attempt_id=attempt_id,
        chunk_id=chunk_id,
        partition_id=partition_id,
        point_start=point_start,
        point_stop=point_stop,
        reciprocal_point_count=contribution_reciprocal_points,
    )
    payload_path = attempt_payload_path(
        output_dir,
        run_digest,
        RESIDUAL_FIELD_STAGE,
        int(chunk_id),
        work_unit_digest,
        attempt_id,
    )
    atomic_hdf5_write(payload_path, datasets, attrs=attrs)
    manifest = ResidualAttemptManifest(
        run_digest=str(run_digest),
        work_unit_digest=work_unit_digest,
        attempt_id=str(attempt_id),
        chunk_id=int(chunk_id),
        partition_id=int(partition_id),
        point_start=int(point_start),
        point_stop=int(point_stop),
        interval_ids=normalized_interval_ids,
        parameter_digest=str(parameter_digest),
        partition_plan_digest=str(partition_plan_digest),
        source_scattering_commit_digest=str(source_scattering_commit_digest),
        source_replacement_digest=(
            None if source_replacement_digest is None else str(source_replacement_digest)
        ),
        backend_policy_digest=str(backend_policy_digest),
        expected_output_digest=str(expected_output_digest),
        contribution_reciprocal_points=int(contribution_reciprocal_points),
        runtime_provenance=_resolve_runtime_provenance(runtime_provenance),
        payload_path=relative_to_output(payload_path, output_dir=output_dir),
        payload_sha256=_payload_digest(
            schema=RESIDUAL_FIELD_ATTEMPT_SCHEMA,
            expected_set_digest=work_unit_digest,
            datasets=datasets,
            attrs=attrs,
        ),
        file_sha256=file_sha256(payload_path),
        payload_nbytes=int(payload_path.stat().st_size),
    )
    write_manifest(
        attempt_manifest_path(
            output_dir,
            run_digest,
            RESIDUAL_FIELD_STAGE,
            int(chunk_id),
            work_unit_digest,
            attempt_id,
        ),
        manifest,
        output_dir=output_dir,
    )
    return manifest


def _attempt_manifest_paths(output_dir: str | Path, run_digest: str, chunk_id: int) -> list[Path]:
    root = chunk_root(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id) / "attempts"
    if not root.exists():
        return []
    return sorted(root.glob("*/*/attempt_*/attempt.json"))


def discover_residual_attempts(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> tuple[ResidualAttemptManifest, ...]:
    return tuple(
        read_manifest(path, codec=ResidualAttemptManifest, output_dir=output_dir)
        for path in _attempt_manifest_paths(output_dir, run_digest, chunk_id)
    )


def load_residual_attempt_payload(
    manifest: ResidualAttemptManifest,
    *,
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload_path = Path(output_dir) / manifest.payload_path
    datasets, attrs = _read_payload(payload_path)
    if file_sha256(payload_path) != manifest.file_sha256:
        raise ValueError(f"Residual attempt file hash mismatch: {manifest.payload_path}")
    if int(payload_path.stat().st_size) != int(manifest.payload_nbytes):
        raise ValueError(f"Residual attempt byte-size mismatch: {manifest.payload_path}")
    expected_payload_sha = _payload_digest(
        schema=RESIDUAL_FIELD_ATTEMPT_SCHEMA,
        expected_set_digest=manifest.work_unit_digest,
        datasets=datasets,
        attrs=attrs,
    )
    if expected_payload_sha != manifest.payload_sha256:
        raise ValueError(f"Residual attempt payload hash mismatch: {manifest.payload_path}")
    return datasets, attrs


def _identity_tuple(manifest: ResidualAttemptManifest) -> tuple[Any, ...]:
    # Device-independent identity. backend_policy_digest (which carries the
    # backend kind) is deliberately EXCLUDED so a CPU attempt and a GPU attempt for
    # the same partition share one identity and are reconciled by the numerical
    # agreement gate, not rejected as "conflicting".
    return (
        manifest.run_digest,
        manifest.parameter_digest,
        manifest.partition_plan_digest,
        manifest.source_scattering_commit_digest,
        manifest.source_replacement_digest,
        manifest.expected_output_digest,
    )


def _expected_work_unit_digest(manifest: ResidualAttemptManifest) -> str:
    return build_residual_work_unit_digest(
        run_digest=manifest.run_digest,
        chunk_id=manifest.chunk_id,
        partition_id=manifest.partition_id,
        point_start=manifest.point_start,
        point_stop=manifest.point_stop,
        interval_ids=manifest.interval_ids,
        parameter_digest=manifest.parameter_digest,
        partition_plan_digest=manifest.partition_plan_digest,
        source_scattering_commit_digest=manifest.source_scattering_commit_digest,
        source_replacement_digest=manifest.source_replacement_digest,
        expected_output_digest=manifest.expected_output_digest,
    )


class ResidualDivergenceError(RuntimeError):
    """Same-work residual results disagree beyond the numerical tolerance."""


# Mirror of the scattering gate: same-work residual retries -- a
# non-deterministic GPU relaunch, or a CPU vs GPU attempt for the same partition --
# need not be bit-identical but MUST agree NUMERICALLY. The tolerance is the PREDICTED
# forward-error bound rtol = S*(eps + M*u)*kappa (core/storage/agreement.py), tied to
# the NUFFT eps, the reciprocal-point summation depth M, and the per-channel
# cancellation factor kappa. A genuine divergence fails closed.


def _assert_residual_partials_agree(
    payloads: Sequence[Mapping[str, np.ndarray]],
    *,
    eps: float,
    summation_terms: int,
    context: str,
) -> None:
    """Fail closed unless every same-work residual payload agrees within the PREDICTED
    forward-error tolerance, evaluated PER FIELD.

    Byte-level ``payload_sha256`` equality is NOT required: bytes may differ
    across non-deterministic launches / devices, but a real numerical
    divergence must not be silently reconciled by deterministic winner selection. The
    tolerance is ``predict_agreement_rtol(eps, M, kappa)`` with ``M`` the
    reciprocal-point summation depth and ``kappa`` = 1 for the average channel and the
    cancellation amplification ``||avg||/||delta||`` for the delta channel.
    """
    reference = payloads[0]
    rtol_by_field = {
        "amplitudes_average": predict_agreement_rtol(
            eps=eps, summation_terms=summation_terms, kappa=1.0
        ),
        "amplitudes_delta": predict_agreement_rtol(
            eps=eps,
            summation_terms=summation_terms,
            kappa=cancellation_kappa(
                reference["amplitudes_average"], reference["amplitudes_delta"]
            ),
        ),
    }
    for other in payloads[1:]:
        for field, rtol in rtol_by_field.items():
            rel = relative_l2(reference[field], other[field])
            if not (rel <= rtol):
                raise ResidualDivergenceError(
                    f"Divergent residual results for {context}: {field} rel-L2 "
                    f"{rel:.3e} exceeds predicted agreement tolerance {rtol:.3e} "
                    f"(eps={eps:.1e}, M={int(summation_terms)})."
                )


def _select_attempts_by_partition(
    attempts: tuple[ResidualAttemptManifest, ...],
    *,
    expected_partitions: Mapping[int, Sequence[int]],
    output_dir: str | Path,
    eps: float = DEFAULT_NUFFT_EPS,
) -> tuple[ResidualAttemptManifest, ...]:
    by_partition: dict[int, list[ResidualAttemptManifest]] = {}
    for attempt in attempts:
        by_partition.setdefault(int(attempt.partition_id), []).append(attempt)
    selected: list[ResidualAttemptManifest] = []
    for partition_id in sorted(int(item) for item in expected_partitions):
        candidates = by_partition.get(partition_id, [])
        if not candidates:
            raise RuntimeError(
                f"Residual commit candidate missing attempts for partition {partition_id}."
            )
        # Bitwise payload equality is REPLACED (not removed) by the PREDICTED
        # numerical agreement gate. Same-work retries (non-deterministic GPU
        # relaunches, or a CPU vs GPU attempt) may differ in bytes but MUST agree
        # within the predicted tolerance; a genuine divergence fails closed.
        if len(candidates) > 1:
            payloads = [
                load_residual_attempt_payload(candidate, output_dir=output_dir)[0]
                for candidate in candidates
            ]
            _assert_residual_partials_agree(
                payloads,
                eps=eps,
                summation_terms=int(candidates[0].contribution_reciprocal_points),
                context=f"partition {partition_id}",
            )
        selected.append(sorted(candidates, key=lambda item: (item.attempt_id, item.payload_path))[0])
    return tuple(selected)


def _candidate_id(
    *,
    chunk_id: int,
    selected_attempts: tuple[ResidualAttemptManifest, ...],
) -> str:
    # Address-based identity: the candidate is addressed by its chunk and the
    # DEVICE-INDEPENDENT work-unit digests of its selected partitions -- NOT by
    # payload bytes. payload_sha256 stays on each attempt manifest and is verified at
    # load time (load_residual_attempt_payload) for disk integrity, but it is NOT part
    # of commit identity, so a CPU-written and a GPU-written candidate for the same
    # science map to ONE address (first valid writer commits; later valid writers,
    # reconciled by the agreement gate, resolve to the same candidate_id).
    return digest_dict(
        {
            "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
            "chunk_id": int(chunk_id),
            "attempt_addresses": [
                {
                    "partition_id": attempt.partition_id,
                    "work_unit_digest": attempt.work_unit_digest,
                }
                for attempt in selected_attempts
            ],
        },
        domain="mosaic.residual_field.commit_candidate_id.v2",
    )[:32]


def create_residual_commit_candidate(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    expected_partitions: Mapping[int, Sequence[int]],
    expected_reciprocal_point_count: int | None = None,
    eps: float = DEFAULT_NUFFT_EPS,
) -> ResidualCommitCandidateManifest:
    normalized_expected = {
        int(partition_id): tuple(int(point_id) for point_id in point_ids)
        for partition_id, point_ids in expected_partitions.items()
    }
    if not normalized_expected:
        raise ValueError("Residual commit candidates require expected partitions.")
    attempts = discover_residual_attempts(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_id=chunk_id,
    )
    expected_partition_ids = set(normalized_expected)
    unexpected = sorted(
        {
            int(attempt.partition_id)
            for attempt in attempts
            if int(attempt.partition_id) not in expected_partition_ids
        }
    )
    if unexpected:
        raise RuntimeError(
            "Residual commit candidate found attempts for unexpected partitions: "
            + ", ".join(str(item) for item in unexpected)
        )
    identities = {_identity_tuple(attempt) for attempt in attempts}
    if len(identities) > 1:
        raise RuntimeError("Conflicting residual attempt source identity for chunk commit.")

    for attempt in attempts:
        if attempt.work_unit_digest != _expected_work_unit_digest(attempt):
            raise RuntimeError(
                "Residual attempt work-unit digest mismatch for "
                f"partition {int(attempt.partition_id)}."
            )
        if int(attempt.point_stop) <= int(attempt.point_start):
            raise RuntimeError("Residual attempt point range must be non-empty.")
        load_residual_attempt_payload(attempt, output_dir=output_dir)

    selected = _select_attempts_by_partition(
        attempts,
        expected_partitions=normalized_expected,
        output_dir=output_dir,
        eps=eps,
    )
    datasets_by_partition: list[tuple[ResidualAttemptManifest, dict[str, np.ndarray]]] = []
    reciprocal_counts: set[int] = set()
    all_point_ids: list[int] = []
    for attempt in selected:
        datasets, _attrs = load_residual_attempt_payload(attempt, output_dir=output_dir)
        point_ids = tuple(int(item) for item in np.asarray(datasets["point_ids"]).reshape(-1))
        expected_point_ids = normalized_expected[int(attempt.partition_id)]
        if point_ids != expected_point_ids:
            raise RuntimeError(
                "Residual attempt point IDs do not match expected partition coverage "
                f"for partition {int(attempt.partition_id)}."
            )
        if len(point_ids) != int(attempt.point_stop) - int(attempt.point_start):
            raise RuntimeError(
                "Residual attempt point range does not match payload length for "
                f"partition {int(attempt.partition_id)}."
            )
        all_point_ids.extend(point_ids)
        reciprocal_counts.add(int(attempt.contribution_reciprocal_points))
        datasets_by_partition.append((attempt, datasets))

    if len(set(all_point_ids)) != len(all_point_ids):
        raise RuntimeError("Residual partition coverage contains duplicate point IDs.")
    if expected_reciprocal_point_count is not None:
        reciprocal_counts.add(int(expected_reciprocal_point_count))
    if len(reciprocal_counts) != 1:
        raise RuntimeError("Residual partition reciprocal point counts do not agree.")
    reciprocal_point_count = next(iter(reciprocal_counts))

    ordered_datasets = [datasets for _attempt, datasets in datasets_by_partition]
    candidate_id = _candidate_id(chunk_id=chunk_id, selected_attempts=selected)
    merged = _payload_datasets(
        point_ids=np.concatenate([np.asarray(item["point_ids"], dtype=np.int64) for item in ordered_datasets]),
        grid_shape_nd=np.asarray(ordered_datasets[0]["grid_shape_nd"], dtype=np.int64),
        amplitudes_delta=np.concatenate(
            [np.asarray(item["amplitudes_delta"], dtype=np.complex128) for item in ordered_datasets]
        ),
        amplitudes_average=np.concatenate(
            [np.asarray(item["amplitudes_average"], dtype=np.complex128) for item in ordered_datasets]
        ),
    )
    attrs = _payload_attrs(
        schema=RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
        run_digest=run_digest,
        work_unit_digest=candidate_id,
        attempt_id=None,
        chunk_id=chunk_id,
        partition_id=-1,
        point_start=0,
        point_stop=int(np.asarray(merged["point_ids"]).reshape(-1).shape[0]),
        reciprocal_point_count=reciprocal_point_count,
    )
    payload_path = commit_candidate_payload_path(
        output_dir,
        run_digest,
        RESIDUAL_FIELD_STAGE,
        int(chunk_id),
        candidate_id,
    )
    atomic_hdf5_write(payload_path, merged, attrs=attrs)
    manifest = ResidualCommitCandidateManifest(
        run_digest=str(run_digest),
        candidate_id=candidate_id,
        chunk_id=int(chunk_id),
        partition_ids=tuple(int(item.partition_id) for item in selected),
        selected_attempts=tuple(
            {
                "partition_id": int(attempt.partition_id),
                "attempt_id": attempt.attempt_id,
                "work_unit_digest": attempt.work_unit_digest,
                "payload_sha256": attempt.payload_sha256,
                "file_sha256": attempt.file_sha256,
                "payload_path": attempt.payload_path,
            }
            for attempt in selected
        ),
        payload_path=relative_to_output(payload_path, output_dir=output_dir),
        payload_sha256=_payload_digest(
            schema=RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
            expected_set_digest=candidate_id,
            datasets=merged,
            attrs=attrs,
        ),
        file_sha256=file_sha256(payload_path),
        payload_nbytes=int(payload_path.stat().st_size),
        reciprocal_point_count=int(reciprocal_point_count),
        source_identity={
            "parameter_digest": selected[0].parameter_digest,
            "partition_plan_digest": selected[0].partition_plan_digest,
            "source_scattering_commit_digest": selected[0].source_scattering_commit_digest,
            "source_replacement_digest": selected[0].source_replacement_digest,
            "backend_policy_digest": selected[0].backend_policy_digest,
            "expected_output_digest": selected[0].expected_output_digest,
        },
    )
    write_manifest(
        commit_candidate_manifest_path(
            output_dir,
            run_digest,
            RESIDUAL_FIELD_STAGE,
            int(chunk_id),
            candidate_id,
        ),
        manifest,
        output_dir=output_dir,
    )
    return manifest


def _commit_candidate_manifest_paths(
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> list[Path]:
    root = commit_attempts_root(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id)
    if not root.exists():
        return []
    return sorted(root.glob("commit_*/commit_candidate.json"))


def discover_residual_commit_candidates(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> tuple[ResidualCommitCandidateManifest, ...]:
    return tuple(
        read_manifest(path, codec=ResidualCommitCandidateManifest, output_dir=output_dir)
        for path in _commit_candidate_manifest_paths(output_dir, run_digest, chunk_id)
    )


def _record_residual_commit_scan_seconds(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    scan_seconds: float,
) -> None:
    write_performance_metrics(
        output_dir=output_dir,
        run_digest=run_digest,
        commit_scan_seconds={
            f"{RESIDUAL_FIELD_STAGE}/chunk_{int(chunk_id)}": float(scan_seconds)
        },
    )


def _load_residual_candidate_payload(
    candidate: ResidualCommitCandidateManifest,
    *,
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload_path = Path(output_dir) / candidate.payload_path
    datasets, attrs = _read_payload(payload_path)
    if file_sha256(payload_path) != candidate.file_sha256:
        raise ValueError(f"Residual candidate file hash mismatch: {candidate.payload_path}")
    if int(payload_path.stat().st_size) != int(candidate.payload_nbytes):
        raise ValueError(f"Residual candidate byte-size mismatch: {candidate.payload_path}")
    expected_payload_sha = _payload_digest(
        schema=RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA,
        expected_set_digest=candidate.candidate_id,
        datasets=datasets,
        attrs=attrs,
    )
    if expected_payload_sha != candidate.payload_sha256:
        raise ValueError(f"Residual candidate payload hash mismatch: {candidate.payload_path}")
    return datasets, attrs


def validate_residual_commit_candidate(
    candidate: ResidualCommitCandidateManifest,
    *,
    output_dir: str | Path,
) -> ResidualCommitCandidateManifest:
    expected_payload_path = relative_to_output(
        commit_candidate_payload_path(
            output_dir,
            candidate.run_digest,
            RESIDUAL_FIELD_STAGE,
            int(candidate.chunk_id),
            candidate.candidate_id,
        ),
        output_dir=output_dir,
    )
    if candidate.payload_path != expected_payload_path:
        raise RuntimeError("Residual commit candidate payload path does not match deterministic path.")
    selected_partitions = tuple(
        int(item["partition_id"]) for item in candidate.selected_attempts
    )
    if tuple(sorted(selected_partitions)) != tuple(sorted(candidate.partition_ids)):
        raise RuntimeError("Residual commit candidate partition coverage is inconsistent.")
    attempts_by_key = {
        (attempt.work_unit_digest, attempt.attempt_id): attempt
        for attempt in discover_residual_attempts(
            output_dir=output_dir,
            run_digest=candidate.run_digest,
            chunk_id=int(candidate.chunk_id),
        )
    }
    for selected in candidate.selected_attempts:
        key = (str(selected["work_unit_digest"]), str(selected["attempt_id"]))
        attempt = attempts_by_key.get(key)
        if attempt is None:
            raise RuntimeError("Residual commit candidate references a missing attempt.")
        if int(selected["partition_id"]) != int(attempt.partition_id):
            raise RuntimeError("Residual commit candidate attempt partition mismatch.")
        if str(selected["payload_sha256"]) != attempt.payload_sha256:
            raise RuntimeError("Residual commit candidate attempt payload hash mismatch.")
        if str(selected["file_sha256"]) != attempt.file_sha256:
            raise RuntimeError("Residual commit candidate attempt file hash mismatch.")
        if str(selected["payload_path"]) != attempt.payload_path:
            raise RuntimeError("Residual commit candidate attempt payload path mismatch.")
        load_residual_attempt_payload(attempt, output_dir=output_dir)
    _load_residual_candidate_payload(candidate, output_dir=output_dir)
    return candidate


def promote_residual_chunk_commit_by_scan(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> ResidualChunkCommitManifest:
    scan_started = time.perf_counter()
    candidates = discover_residual_commit_candidates(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_id=int(chunk_id),
    )
    valid: list[ResidualCommitCandidateManifest] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            valid.append(validate_residual_commit_candidate(candidate, output_dir=output_dir))
        except Exception as exc:
            errors.append(f"{candidate.candidate_id}: {type(exc).__name__}: {exc}")
    if not valid:
        detail = "; ".join(errors) if errors else "no candidate manifests found"
        raise RuntimeError(
            f"No valid residual commit candidates for chunk {int(chunk_id)}: {detail}"
        )
    # Cross-candidate bitwise payload_sha256 equality is NOT required.
    # Attempt-level numerical agreement (_select_attempts_by_partition) already
    # guarantees that the candidates for this chunk are numerically consistent, so a
    # deterministic winner (lowest candidate_id) is canonical; bytes need not match
    # across non-deterministic launches / devices.
    candidate = sorted(valid, key=lambda item: item.candidate_id)[0]
    candidate_manifest = commit_candidate_manifest_path(
        output_dir,
        candidate.run_digest,
        RESIDUAL_FIELD_STAGE,
        int(candidate.chunk_id),
        candidate.candidate_id,
    )
    target = chunk_commit_path(output_dir, candidate.run_digest, RESIDUAL_FIELD_STAGE, candidate.chunk_id)
    manifest = ResidualChunkCommitManifest(
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
        selected_candidate_id=candidate.candidate_id,
        candidate_manifest_path=relative_to_output(candidate_manifest, output_dir=output_dir),
        candidate_payload_path=candidate.payload_path,
        payload_sha256=candidate.payload_sha256,
        file_sha256=candidate.file_sha256,
        payload_nbytes=int(candidate.payload_nbytes),
    )
    committed, _created = try_commit_manifest(
        target,
        manifest,
        codec=ResidualChunkCommitManifest,
        output_dir=output_dir,
        conflict_error=lambda _existing: RuntimeError(
            f"Residual chunk {candidate.chunk_id} already committed to a different candidate."
        ),
    )
    _record_residual_commit_scan_seconds(
        output_dir=output_dir,
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
        scan_seconds=time.perf_counter() - scan_started,
    )
    return committed


def promote_residual_chunk_commit(
    *,
    output_dir: str | Path,
    candidate: ResidualCommitCandidateManifest,
) -> ResidualChunkCommitManifest:
    validate_residual_commit_candidate(candidate, output_dir=output_dir)
    candidate_manifest = commit_candidate_manifest_path(
        output_dir,
        candidate.run_digest,
        RESIDUAL_FIELD_STAGE,
        int(candidate.chunk_id),
        candidate.candidate_id,
    )
    target = chunk_commit_path(output_dir, candidate.run_digest, RESIDUAL_FIELD_STAGE, candidate.chunk_id)
    manifest = ResidualChunkCommitManifest(
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
        selected_candidate_id=candidate.candidate_id,
        candidate_manifest_path=relative_to_output(candidate_manifest, output_dir=output_dir),
        candidate_payload_path=candidate.payload_path,
        payload_sha256=candidate.payload_sha256,
        file_sha256=candidate.file_sha256,
        payload_nbytes=int(candidate.payload_nbytes),
    )
    if target.exists():
        existing = read_manifest(target, codec=ResidualChunkCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError(
                f"Residual chunk {candidate.chunk_id} already committed to a different candidate."
            )
        return existing
    return promote_residual_chunk_commit_by_scan(
        output_dir=output_dir,
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
    )


def _normalize_stage_plan(
    expected_by_chunk: Mapping[int, Sequence[int]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "chunk_id": int(chunk_id),
            "partition_ids": sorted({int(partition_id) for partition_id in partition_ids}),
        }
        for chunk_id, partition_ids in sorted(expected_by_chunk.items())
    )


def write_residual_stage_plan(
    *,
    output_dir: str | Path,
    run_digest: str,
    expected_by_chunk: Mapping[int, Sequence[int]],
) -> ResidualStagePlanManifest:
    expected = _normalize_stage_plan(expected_by_chunk)
    manifest = ResidualStagePlanManifest(
        run_digest=str(run_digest),
        expected_by_chunk=expected,
        stage_plan_digest=digest_dict(
            {
                "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
                "run_digest": str(run_digest),
                "stage": RESIDUAL_FIELD_STAGE,
                "expected_by_chunk": expected,
            },
            domain="mosaic.residual_field.stage_plan.v1",
        ),
    )
    target = stage_plan_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE)
    if target.exists():
        existing = read_manifest(target, codec=ResidualStagePlanManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError("Residual stage_plan.json already exists with different coverage.")
        return existing
    write_manifest(target, manifest, output_dir=output_dir)
    return manifest


def write_residual_stage_commit(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_ids: Sequence[int],
) -> ResidualStageCommitManifest:
    requested = tuple(sorted(int(item) for item in chunk_ids))
    plan_path = stage_plan_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE)
    stage_plan = (
        read_manifest(plan_path, codec=ResidualStagePlanManifest, output_dir=output_dir)
        if plan_path.exists()
        else None
    )
    if stage_plan is not None:
        expected_chunks = tuple(int(item["chunk_id"]) for item in stage_plan.expected_by_chunk)
        if requested != expected_chunks:
            raise RuntimeError("Residual stage_commit chunk IDs do not match stage_plan.json.")
        expected_stage_plan_digest = digest_dict(
            {
                "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
                "run_digest": str(run_digest),
                "stage": RESIDUAL_FIELD_STAGE,
                "expected_by_chunk": stage_plan.expected_by_chunk,
            },
            domain="mosaic.residual_field.stage_plan.v1",
        )
        if stage_plan.stage_plan_digest != expected_stage_plan_digest:
            raise RuntimeError("Residual stage_plan.json digest does not match coverage.")
    chunks_dir = stage_root(output_dir, run_digest, RESIDUAL_FIELD_STAGE) / "chunks"
    committed_on_disk: tuple[int, ...] = ()
    if chunks_dir.exists():
        chunk_ids_on_disk: list[int] = []
        for path in sorted(chunks_dir.glob("chunk_*/chunk_commit.json")):
            try:
                chunk_ids_on_disk.append(int(path.parent.name.removeprefix("chunk_")))
            except ValueError:
                raise RuntimeError(f"Invalid residual chunk commit directory: {path.parent.name!r}")
        committed_on_disk = tuple(sorted(chunk_ids_on_disk))
    if committed_on_disk != requested:
        raise RuntimeError(
            "Residual stage_commit coverage must match committed chunk manifests: "
            f"requested={requested}, committed={committed_on_disk}."
        )
    commits: list[ResidualChunkCommitManifest] = []
    paths: list[str] = []
    for chunk_id in requested:
        path = chunk_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE, chunk_id)
        if not path.exists():
            raise RuntimeError(f"Missing residual chunk_commit.json for chunk {chunk_id}.")
        commit = read_manifest(path, codec=ResidualChunkCommitManifest, output_dir=output_dir)
        if commit.run_digest != str(run_digest):
            raise RuntimeError("Residual chunk commit run identity mismatch.")
        candidate = read_manifest(
            Path(output_dir) / commit.candidate_manifest_path,
            codec=ResidualCommitCandidateManifest,
            output_dir=output_dir,
        )
        validate_residual_commit_candidate(candidate, output_dir=output_dir)
        if candidate.candidate_id != commit.selected_candidate_id:
            raise RuntimeError("Residual chunk commit selected candidate mismatch.")
        if candidate.payload_path != commit.candidate_payload_path:
            raise RuntimeError("Residual chunk commit payload path mismatch.")
        if candidate.payload_sha256 != commit.payload_sha256:
            raise RuntimeError("Residual chunk commit payload hash mismatch.")
        if candidate.file_sha256 != commit.file_sha256:
            raise RuntimeError("Residual chunk commit file hash mismatch.")
        if int(candidate.payload_nbytes) != int(commit.payload_nbytes):
            raise RuntimeError("Residual chunk commit byte-size mismatch.")
        commits.append(commit)
        paths.append(relative_to_output(path, output_dir=output_dir))
    stage_digest = digest_dict(
        {
            "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
            "stage_plan_digest": None if stage_plan is None else stage_plan.stage_plan_digest,
            "chunk_commits": [commit.to_payload() for commit in commits],
        },
        domain="mosaic.residual_field.stage_commit.v1",
    )
    manifest = ResidualStageCommitManifest(
        run_digest=str(run_digest),
        chunk_ids=requested,
        chunk_commit_paths=tuple(paths),
        stage_digest=stage_digest,
        stage_plan_digest=None if stage_plan is None else stage_plan.stage_plan_digest,
    )
    target = stage_commit_path(output_dir, run_digest, RESIDUAL_FIELD_STAGE)
    if target.exists():
        existing = read_manifest(target, codec=ResidualStageCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError("Residual stage_commit.json already exists with different identity.")
        return existing
    write_manifest(
        target,
        manifest,
        output_dir=output_dir,
    )
    return manifest


def _no_output_path(output_dir: str | Path, run_digest: str) -> Path:
    return stage_root(output_dir, run_digest, RESIDUAL_FIELD_STAGE) / "no_output.json"


def write_residual_no_output_manifest(
    *,
    output_dir: str | Path,
    run_digest: str,
    source_scattering_commit_digest: str,
    reason: str = "empty replacement coverage",
) -> ResidualNoOutputManifest:
    digest = digest_dict(
        {
            "schema_version": RESIDUAL_FIELD_COMMIT_SCHEMA_VERSION,
            "run_digest": str(run_digest),
            "stage": RESIDUAL_FIELD_STAGE,
            "source_scattering_commit_digest": str(source_scattering_commit_digest),
            "reason": str(reason),
        },
        domain="mosaic.residual_field.no_output.v1",
    )
    manifest = ResidualNoOutputManifest(
        run_digest=str(run_digest),
        source_scattering_commit_digest=str(source_scattering_commit_digest),
        reason=str(reason),
        no_output_digest=digest,
    )
    target = _no_output_path(output_dir, run_digest)
    if target.exists():
        existing = read_manifest(target, codec=ResidualNoOutputManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError("Residual no_output.json already exists with different identity.")
        return existing
    write_manifest(target, manifest, output_dir=output_dir)
    return manifest


def require_residual_no_output_manifest(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> ResidualNoOutputManifest:
    stale_outputs = sorted(Path(output_dir).glob("residual_chunk_*"))
    target = _no_output_path(output_dir, run_digest)
    if not target.exists():
        if stale_outputs:
            raise RuntimeError(
                "Stale residual_chunk_* files are not valid no-output evidence without "
                "run-scoped residual no_output.json."
            )
        raise RuntimeError("Missing residual no_output.json.")
    return read_manifest(target, codec=ResidualNoOutputManifest, output_dir=output_dir)


__all__ = [
    "RESIDUAL_FIELD_ATTEMPT_SCHEMA",
    "RESIDUAL_FIELD_CHUNK_COMMIT_SCHEMA",
    "RESIDUAL_FIELD_COMMIT_CANDIDATE_SCHEMA",
    "RESIDUAL_FIELD_NO_OUTPUT_SCHEMA",
    "RESIDUAL_FIELD_STAGE_COMMIT_SCHEMA",
    "RESIDUAL_FIELD_STAGE_PLAN_SCHEMA",
    "ResidualAttemptManifest",
    "ResidualChunkCommitManifest",
    "ResidualCommitCandidateManifest",
    "ResidualNoOutputManifest",
    "ResidualStageCommitManifest",
    "ResidualStagePlanManifest",
    "build_residual_work_unit_digest",
    "create_residual_commit_candidate",
    "discover_residual_attempts",
    "discover_residual_commit_candidates",
    "load_residual_attempt_payload",
    "promote_residual_chunk_commit_by_scan",
    "promote_residual_chunk_commit",
    "require_residual_no_output_manifest",
    "validate_residual_commit_candidate",
    "write_residual_attempt",
    "write_residual_no_output_manifest",
    "write_residual_stage_commit",
    "write_residual_stage_plan",
]
