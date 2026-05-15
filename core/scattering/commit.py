from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, ClassVar, Mapping

import h5py
import numpy as np

from core.scattering.accumulation import (
    build_scattering_partial_result,
    merge_scattering_partial_results,
)
from core.scattering.artifacts import _atomic_hdf5_write
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
from core.storage.digests import digest_dict, normalize_digest_input
from core.storage.fingerprint import file_sha256, payload_sha256
from core.storage.manifest import read_manifest, write_manifest
from core.storage.performance import write_performance_metrics
from core.runtime.gpu_admission import runtime_provenance_for_attempt


SCATTERING_ATTEMPT_SCHEMA = "mosaic.scattering.attempt"
SCATTERING_COMMIT_CANDIDATE_SCHEMA = "mosaic.scattering.commit_candidate"
SCATTERING_CHUNK_COMMIT_SCHEMA = "mosaic.scattering.chunk_commit"
SCATTERING_STAGE_PLAN_SCHEMA = "mosaic.scattering.stage_plan"
SCATTERING_STAGE_COMMIT_SCHEMA = "mosaic.scattering.stage_commit"
SCATTERING_COMMIT_SCHEMA_VERSION = 1
SCATTERING_STAGE = "scattering"


def build_scattering_work_unit_digest(
    *,
    interval_id: int,
    chunk_id: int,
    scientific_digest: str,
    execution_digest: str,
    qspace_plan_digest: str,
    backend_policy_digest: str,
    source_structure_digest: str,
) -> str:
    return digest_dict(
        {
            "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
            "stage": SCATTERING_STAGE,
            "interval_id": int(interval_id),
            "chunk_id": int(chunk_id),
            "scientific_digest": str(scientific_digest),
            "execution_digest": str(execution_digest),
            "qspace_plan_digest": str(qspace_plan_digest),
            "backend_policy_digest": str(backend_policy_digest),
            "source_structure_digest": str(source_structure_digest),
        },
        domain="mosaic.scattering.work_unit.v1",
    )


def _payload_datasets(
    *,
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
) -> dict[str, np.ndarray]:
    delta = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
    average = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
    point_id_arr = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    if average.shape != delta.shape:
        raise ValueError("amplitudes_delta and amplitudes_average must have matching shapes.")
    if point_id_arr.shape != delta.shape:
        raise ValueError("point_ids must align with amplitude arrays.")
    return {
        "point_ids": point_id_arr,
        "grid_shape_nd": np.asarray(grid_shape_nd, dtype=np.int64),
        "amplitudes_delta": delta,
        "amplitudes_average": average,
    }


def _attempt_payload_attrs(
    *,
    run_digest: str,
    work_unit_digest: str,
    attempt_id: str,
    interval_id: int,
    chunk_id: int,
    contribution_reciprocal_points: int,
) -> dict[str, object]:
    return {
        "schema": SCATTERING_ATTEMPT_SCHEMA,
        "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
        "run_digest": str(run_digest),
        "work_unit_digest": str(work_unit_digest),
        "attempt_id": str(attempt_id),
        "stage": SCATTERING_STAGE,
        "interval_id": int(interval_id),
        "chunk_id": int(chunk_id),
        "contribution_reciprocal_points": int(contribution_reciprocal_points),
    }


def _candidate_payload_attrs(
    *,
    run_digest: str,
    candidate_id: str,
    chunk_id: int,
    reciprocal_point_count: int,
) -> dict[str, object]:
    return {
        "schema": SCATTERING_COMMIT_CANDIDATE_SCHEMA,
        "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
        "run_digest": str(run_digest),
        "candidate_id": str(candidate_id),
        "stage": SCATTERING_STAGE,
        "chunk_id": int(chunk_id),
        "reciprocal_point_count": int(reciprocal_point_count),
    }


def _payload_digest(
    *,
    expected_set_digest: str,
    datasets: Mapping[str, np.ndarray],
    attrs: Mapping[str, Any],
) -> str:
    semantic_attrs = {
        key: value
        for key, value in attrs.items()
        if key not in {"attempt_id"}
    }
    return payload_sha256(
        schema=str(attrs["schema"]),
        expected_set_digest=str(expected_set_digest),
        datasets=datasets,
        attrs=semantic_attrs,
    )


def _read_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as h5file:
        datasets = {name: np.asarray(h5file[name]) for name in h5file.keys()}
        attrs = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in h5file.attrs.items()
        }
    return datasets, attrs


def _resolve_runtime_provenance(
    runtime_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = dict(runtime_provenance or {})
    return runtime_provenance_for_attempt(
        fs_capability_digest=base.get("fs_capability_digest"),
        scheduler_kind=str(base.get("scheduler_kind", "local")),
        nufft_policy=base.get("nufft_policy", "auto"),
        resource_requirements=base.get("resource_requirements"),
        cuda_probe=str(base.get("nufft_policy", "auto")) in {"gpu-required", "allow-fallback"},
    )


@dataclass(frozen=True)
class ScatteringAttemptManifest:
    run_digest: str
    work_unit_digest: str
    attempt_id: str
    interval_id: int
    chunk_id: int
    scientific_digest: str
    execution_digest: str
    qspace_plan_digest: str
    backend_policy_digest: str
    source_structure_digest: str
    contribution_reciprocal_points: int
    runtime_provenance: dict[str, Any]
    payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int

    schema: ClassVar[str] = SCATTERING_ATTEMPT_SCHEMA
    schema_version: ClassVar[int] = SCATTERING_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": SCATTERING_STAGE,
            "work_unit_digest": self.work_unit_digest,
            "attempt_id": self.attempt_id,
            "interval_id": int(self.interval_id),
            "chunk_id": int(self.chunk_id),
            "scientific_digest": self.scientific_digest,
            "execution_digest": self.execution_digest,
            "qspace_plan_digest": self.qspace_plan_digest,
            "backend_policy_digest": self.backend_policy_digest,
            "source_structure_digest": self.source_structure_digest,
            "contribution_reciprocal_points": int(self.contribution_reciprocal_points),
            "runtime_provenance": dict(self.runtime_provenance),
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScatteringAttemptManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            work_unit_digest=str(payload["work_unit_digest"]),
            attempt_id=str(payload["attempt_id"]),
            interval_id=int(payload["interval_id"]),
            chunk_id=int(payload["chunk_id"]),
            scientific_digest=str(payload["scientific_digest"]),
            execution_digest=str(payload["execution_digest"]),
            qspace_plan_digest=str(payload["qspace_plan_digest"]),
            backend_policy_digest=str(payload["backend_policy_digest"]),
            source_structure_digest=str(payload["source_structure_digest"]),
            contribution_reciprocal_points=int(payload["contribution_reciprocal_points"]),
            runtime_provenance=dict(payload["runtime_provenance"]),
            payload_path=str(payload["payload_path"]),
            payload_sha256=str(payload["payload_sha256"]),
            file_sha256=str(payload["file_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
        )


@dataclass(frozen=True)
class ScatteringCommitCandidateManifest:
    run_digest: str
    candidate_id: str
    chunk_id: int
    contributing_interval_ids: tuple[int, ...]
    selected_attempts: tuple[dict[str, Any], ...]
    payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int
    reciprocal_point_count: int

    schema: ClassVar[str] = SCATTERING_COMMIT_CANDIDATE_SCHEMA
    schema_version: ClassVar[int] = SCATTERING_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": SCATTERING_STAGE,
            "candidate_id": self.candidate_id,
            "chunk_id": int(self.chunk_id),
            "contributing_interval_ids": [int(item) for item in self.contributing_interval_ids],
            "selected_attempts": [dict(item) for item in self.selected_attempts],
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
            "reciprocal_point_count": int(self.reciprocal_point_count),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScatteringCommitCandidateManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            candidate_id=str(payload["candidate_id"]),
            chunk_id=int(payload["chunk_id"]),
            contributing_interval_ids=tuple(int(item) for item in payload["contributing_interval_ids"]),
            selected_attempts=tuple(dict(item) for item in payload["selected_attempts"]),
            payload_path=str(payload["payload_path"]),
            payload_sha256=str(payload["payload_sha256"]),
            file_sha256=str(payload["file_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
            reciprocal_point_count=int(payload["reciprocal_point_count"]),
        )


@dataclass(frozen=True)
class ScatteringChunkCommitManifest:
    run_digest: str
    chunk_id: int
    selected_candidate_id: str
    candidate_manifest_path: str
    candidate_payload_path: str
    payload_sha256: str
    file_sha256: str
    payload_nbytes: int

    schema: ClassVar[str] = SCATTERING_CHUNK_COMMIT_SCHEMA
    schema_version: ClassVar[int] = SCATTERING_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": SCATTERING_STAGE,
            "chunk_id": int(self.chunk_id),
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_manifest_path": self.candidate_manifest_path,
            "candidate_payload_path": self.candidate_payload_path,
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "payload_nbytes": int(self.payload_nbytes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScatteringChunkCommitManifest":
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
class ScatteringStagePlanManifest:
    run_digest: str
    expected_by_chunk: tuple[dict[str, Any], ...]
    stage_plan_digest: str

    schema: ClassVar[str] = SCATTERING_STAGE_PLAN_SCHEMA
    schema_version: ClassVar[int] = SCATTERING_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": SCATTERING_STAGE,
            "expected_by_chunk": [dict(item) for item in self.expected_by_chunk],
            "stage_plan_digest": self.stage_plan_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScatteringStagePlanManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            expected_by_chunk=tuple(
                {
                    "chunk_id": int(item["chunk_id"]),
                    "expected_interval_ids": [
                        int(interval_id) for interval_id in item["expected_interval_ids"]
                    ],
                }
                for item in payload["expected_by_chunk"]
            ),
            stage_plan_digest=str(payload["stage_plan_digest"]),
        )


@dataclass(frozen=True)
class ScatteringStageCommitManifest:
    run_digest: str
    chunk_ids: tuple[int, ...]
    chunk_commit_paths: tuple[str, ...]
    stage_digest: str
    stage_plan_digest: str | None = None

    schema: ClassVar[str] = SCATTERING_STAGE_COMMIT_SCHEMA
    schema_version: ClassVar[int] = SCATTERING_COMMIT_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "stage": SCATTERING_STAGE,
            "chunk_ids": [int(item) for item in self.chunk_ids],
            "chunk_commit_paths": list(self.chunk_commit_paths),
            "stage_digest": self.stage_digest,
            "stage_plan_digest": self.stage_plan_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScatteringStageCommitManifest":
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


def write_scattering_attempt(
    *,
    output_dir: str | Path,
    run_digest: str,
    interval_id: int,
    chunk_id: int,
    attempt_id: str,
    scientific_digest: str,
    execution_digest: str,
    qspace_plan_digest: str,
    backend_policy_digest: str,
    source_structure_digest: str,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    contribution_reciprocal_points: int,
    point_ids: np.ndarray | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> ScatteringAttemptManifest:
    if point_ids is None:
        point_ids = np.arange(np.asarray(amplitudes_delta).reshape(-1).shape[0], dtype=np.int64)
    work_unit_digest = build_scattering_work_unit_digest(
        interval_id=interval_id,
        chunk_id=chunk_id,
        scientific_digest=scientific_digest,
        execution_digest=execution_digest,
        qspace_plan_digest=qspace_plan_digest,
        backend_policy_digest=backend_policy_digest,
        source_structure_digest=source_structure_digest,
    )
    datasets = _payload_datasets(
        point_ids=np.asarray(point_ids),
        grid_shape_nd=np.asarray(grid_shape_nd),
        amplitudes_delta=np.asarray(amplitudes_delta),
        amplitudes_average=np.asarray(amplitudes_average),
    )
    attrs = _attempt_payload_attrs(
        run_digest=run_digest,
        work_unit_digest=work_unit_digest,
        attempt_id=attempt_id,
        interval_id=interval_id,
        chunk_id=chunk_id,
        contribution_reciprocal_points=contribution_reciprocal_points,
    )
    payload_path = attempt_payload_path(
        output_dir,
        run_digest,
        SCATTERING_STAGE,
        int(chunk_id),
        work_unit_digest,
        attempt_id,
    )
    _atomic_hdf5_write(payload_path, datasets, attrs=attrs)
    manifest = ScatteringAttemptManifest(
        run_digest=str(run_digest),
        work_unit_digest=work_unit_digest,
        attempt_id=str(attempt_id),
        interval_id=int(interval_id),
        chunk_id=int(chunk_id),
        scientific_digest=str(scientific_digest),
        execution_digest=str(execution_digest),
        qspace_plan_digest=str(qspace_plan_digest),
        backend_policy_digest=str(backend_policy_digest),
        source_structure_digest=str(source_structure_digest),
        contribution_reciprocal_points=int(contribution_reciprocal_points),
        runtime_provenance=_resolve_runtime_provenance(runtime_provenance),
        payload_path=relative_to_output(payload_path, output_dir=output_dir),
        payload_sha256=_payload_digest(
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
            SCATTERING_STAGE,
            int(chunk_id),
            work_unit_digest,
            attempt_id,
        ),
        manifest,
        output_dir=output_dir,
    )
    return manifest


def _attempt_manifest_paths(output_dir: str | Path, run_digest: str, chunk_id: int) -> list[Path]:
    root = chunk_root(output_dir, run_digest, SCATTERING_STAGE, chunk_id) / "attempts"
    if not root.exists():
        return []
    return sorted(root.glob("*/*/attempt_*/attempt.json"))


def discover_scattering_attempts(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> tuple[ScatteringAttemptManifest, ...]:
    return tuple(
        read_manifest(path, codec=ScatteringAttemptManifest, output_dir=output_dir)
        for path in _attempt_manifest_paths(output_dir, run_digest, chunk_id)
    )


def load_scattering_attempt_partial(
    manifest: ScatteringAttemptManifest,
    *,
    output_dir: str | Path,
):
    payload_path = Path(output_dir) / manifest.payload_path
    datasets, attrs = _read_payload(payload_path)
    if file_sha256(payload_path) != manifest.file_sha256:
        raise ValueError(f"Attempt file hash mismatch: {manifest.payload_path}")
    if int(payload_path.stat().st_size) != int(manifest.payload_nbytes):
        raise ValueError(f"Attempt payload byte-size mismatch: {manifest.payload_path}")
    expected_payload_sha = _payload_digest(
        expected_set_digest=manifest.work_unit_digest,
        datasets=datasets,
        attrs=attrs,
    )
    if expected_payload_sha != manifest.payload_sha256:
        raise ValueError(f"Attempt payload hash mismatch: {manifest.payload_path}")
    return build_scattering_partial_result(
        chunk_id=manifest.chunk_id,
        interval_id=manifest.interval_id,
        point_ids=np.asarray(datasets["point_ids"], dtype=np.int64),
        grid_shape_nd=np.asarray(datasets["grid_shape_nd"], dtype=np.int64),
        amplitudes_delta=np.asarray(datasets["amplitudes_delta"], dtype=np.complex128),
        amplitudes_average=np.asarray(datasets["amplitudes_average"], dtype=np.complex128),
        reciprocal_point_count=int(manifest.contribution_reciprocal_points),
    )


def _attempt_identity_tuple(manifest: ScatteringAttemptManifest) -> tuple[str, ...]:
    return (
        manifest.run_digest,
        manifest.scientific_digest,
        manifest.execution_digest,
        manifest.qspace_plan_digest,
        manifest.backend_policy_digest,
        manifest.source_structure_digest,
    )


def _expected_work_unit_digest(manifest: ScatteringAttemptManifest) -> str:
    return build_scattering_work_unit_digest(
        interval_id=manifest.interval_id,
        chunk_id=manifest.chunk_id,
        scientific_digest=manifest.scientific_digest,
        execution_digest=manifest.execution_digest,
        qspace_plan_digest=manifest.qspace_plan_digest,
        backend_policy_digest=manifest.backend_policy_digest,
        source_structure_digest=manifest.source_structure_digest,
    )


def _validate_attempt_manifests(
    attempts: tuple[ScatteringAttemptManifest, ...],
    *,
    expected_interval_ids: tuple[int, ...],
    output_dir: str | Path,
) -> None:
    expected_set = {int(item) for item in expected_interval_ids}
    unexpected = sorted(
        {
            int(attempt.interval_id)
            for attempt in attempts
            if int(attempt.interval_id) not in expected_set
        }
    )
    if unexpected:
        raise RuntimeError(
            "Scattering commit candidate found attempts for unexpected intervals: "
            + ", ".join(str(item) for item in unexpected)
        )

    identities = {_attempt_identity_tuple(attempt) for attempt in attempts}
    if len(identities) > 1:
        raise RuntimeError("Conflicting scattering attempt identity for chunk commit.")

    for attempt in attempts:
        expected_digest = _expected_work_unit_digest(attempt)
        if attempt.work_unit_digest != expected_digest:
            raise RuntimeError(
                "Scattering attempt work-unit digest mismatch for "
                f"interval {int(attempt.interval_id)}."
            )
        load_scattering_attempt_partial(attempt, output_dir=output_dir)


def _select_attempts_by_interval(
    attempts: tuple[ScatteringAttemptManifest, ...],
    *,
    expected_interval_ids: tuple[int, ...],
) -> tuple[ScatteringAttemptManifest, ...]:
    selected: list[ScatteringAttemptManifest] = []
    by_interval: dict[int, list[ScatteringAttemptManifest]] = {}
    for attempt in attempts:
        by_interval.setdefault(int(attempt.interval_id), []).append(attempt)
    for interval_id in expected_interval_ids:
        candidates = by_interval.get(int(interval_id), [])
        if not candidates:
            raise RuntimeError(
                f"Scattering commit candidate missing attempts for interval {int(interval_id)}."
            )
        payload_hashes = {candidate.payload_sha256 for candidate in candidates}
        if len(payload_hashes) != 1:
            raise RuntimeError(
                f"Conflicting scattering attempts for interval {int(interval_id)}."
            )
        identities = {_attempt_identity_tuple(candidate) for candidate in candidates}
        if len(identities) != 1:
            raise RuntimeError(
                f"Conflicting scattering attempt identity for interval {int(interval_id)}."
            )
        selected.append(sorted(candidates, key=lambda item: (item.attempt_id, item.payload_path))[0])
    return tuple(selected)


def _candidate_id(
    *,
    chunk_id: int,
    selected_attempts: tuple[ScatteringAttemptManifest, ...],
) -> str:
    return digest_dict(
        {
            "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
            "chunk_id": int(chunk_id),
            "attempt_payloads": [
                {
                    "interval_id": attempt.interval_id,
                    "work_unit_digest": attempt.work_unit_digest,
                    "payload_sha256": attempt.payload_sha256,
                }
                for attempt in selected_attempts
            ],
        },
        domain="mosaic.scattering.commit_candidate_id.v1",
    )[:32]


def create_scattering_commit_candidate(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
    expected_interval_ids: tuple[int, ...],
) -> ScatteringCommitCandidateManifest:
    expected = tuple(sorted(int(item) for item in expected_interval_ids))
    if not expected:
        raise ValueError("Scattering commit candidates require expected interval IDs.")
    attempts = discover_scattering_attempts(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_id=chunk_id,
    )
    _validate_attempt_manifests(
        attempts,
        expected_interval_ids=expected,
        output_dir=output_dir,
    )
    selected = _select_attempts_by_interval(attempts, expected_interval_ids=expected)
    merged = None
    for attempt in selected:
        partial = load_scattering_attempt_partial(attempt, output_dir=output_dir)
        merged = partial if merged is None else merge_scattering_partial_results(merged, partial)
    if merged is None:
        raise RuntimeError("No scattering attempts selected for commit candidate.")
    candidate_id = _candidate_id(chunk_id=chunk_id, selected_attempts=selected)
    datasets = _payload_datasets(
        point_ids=merged.point_ids,
        grid_shape_nd=merged.grid_shape_nd,
        amplitudes_delta=merged.amplitudes_delta,
        amplitudes_average=merged.amplitudes_average,
    )
    attrs = _candidate_payload_attrs(
        run_digest=run_digest,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        reciprocal_point_count=merged.reciprocal_point_count,
    )
    payload_path = commit_candidate_payload_path(
        output_dir,
        run_digest,
        SCATTERING_STAGE,
        int(chunk_id),
        candidate_id,
    )
    _atomic_hdf5_write(payload_path, datasets, attrs=attrs)
    manifest = ScatteringCommitCandidateManifest(
        run_digest=str(run_digest),
        candidate_id=candidate_id,
        chunk_id=int(chunk_id),
        contributing_interval_ids=tuple(int(item) for item in merged.contributing_interval_ids),
        selected_attempts=tuple(
            {
                "interval_id": int(attempt.interval_id),
                "attempt_id": attempt.attempt_id,
                "work_unit_digest": attempt.work_unit_digest,
                "payload_sha256": attempt.payload_sha256,
                "payload_path": attempt.payload_path,
            }
            for attempt in selected
        ),
        payload_path=relative_to_output(payload_path, output_dir=output_dir),
        payload_sha256=_payload_digest(
            expected_set_digest=candidate_id,
            datasets=datasets,
            attrs=attrs,
        ),
        file_sha256=file_sha256(payload_path),
        payload_nbytes=int(payload_path.stat().st_size),
        reciprocal_point_count=int(merged.reciprocal_point_count),
    )
    write_manifest(
        commit_candidate_manifest_path(
            output_dir,
            run_digest,
            SCATTERING_STAGE,
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
    root = commit_attempts_root(output_dir, run_digest, SCATTERING_STAGE, chunk_id)
    if not root.exists():
        return []
    return sorted(root.glob("commit_*/commit_candidate.json"))


def discover_scattering_commit_candidates(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> tuple[ScatteringCommitCandidateManifest, ...]:
    return tuple(
        read_manifest(path, codec=ScatteringCommitCandidateManifest, output_dir=output_dir)
        for path in _commit_candidate_manifest_paths(output_dir, run_digest, chunk_id)
    )


def _record_scattering_commit_scan_seconds(
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
            f"{SCATTERING_STAGE}/chunk_{int(chunk_id)}": float(scan_seconds)
        },
    )


def _load_scattering_candidate_payload(
    candidate: ScatteringCommitCandidateManifest,
    *,
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload_path = Path(output_dir) / candidate.payload_path
    datasets, attrs = _read_payload(payload_path)
    if file_sha256(payload_path) != candidate.file_sha256:
        raise ValueError(f"Scattering candidate file hash mismatch: {candidate.payload_path}")
    if int(payload_path.stat().st_size) != int(candidate.payload_nbytes):
        raise ValueError(f"Scattering candidate byte-size mismatch: {candidate.payload_path}")
    expected_payload_sha = _payload_digest(
        expected_set_digest=candidate.candidate_id,
        datasets=datasets,
        attrs=attrs,
    )
    if expected_payload_sha != candidate.payload_sha256:
        raise ValueError(f"Scattering candidate payload hash mismatch: {candidate.payload_path}")
    return datasets, attrs


def validate_scattering_commit_candidate(
    candidate: ScatteringCommitCandidateManifest,
    *,
    output_dir: str | Path,
) -> ScatteringCommitCandidateManifest:
    expected_payload_path = relative_to_output(
        commit_candidate_payload_path(
            output_dir,
            candidate.run_digest,
            SCATTERING_STAGE,
            int(candidate.chunk_id),
            candidate.candidate_id,
        ),
        output_dir=output_dir,
    )
    if candidate.payload_path != expected_payload_path:
        raise RuntimeError(
            "Scattering commit candidate payload path does not match deterministic path."
        )
    selected_intervals = tuple(
        int(item["interval_id"]) for item in candidate.selected_attempts
    )
    if tuple(sorted(selected_intervals)) != tuple(sorted(candidate.contributing_interval_ids)):
        raise RuntimeError("Scattering commit candidate interval coverage is inconsistent.")
    attempts_by_key = {
        (attempt.work_unit_digest, attempt.attempt_id): attempt
        for attempt in discover_scattering_attempts(
            output_dir=output_dir,
            run_digest=candidate.run_digest,
            chunk_id=int(candidate.chunk_id),
        )
    }
    for selected in candidate.selected_attempts:
        key = (str(selected["work_unit_digest"]), str(selected["attempt_id"]))
        attempt = attempts_by_key.get(key)
        if attempt is None:
            raise RuntimeError("Scattering commit candidate references a missing attempt.")
        if int(selected["interval_id"]) != int(attempt.interval_id):
            raise RuntimeError("Scattering commit candidate attempt interval mismatch.")
        if str(selected["payload_sha256"]) != attempt.payload_sha256:
            raise RuntimeError("Scattering commit candidate attempt payload hash mismatch.")
        if str(selected["payload_path"]) != attempt.payload_path:
            raise RuntimeError("Scattering commit candidate attempt payload path mismatch.")
        load_scattering_attempt_partial(attempt, output_dir=output_dir)
    _load_scattering_candidate_payload(candidate, output_dir=output_dir)
    return candidate


def promote_scattering_chunk_commit_by_scan(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_id: int,
) -> ScatteringChunkCommitManifest:
    scan_started = time.perf_counter()
    candidates = discover_scattering_commit_candidates(
        output_dir=output_dir,
        run_digest=run_digest,
        chunk_id=int(chunk_id),
    )
    valid: list[ScatteringCommitCandidateManifest] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            valid.append(validate_scattering_commit_candidate(candidate, output_dir=output_dir))
        except Exception as exc:
            errors.append(f"{candidate.candidate_id}: {type(exc).__name__}: {exc}")
    if not valid:
        detail = "; ".join(errors) if errors else "no candidate manifests found"
        raise RuntimeError(
            f"No valid scattering commit candidates for chunk {int(chunk_id)}: {detail}"
        )
    payload_hashes = {candidate.payload_sha256 for candidate in valid}
    if len(payload_hashes) != 1:
        raise RuntimeError(
            "Conflicting valid scattering commit candidates for chunk "
            f"{int(chunk_id)}: "
            + ", ".join(
                f"{candidate.candidate_id}:{candidate.payload_sha256}"
                for candidate in sorted(valid, key=lambda item: item.candidate_id)
            )
        )
    candidate = sorted(valid, key=lambda item: item.candidate_id)[0]
    candidate_manifest = commit_candidate_manifest_path(
        output_dir,
        candidate.run_digest,
        SCATTERING_STAGE,
        int(candidate.chunk_id),
        candidate.candidate_id,
    )
    target = chunk_commit_path(output_dir, candidate.run_digest, SCATTERING_STAGE, candidate.chunk_id)
    manifest = ScatteringChunkCommitManifest(
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
        existing = read_manifest(target, codec=ScatteringChunkCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError(
                f"Chunk {candidate.chunk_id} already committed to a different candidate."
            )
        _record_scattering_commit_scan_seconds(
            output_dir=output_dir,
            run_digest=candidate.run_digest,
            chunk_id=int(candidate.chunk_id),
            scan_seconds=time.perf_counter() - scan_started,
        )
        return existing
    write_manifest(target, manifest, output_dir=output_dir)
    _record_scattering_commit_scan_seconds(
        output_dir=output_dir,
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
        scan_seconds=time.perf_counter() - scan_started,
    )
    return manifest


def promote_scattering_chunk_commit(
    *,
    output_dir: str | Path,
    candidate: ScatteringCommitCandidateManifest,
) -> ScatteringChunkCommitManifest:
    validate_scattering_commit_candidate(candidate, output_dir=output_dir)
    candidate_manifest = commit_candidate_manifest_path(
        output_dir,
        candidate.run_digest,
        SCATTERING_STAGE,
        int(candidate.chunk_id),
        candidate.candidate_id,
    )
    target = chunk_commit_path(output_dir, candidate.run_digest, SCATTERING_STAGE, candidate.chunk_id)
    manifest = ScatteringChunkCommitManifest(
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
        existing = read_manifest(target, codec=ScatteringChunkCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError(
                f"Chunk {candidate.chunk_id} already committed to a different candidate."
            )
        return existing
    return promote_scattering_chunk_commit_by_scan(
        output_dir=output_dir,
        run_digest=candidate.run_digest,
        chunk_id=int(candidate.chunk_id),
    )


def _normalize_expected_by_chunk(
    expected_by_chunk: Mapping[int, tuple[int, ...] | list[int]],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for chunk_id, interval_ids in expected_by_chunk.items():
        expected_interval_ids = sorted({int(item) for item in interval_ids})
        if not expected_interval_ids:
            raise ValueError(
                f"Scattering stage plan chunk {int(chunk_id)} has no expected intervals."
            )
        normalized.append(
            {
                "chunk_id": int(chunk_id),
                "expected_interval_ids": expected_interval_ids,
            }
        )
    return tuple(sorted(normalized, key=lambda item: int(item["chunk_id"])))


def _build_stage_plan_digest(
    *,
    run_digest: str,
    expected_by_chunk: tuple[dict[str, Any], ...],
) -> str:
    return digest_dict(
        {
            "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
            "run_digest": str(run_digest),
            "stage": SCATTERING_STAGE,
            "expected_by_chunk": expected_by_chunk,
        },
        domain="mosaic.scattering.stage_plan.v1",
    )


def write_scattering_stage_plan(
    *,
    output_dir: str | Path,
    run_digest: str,
    expected_by_chunk: Mapping[int, tuple[int, ...] | list[int]],
) -> ScatteringStagePlanManifest:
    expected = _normalize_expected_by_chunk(expected_by_chunk)
    manifest = ScatteringStagePlanManifest(
        run_digest=str(run_digest),
        expected_by_chunk=expected,
        stage_plan_digest=_build_stage_plan_digest(
            run_digest=str(run_digest),
            expected_by_chunk=expected,
        ),
    )
    target = stage_plan_path(output_dir, run_digest, SCATTERING_STAGE)
    if target.exists():
        existing = read_manifest(
            target,
            codec=ScatteringStagePlanManifest,
            output_dir=output_dir,
        )
        if existing != manifest:
            raise RuntimeError(
                "Scattering stage_plan.json already exists with different expected coverage."
            )
        return existing
    write_manifest(target, manifest, output_dir=output_dir)
    return manifest


def _load_scattering_stage_plan_if_present(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> ScatteringStagePlanManifest | None:
    path = stage_plan_path(output_dir, run_digest, SCATTERING_STAGE)
    if not path.exists():
        return None
    return read_manifest(path, codec=ScatteringStagePlanManifest, output_dir=output_dir)


def _committed_chunk_ids_on_disk(
    *,
    output_dir: str | Path,
    run_digest: str,
) -> tuple[int, ...]:
    chunks_dir = stage_root(output_dir, run_digest, SCATTERING_STAGE) / "chunks"
    if not chunks_dir.exists():
        return ()
    chunk_ids: list[int] = []
    for path in sorted(chunks_dir.glob("chunk_*/chunk_commit.json")):
        try:
            chunk_ids.append(int(path.parent.name.removeprefix("chunk_")))
        except ValueError:
            raise RuntimeError(f"Invalid scattering chunk commit directory: {path.parent.name!r}")
    return tuple(sorted(chunk_ids))


def _validate_scattering_chunk_commit(
    commit: ScatteringChunkCommitManifest,
    *,
    output_dir: str | Path,
    run_digest: str,
) -> ScatteringChunkCommitManifest:
    if commit.run_digest != str(run_digest):
        raise RuntimeError("Scattering chunk commit run identity mismatch.")
    candidate = read_manifest(
        Path(output_dir) / commit.candidate_manifest_path,
        codec=ScatteringCommitCandidateManifest,
        output_dir=output_dir,
    )
    validate_scattering_commit_candidate(candidate, output_dir=output_dir)
    if candidate.candidate_id != commit.selected_candidate_id:
        raise RuntimeError("Scattering chunk commit selected candidate mismatch.")
    if candidate.payload_path != commit.candidate_payload_path:
        raise RuntimeError("Scattering chunk commit payload path mismatch.")
    if candidate.payload_sha256 != commit.payload_sha256:
        raise RuntimeError("Scattering chunk commit payload hash mismatch.")
    if candidate.file_sha256 != commit.file_sha256:
        raise RuntimeError("Scattering chunk commit file hash mismatch.")
    if int(candidate.payload_nbytes) != int(commit.payload_nbytes):
        raise RuntimeError("Scattering chunk commit byte-size mismatch.")
    return commit


def write_scattering_stage_commit(
    *,
    output_dir: str | Path,
    run_digest: str,
    chunk_ids: tuple[int, ...],
) -> ScatteringStageCommitManifest:
    requested_chunk_ids = tuple(sorted(int(item) for item in chunk_ids))
    stage_plan = _load_scattering_stage_plan_if_present(
        output_dir=output_dir,
        run_digest=run_digest,
    )
    if stage_plan is not None:
        expected_chunk_ids = tuple(
            int(item["chunk_id"]) for item in stage_plan.expected_by_chunk
        )
        if requested_chunk_ids != expected_chunk_ids:
            raise RuntimeError(
                "Scattering stage_commit chunk IDs do not match stage_plan.json "
                f"coverage: requested={requested_chunk_ids}, expected={expected_chunk_ids}."
            )
        expected_digest = _build_stage_plan_digest(
            run_digest=str(run_digest),
            expected_by_chunk=stage_plan.expected_by_chunk,
        )
        if stage_plan.stage_plan_digest != expected_digest:
            raise RuntimeError("Scattering stage_plan.json digest does not match coverage.")

    committed_on_disk = _committed_chunk_ids_on_disk(
        output_dir=output_dir,
        run_digest=run_digest,
    )
    if committed_on_disk != requested_chunk_ids:
        raise RuntimeError(
            "Scattering stage_commit coverage must match committed chunk manifests: "
            f"requested={requested_chunk_ids}, committed={committed_on_disk}."
        )

    chunk_commits: list[ScatteringChunkCommitManifest] = []
    chunk_commit_paths: list[str] = []
    for chunk_id in requested_chunk_ids:
        path = chunk_commit_path(output_dir, run_digest, SCATTERING_STAGE, chunk_id)
        if not path.exists():
            raise RuntimeError(f"Missing scattering chunk_commit.json for chunk {chunk_id}.")
        chunk_commits.append(
            _validate_scattering_chunk_commit(
                read_manifest(path, codec=ScatteringChunkCommitManifest, output_dir=output_dir),
                output_dir=output_dir,
                run_digest=run_digest,
            )
        )
        chunk_commit_paths.append(relative_to_output(path, output_dir=output_dir))
    stage_digest = digest_dict(
        {
            "schema_version": SCATTERING_COMMIT_SCHEMA_VERSION,
            "stage_plan_digest": None if stage_plan is None else stage_plan.stage_plan_digest,
            "chunk_commits": [commit.to_payload() for commit in chunk_commits],
        },
        domain="mosaic.scattering.stage_commit.v1",
    )
    manifest = ScatteringStageCommitManifest(
        run_digest=str(run_digest),
        chunk_ids=requested_chunk_ids,
        chunk_commit_paths=tuple(chunk_commit_paths),
        stage_digest=stage_digest,
        stage_plan_digest=None if stage_plan is None else stage_plan.stage_plan_digest,
    )
    target = stage_commit_path(output_dir, run_digest, SCATTERING_STAGE)
    if target.exists():
        existing = read_manifest(target, codec=ScatteringStageCommitManifest, output_dir=output_dir)
        if existing != manifest:
            raise RuntimeError("Scattering stage_commit.json already exists with different identity.")
        return existing
    write_manifest(
        target,
        manifest,
        output_dir=output_dir,
    )
    return manifest


__all__ = [
    "SCATTERING_ATTEMPT_SCHEMA",
    "SCATTERING_CHUNK_COMMIT_SCHEMA",
    "SCATTERING_COMMIT_CANDIDATE_SCHEMA",
    "SCATTERING_STAGE_COMMIT_SCHEMA",
    "SCATTERING_STAGE_PLAN_SCHEMA",
    "ScatteringAttemptManifest",
    "ScatteringChunkCommitManifest",
    "ScatteringCommitCandidateManifest",
    "ScatteringStageCommitManifest",
    "ScatteringStagePlanManifest",
    "build_scattering_work_unit_digest",
    "create_scattering_commit_candidate",
    "discover_scattering_attempts",
    "discover_scattering_commit_candidates",
    "load_scattering_attempt_partial",
    "promote_scattering_chunk_commit_by_scan",
    "promote_scattering_chunk_commit",
    "validate_scattering_commit_candidate",
    "write_scattering_attempt",
    "write_scattering_stage_commit",
    "write_scattering_stage_plan",
]
