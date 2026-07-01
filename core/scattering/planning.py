from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping
from pathlib import Path

import numpy as np

from core.qspace.normalization import (
    QNormalizationContract,
    write_q_normalization_sidecar,
)
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.half_space import (
    classify_interval_half_space_role,
    half_space_role_multiplicity,
)
from core.scattering.kernels import reciprocal_space_points_counter, to_interval_dict
from core.scattering.grid import generate_q_space_grid_sync
from core.storage.attempt_store import qspace_plan_path, run_manifest_path
from core.storage.digests import (
    build_execution_digest as _build_execution_digest,
    build_run_digest as _build_run_digest,
    build_run_identity_digest as _build_run_identity_digest,
    digest_dict,
    normalize_digest_input,
    require_sha256_hex,
)
from core.storage.fingerprint import file_sha256, payload_sha256
from core.storage.manifest import write_manifest


logger = logging.getLogger(__name__)

SCATTERING_IDENTITY_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA = "mosaic.run_manifest"
QSPACE_PLAN_SCHEMA = "mosaic.qspace_plan"

_SCIENTIFIC_KEYS = (
    "structure_content_digest",
    "structure_file_sha256",
    "structure_digest",
    "supercell",
    "vectors",
    "reciprocal_space_intervals",
    "reciprocal_space_intervals_all",
    "mask",
    "mask_parameters",
    "MaskStrategyParameters",
    "mask_strategy",
    "charge",
    "use_coeff",
    "coeff_val",
    "coefficients",
    "point_chunk_size",
    "point_chunks",
    "postprocessing_mode",
    "scattering_weights_digest",
    "scattering_calculator_version",
)


@dataclass(frozen=True)
class ScatteringExecutionPlan:
    interval_work_units: tuple[ScatteringWorkUnit, ...]
    chunk_work_units: tuple[ScatteringWorkUnit, ...]
    chunk_ids: tuple[int, ...]
    total_reciprocal_points: int


@dataclass(frozen=True)
class ScatteringRunIdentity:
    scientific_digest: str
    execution_digest: str
    run_digest: str


@dataclass(frozen=True)
class ScatteringWorkIdentity:
    scientific_digest: str
    execution_digest: str
    run_digest: str
    qspace_plan_digest: str
    backend_policy_digest: str
    source_structure_digest: str

    def __post_init__(self) -> None:
        require_sha256_hex(self.scientific_digest, field_name="scientific_digest")
        require_sha256_hex(self.execution_digest, field_name="execution_digest")
        require_sha256_hex(self.qspace_plan_digest, field_name="qspace_plan_digest")
        require_sha256_hex(self.backend_policy_digest, field_name="backend_policy_digest")
        require_sha256_hex(
            self.source_structure_digest,
            field_name="source_structure_digest",
        )
        if not self.run_digest:
            raise ValueError("run_digest must not be empty.")

    def to_work_unit_kwargs(self) -> dict[str, str]:
        return {
            "scientific_digest": self.scientific_digest,
            "execution_digest": self.execution_digest,
            "run_digest": self.run_digest,
            "qspace_plan_digest": self.qspace_plan_digest,
            "backend_policy_digest": self.backend_policy_digest,
            "source_structure_digest": self.source_structure_digest,
        }


@dataclass(frozen=True)
class RunManifest:
    scientific_digest: str
    execution_digest: str
    run_digest: str
    execution_contract: Mapping[str, Any]
    schema: str = RUN_MANIFEST_SCHEMA
    schema_version: int = SCATTERING_IDENTITY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "scientific_digest": self.scientific_digest,
            "execution_digest": self.execution_digest,
            "run_digest": self.run_digest,
            "execution_contract": normalize_digest_input(dict(self.execution_contract)),
        }


@dataclass(frozen=True)
class QSpaceIntervalPlan:
    interval_id: int
    h_bounds: tuple[float, float]
    k_bounds: tuple[float, float]
    l_bounds: tuple[float, float]
    half_space_role: str
    reciprocal_multiplicity: int
    reciprocal_point_count: int
    q_grid_digest: str
    mask_digest: str
    l_coverage: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "interval_id": int(self.interval_id),
            "h_bounds": list(self.h_bounds),
            "k_bounds": list(self.k_bounds),
            "l_bounds": list(self.l_bounds),
            "half_space_role": self.half_space_role,
            "reciprocal_multiplicity": int(self.reciprocal_multiplicity),
            "reciprocal_point_count": int(self.reciprocal_point_count),
            "q_grid_digest": self.q_grid_digest,
            "mask_digest": self.mask_digest,
            "l_coverage": self.l_coverage,
        }


@dataclass(frozen=True)
class QSpacePlan:
    run_digest: str
    scientific_digest: str
    execution_digest: str
    mask_digest: str
    intervals: tuple[QSpaceIntervalPlan, ...]
    q_grid_set_digest: str
    schema: str = QSPACE_PLAN_SCHEMA
    schema_version: int = SCATTERING_IDENTITY_SCHEMA_VERSION
    # NOT part of the persisted qspace_plan.json payload (see to_payload below) and so
    # NOT part of qspace_plan_digest. These multiplicity-free planned/accepted contracts
    # ride alongside the plan in memory and are written to the SEPARATE sidecar
    # q_normalization.json by prepare_scattering_run_identity. Keeping them off
    # to_payload is what guarantees the plan file's bytes -- hence its digest -- are
    # unchanged by this feature.
    q_normalization_contracts: tuple[QNormalizationContract, ...] = field(
        default=(), compare=False
    )

    def to_payload(self) -> dict[str, Any]:
        # Deliberately EXCLUDES q_normalization_contracts: the q-normalization data is
        # persisted to the q_normalization.json sidecar, never into this identity-bearing
        # file. Adding a field here would change qspace_plan_digest.
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "scientific_digest": self.scientific_digest,
            "execution_digest": self.execution_digest,
            "mask_digest": self.mask_digest,
            "q_grid_set_digest": self.q_grid_set_digest,
            "intervals": [interval.to_payload() for interval in self.intervals],
        }


def _normalize_interval_for_identity(interval: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "id" in interval:
        payload["id"] = int(interval["id"])
    payload.update(to_interval_dict(dict(interval)))
    for key in sorted(interval):
        if key in payload or key.endswith("_range") or key.endswith("_start") or key.endswith("_end"):
            continue
        if key == "id":
            continue
        payload[str(key)] = interval[key]
    return normalize_digest_input(payload)


def _normalize_intervals_for_identity(intervals: list[dict]) -> list[dict[str, Any]]:
    return sorted(
        (_normalize_interval_for_identity(interval) for interval in intervals),
        key=lambda item: int(item.get("id", 0)),
    )


def _half_space_metadata_for_identity(
    intervals: list[dict],
    supercell: Any,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for index, interval in enumerate(_normalize_intervals_for_identity(intervals)):
        role = classify_interval_half_space_role(interval, np.asarray(supercell))
        metadata.append(
            {
                "interval_id": int(interval.get("id", index)),
                "half_space_role": role,
                "reciprocal_multiplicity": int(half_space_role_multiplicity(role)),
            }
        )
    return metadata


def build_scientific_identity_payload(parameters: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": SCATTERING_IDENTITY_SCHEMA_VERSION}
    for key in _SCIENTIFIC_KEYS:
        if key not in parameters:
            continue
        value = parameters[key]
        if key in {"reciprocal_space_intervals", "reciprocal_space_intervals_all"}:
            value = _normalize_intervals_for_identity(list(value))
        elif key == "mask_strategy":
            value = _mask_strategy_name(value)
        payload[key] = normalize_digest_input(value)
    identity_intervals = parameters.get("reciprocal_space_intervals_all")
    if identity_intervals is None:
        identity_intervals = parameters.get("reciprocal_space_intervals")
    if identity_intervals is not None and parameters.get("supercell") is not None:
        payload["half_space_metadata"] = normalize_digest_input(
            _half_space_metadata_for_identity(
                list(identity_intervals),
                parameters["supercell"],
            )
        )
    return payload


def build_scientific_digest(parameters: Mapping[str, Any]) -> str:
    return digest_dict(
        build_scientific_identity_payload(parameters),
        domain="mosaic.scattering.scientific.v1",
    )


def build_execution_digest(
    *,
    scientific_digest: str,
    backend: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    backend_policy_digest: str | None = None,
) -> str:
    return _build_execution_digest(
        scientific_digest=scientific_digest,
        backend=backend,
        eps=eps,
        dtype=dtype,
        pre_sum_mode=pre_sum_mode,
        reducer_strategy=reducer_strategy,
        backend_policy_digest=backend_policy_digest,
        schema_version=SCATTERING_IDENTITY_SCHEMA_VERSION,
        domain="mosaic.scattering.execution.v1",
    )


def build_run_digest(execution_digest: str) -> str:
    return _build_run_digest(
        execution_digest,
        schema_version=SCATTERING_IDENTITY_SCHEMA_VERSION,
        domain="mosaic.run.v1",
        length=32,
    )


def build_run_identity(
    parameters: Mapping[str, Any],
    *,
    backend: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    backend_policy_digest: str | None = None,
) -> ScatteringRunIdentity:
    scientific_digest = build_scientific_digest(parameters)
    execution_digest = build_execution_digest(
        scientific_digest=scientific_digest,
        backend=backend,
        eps=eps,
        dtype=dtype,
        pre_sum_mode=pre_sum_mode,
        reducer_strategy=reducer_strategy,
        backend_policy_digest=backend_policy_digest,
    )
    # Device-independent identity: the run/checkpoint tree is addressed by the DEVICE-INDEPENDENT run
    # identity (science + numerical contract: eps/dtype/pre-sum/reducer), NOT by the
    # device-bound execution_digest. A CPU run and a GPU run of the same science thus
    # share one `.mosaic/runs/<run_digest>/` tree; the device-bound execution_digest
    # is retained on the identity as metadata (and in the run manifest's execution
    # contract), never as the path address.
    run_digest = _build_run_identity_digest(
        scientific_digest=scientific_digest,
        eps=eps,
        dtype=dtype,
        pre_sum_mode=pre_sum_mode,
        reducer_strategy=reducer_strategy,
        schema_version=SCATTERING_IDENTITY_SCHEMA_VERSION,
    )
    return ScatteringRunIdentity(
        scientific_digest=scientific_digest,
        execution_digest=execution_digest,
        run_digest=run_digest,
    )


def build_source_structure_digest(parameters: Mapping[str, Any]) -> str:
    for key in (
        "structure_content_digest",
        "structure_file_sha256",
        "structure_digest",
    ):
        value = parameters.get(key)
        if value:
            return require_sha256_hex(str(value), field_name=key)

    source_payload = {
        "schema_version": SCATTERING_IDENTITY_SCHEMA_VERSION,
        "original_coords": parameters.get("original_coords"),
        "average_coords": parameters.get("average_coords"),
        "cells_origin": parameters.get("cells_origin"),
        "elements": parameters.get("elements"),
        "refnumbers": parameters.get("refnumbers"),
        "vectors": parameters.get("vectors"),
        "supercell": parameters.get("supercell"),
        "coeff": parameters.get("coeff"),
    }
    return digest_dict(
        normalize_digest_input(source_payload),
        domain="mosaic.scattering.source_structure.v1",
    )


def build_scattering_backend_policy_digest(
    *,
    backend: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    scheduler_kind: str,
    interval_artifact_policy: str,
    deterministic_mode: str = "stable-v1",
    thread_count: int | None = None,
    requested_nufft_policy: str | None = None,
    execution_nufft_policy: str | None = None,
) -> str:
    return digest_dict(
        {
            "schema_version": SCATTERING_IDENTITY_SCHEMA_VERSION,
            "backend": str(backend),
            "eps": float(eps),
            "dtype": str(dtype),
            "pre_sum_mode": str(pre_sum_mode),
            "reducer_strategy": str(reducer_strategy),
            "scheduler_kind": str(scheduler_kind),
            "interval_artifact_policy": str(interval_artifact_policy),
            "deterministic_mode": str(deterministic_mode),
            "thread_count": None if thread_count is None else int(thread_count),
            "requested_nufft_policy": (
                None if requested_nufft_policy is None else str(requested_nufft_policy)
            ),
            "execution_nufft_policy": (
                None if execution_nufft_policy is None else str(execution_nufft_policy)
            ),
        },
        domain="mosaic.scattering.backend_policy.v1",
    )


def prepare_scattering_run_identity(
    *,
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    B_: np.ndarray,
    mask_params: Mapping[str, Any] | None,
    MaskStrategy,
    backend: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
    reducer_strategy: str,
    scheduler_kind: str,
    interval_artifact_policy: str,
    deterministic_mode: str = "stable-v1",
    thread_count: int | None = None,
    requested_nufft_policy: str | None = None,
    execution_nufft_policy: str | None = None,
) -> ScatteringWorkIdentity:
    backend_policy_digest = build_scattering_backend_policy_digest(
        backend=backend,
        eps=eps,
        dtype=dtype,
        pre_sum_mode=pre_sum_mode,
        reducer_strategy=reducer_strategy,
        scheduler_kind=scheduler_kind,
        interval_artifact_policy=interval_artifact_policy,
        deterministic_mode=deterministic_mode,
        thread_count=thread_count,
        requested_nufft_policy=requested_nufft_policy,
        execution_nufft_policy=execution_nufft_policy,
    )
    identity_parameters = dict(parameters)
    identity_parameters.setdefault("mask_parameters", dict(mask_params or {}))
    identity_parameters.setdefault("MaskStrategyParameters", dict(mask_params or {}))
    identity_parameters.setdefault("mask_strategy", _mask_strategy_name(MaskStrategy))
    run_identity = build_run_identity(
        identity_parameters,
        backend=backend,
        eps=eps,
        dtype=dtype,
        pre_sum_mode=pre_sum_mode,
        reducer_strategy=reducer_strategy,
        backend_policy_digest=backend_policy_digest,
    )
    write_run_manifest(
        output_dir,
        run_identity,
        execution_contract={
            "backend": str(backend),
            "eps": float(eps),
            "dtype": str(dtype),
            "pre_sum_mode": str(pre_sum_mode),
            "reducer_strategy": str(reducer_strategy),
            "scheduler_kind": str(scheduler_kind),
            "interval_artifact_policy": str(interval_artifact_policy),
            "deterministic_mode": str(deterministic_mode),
            "thread_count": None if thread_count is None else int(thread_count),
            "requested_nufft_policy": (
                None if requested_nufft_policy is None else str(requested_nufft_policy)
            ),
            "execution_nufft_policy": (
                None if execution_nufft_policy is None else str(execution_nufft_policy)
            ),
            "backend_policy_digest": backend_policy_digest,
            "source_structure_digest": build_source_structure_digest(parameters),
        },
    )
    qspace_plan = build_qspace_plan(
        parameters=identity_parameters,
        identity=run_identity,
        B_=B_,
        mask_params=mask_params,
        MaskStrategy=MaskStrategy,
    )
    qspace_path = write_qspace_plan(output_dir, qspace_plan)
    # Persist the per-interval q-normalization contracts to the SEPARATE sidecar
    # (q_normalization.json) next to qspace_plan.json. This never touches the plan file,
    # so qspace_plan_digest (file_sha256 of qspace_path) below is unchanged.
    write_q_normalization_sidecar(
        output_dir,
        qspace_plan.run_digest,
        qspace_plan.q_normalization_contracts,
    )
    return ScatteringWorkIdentity(
        scientific_digest=run_identity.scientific_digest,
        execution_digest=run_identity.execution_digest,
        run_digest=run_identity.run_digest,
        qspace_plan_digest=file_sha256(qspace_path),
        backend_policy_digest=backend_policy_digest,
        source_structure_digest=build_source_structure_digest(parameters),
    )


def q_grid_sha256(q_grid: np.ndarray) -> str:
    return payload_sha256(
        schema="mosaic.q_grid.v1",
        expected_set_digest="q-grid",
        datasets={"q_grid": np.asarray(q_grid)},
        attrs={},
    )


def _mask_strategy_name(mask_strategy) -> str:
    if mask_strategy is None:
        return "none"
    if isinstance(mask_strategy, str):
        return mask_strategy
    return type(mask_strategy).__name__


def build_mask_digest(mask_params: Mapping[str, Any] | None, mask_strategy) -> str:
    return digest_dict(
        {
            "schema_version": SCATTERING_IDENTITY_SCHEMA_VERSION,
            "mask_params": dict(mask_params or {}),
            "mask_strategy": _mask_strategy_name(mask_strategy),
        },
        domain="mosaic.qspace.mask.v1",
    )


def _axis_bounds(interval_dict: Mapping[str, float], axis: str) -> tuple[float, float]:
    return (
        float(interval_dict.get(f"{axis}_start", 0.0)),
        float(interval_dict.get(f"{axis}_end", 0.0)),
    )


def _l_coverage_for_role(role: str) -> str:
    if role == "zero_plane":
        return "L=0"
    if role == "positive_half":
        return "positive-L"
    return "full"


def build_qspace_plan(
    *,
    parameters: Mapping[str, Any],
    identity: ScatteringRunIdentity,
    B_: np.ndarray,
    mask_params: Mapping[str, Any] | None,
    MaskStrategy,
) -> QSpacePlan:
    supercell = np.asarray(parameters["supercell"])
    intervals = list(parameters["reciprocal_space_intervals"])
    mask_digest = build_mask_digest(mask_params, MaskStrategy)
    interval_plans: list[QSpaceIntervalPlan] = []
    normalization_contracts: list[QNormalizationContract] = []
    for interval in sorted(intervals, key=lambda item: int(item["id"])):
        if "id" not in interval:
            raise ValueError("qspace_plan intervals must include stable interval IDs.")
        interval_dict = to_interval_dict(dict(interval))
        role = classify_interval_half_space_role(interval, supercell)
        multiplicity = int(half_space_role_multiplicity(role))
        q_grid = generate_q_space_grid_sync(
            dict(interval),
            np.asarray(B_),
            dict(mask_params or {}),
            MaskStrategy,
            supercell,
        )
        # The masked q_grid is already in scope -- its row count is the
        # multiplicity-FREE accepted (post-mask) count.
        accepted_count = int(np.asarray(q_grid).shape[0])
        # Multiplicity-FREE dense planned count (mask-blind). This is comparable to
        # accepted_count; the multiplicity-FOLDED value still feeds the persisted
        # QSpaceIntervalPlan.reciprocal_point_count below (byte-identical, unchanged).
        planned_count = int(
            reciprocal_space_points_counter(
                interval_dict, supercell, include_multiplicity=False
            )
        )
        q_digest = q_grid_sha256(q_grid)
        contract = QNormalizationContract(
            planned_count=planned_count,
            accepted_count=accepted_count,
            multiplicity=multiplicity,
            half_space_role=role,
            interval_id=int(interval["id"]),
            q_digest=q_digest,
        )
        normalization_contracts.append(contract)
        logger.debug(
            "qspace interval %d q-normalization: planned=%d accepted=%d "
            "mask_rejected=%d multiplicity=%d (role=%s)",
            int(interval["id"]),
            planned_count,
            accepted_count,
            contract.mask_rejected,
            multiplicity,
            role,
        )
        interval_plans.append(
            QSpaceIntervalPlan(
                interval_id=int(interval["id"]),
                h_bounds=_axis_bounds(interval_dict, "h"),
                k_bounds=_axis_bounds(interval_dict, "k"),
                l_bounds=_axis_bounds(interval_dict, "l"),
                half_space_role=role,
                reciprocal_multiplicity=multiplicity,
                reciprocal_point_count=int(reciprocal_space_points_counter(interval_dict, supercell)),
                q_grid_digest=q_digest,
                mask_digest=mask_digest,
                l_coverage=_l_coverage_for_role(role),
            )
        )
    # One INFO summary instead of one line per interval (per-interval detail is at DEBUG).
    _total_mask_rejected = sum(c.mask_rejected for c in normalization_contracts)
    _multiplicity_counts: dict[int, int] = {}
    for _contract in normalization_contracts:
        _multiplicity_counts[_contract.multiplicity] = (
            _multiplicity_counts.get(_contract.multiplicity, 0) + 1
        )
    logger.info(
        "qspace q-normalization: %d intervals | total mask_rejected=%d | multiplicities=%s "
        "(per-interval detail at DEBUG)",
        len(normalization_contracts),
        _total_mask_rejected,
        dict(sorted(_multiplicity_counts.items())),
    )
    q_grid_set_digest = digest_dict(
        {
            "schema_version": SCATTERING_IDENTITY_SCHEMA_VERSION,
            "interval_q_grid_digests": {
                str(interval.interval_id): interval.q_grid_digest
                for interval in interval_plans
            },
        },
        domain="mosaic.qspace.q_grid_set.v1",
    )
    return QSpacePlan(
        run_digest=identity.run_digest,
        scientific_digest=identity.scientific_digest,
        execution_digest=identity.execution_digest,
        mask_digest=mask_digest,
        intervals=tuple(interval_plans),
        q_grid_set_digest=q_grid_set_digest,
        q_normalization_contracts=tuple(normalization_contracts),
    )


def write_run_manifest(
    output_dir: str | Path,
    identity: ScatteringRunIdentity,
    *,
    execution_contract: Mapping[str, Any],
) -> Path:
    manifest = RunManifest(
        scientific_digest=identity.scientific_digest,
        execution_digest=identity.execution_digest,
        run_digest=identity.run_digest,
        execution_contract=execution_contract,
    )
    path = run_manifest_path(output_dir, identity.run_digest)
    write_manifest(path, manifest, output_dir=output_dir)
    return path


def write_qspace_plan(output_dir: str | Path, plan: QSpacePlan) -> Path:
    path = qspace_plan_path(output_dir, plan.run_digest)
    write_manifest(path, plan, output_dir=output_dir)
    return path


def build_scattering_interval_lookup(
    reciprocal_space_intervals: list[dict],
) -> dict[int, dict]:
    return {
        int(interval["id"]): interval
        for interval in reciprocal_space_intervals
    }


def build_scattering_precompute_work_units(
    reciprocal_space_intervals: list[dict],
    *,
    dimension: int,
    output_dir: str,
    work_identity: ScatteringWorkIdentity | None = None,
) -> list[ScatteringWorkUnit]:
    identity_kwargs = (
        work_identity.to_work_unit_kwargs()
        if work_identity is not None
        else {}
    )
    return [
        ScatteringWorkUnit.precompute_interval(
            interval_id=int(interval["id"]),
            dimension=dimension,
            output_dir=output_dir,
            **identity_kwargs,
        )
        for interval in sorted(reciprocal_space_intervals, key=lambda item: int(item["id"]))
    ]


def build_scattering_interval_chunk_work_units(
    unsaved_interval_chunks: list[tuple[int, int]],
    *,
    dimension: int,
    output_dir: str,
    work_identity: ScatteringWorkIdentity | None = None,
) -> list[ScatteringWorkUnit]:
    identity_kwargs = (
        work_identity.to_work_unit_kwargs()
        if work_identity is not None
        else {}
    )
    return [
        ScatteringWorkUnit.interval_chunk(
            interval_id=int(interval_id),
            chunk_id=int(chunk_id),
            dimension=dimension,
            output_dir=output_dir,
            **identity_kwargs,
        )
        for interval_id, chunk_id in sorted(
            {(int(interval_id), int(chunk_id)) for interval_id, chunk_id in unsaved_interval_chunks}
        )
    ]


def chunk_ids_for_work_units(work_units: list[ScatteringWorkUnit]) -> list[int]:
    return sorted(
        {
            int(work_unit.chunk_id)
            for work_unit in work_units
            if work_unit.chunk_id is not None
        }
    )


def interval_ids_for_work_units(work_units: list[ScatteringWorkUnit]) -> list[int]:
    return sorted({int(work_unit.interval_id) for work_unit in work_units})


def interval_paths_for_work_units(work_units: list[ScatteringWorkUnit]) -> dict[int, Path]:
    interval_paths: dict[int, Path] = {}
    for work_unit in work_units:
        if work_unit.interval_artifact is None or work_unit.interval_artifact.path is None:
            continue
        interval_paths[int(work_unit.interval_id)] = Path(work_unit.interval_artifact.path)
    return interval_paths


def build_scattering_execution_plan(
    *,
    parameters: dict,
    db_manager,
    output_dir: str,
    work_identity: ScatteringWorkIdentity | None = None,
) -> ScatteringExecutionPlan:
    supercell = np.asarray(parameters["supercell"])
    dimension = int(len(supercell))
    interval_work_units = tuple(
        build_scattering_precompute_work_units(
            list(parameters["reciprocal_space_intervals"]),
            dimension=dimension,
            output_dir=output_dir,
            work_identity=work_identity,
        )
    )
    chunk_work_units = tuple(
        build_scattering_interval_chunk_work_units(
            list(
                db_manager.get_interval_chunks()
                if hasattr(db_manager, "get_interval_chunks")
                else db_manager.get_unsaved_interval_chunks()
            ),
            dimension=dimension,
            output_dir=output_dir,
            work_identity=work_identity,
        )
    )
    total_reciprocal_points = sum(
        reciprocal_space_points_counter(to_interval_dict(interval), supercell)
        for interval in parameters["reciprocal_space_intervals_all"]
    )
    return ScatteringExecutionPlan(
        interval_work_units=interval_work_units,
        chunk_work_units=chunk_work_units,
        chunk_ids=tuple(chunk_ids_for_work_units(list(chunk_work_units))),
        total_reciprocal_points=int(total_reciprocal_points),
    )


__all__ = [
    "QSPACE_PLAN_SCHEMA",
    "QSpaceIntervalPlan",
    "QSpacePlan",
    "RUN_MANIFEST_SCHEMA",
    "RunManifest",
    "SCATTERING_IDENTITY_SCHEMA_VERSION",
    "ScatteringExecutionPlan",
    "ScatteringRunIdentity",
    "ScatteringWorkIdentity",
    "build_execution_digest",
    "build_mask_digest",
    "build_qspace_plan",
    "build_run_digest",
    "build_run_identity",
    "build_scattering_backend_policy_digest",
    "build_scattering_interval_chunk_work_units",
    "build_scattering_execution_plan",
    "build_scattering_interval_lookup",
    "build_scattering_precompute_work_units",
    "build_scientific_digest",
    "build_scientific_identity_payload",
    "build_source_structure_digest",
    "chunk_ids_for_work_units",
    "interval_ids_for_work_units",
    "interval_paths_for_work_units",
    "prepare_scattering_run_identity",
    "q_grid_sha256",
    "write_qspace_plan",
    "write_run_manifest",
]
