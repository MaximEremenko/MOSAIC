"""
core/qspace/normalization.py

Single authoritative q-normalization record and byte estimators.

Three reciprocal-space counts are computed independently and must be reconciled:

* the PLANNED interval-bound count (mask-blind, dense) -- folds multiplicity in;
* the ACCEPTED masked q_grid count (multiplicity-free, currently telemetry-only);
* the PERSISTED ``contribution_reciprocal_points = accepted * multiplicity``.

This module provides one immutable record that holds the multiplicity-FREE planned and
accepted counts side by side, applies half-space multiplicity in EXACTLY one place
(``reciprocal_point_count``), and makes mask rejection explicit (``mask_rejected``). It is
a pure leaf (imports only the stdlib) so it can be referenced from any layer without an
import cycle.

It is intentionally stored outside the durable qspace plan / commit identity.
``build_qspace_plan`` writes the sidecar, and commit-time validation compares the
persisted ``contribution_reciprocal_points`` against it explicitly.
"""
from __future__ import annotations

import functools

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # imported lazily inside the sidecar I/O helpers (keeps this a leaf)
    from pathlib import Path


# Output payload element sizes (the durable residual/scattering chunk payload).
_COMPLEX128_BYTES = 16
_INT64_BYTES = 8
_FLOAT64_BYTES = 8

# Sidecar that records the q-normalization contracts NEXT TO ``qspace_plan.json``.
# It is deliberately SEPARATE from the plan file: ``qspace_plan.json``'s file_sha256 is
# ``qspace_plan_digest`` (part of work-unit identity, planning.py), so its bytes must not
# change. The sidecar carries the planned/accepted/multiplicity reconciliation data
# without touching that durable digest.
Q_NORMALIZATION_SIDECAR_NAME = "q_normalization.json"
Q_NORMALIZATION_SIDECAR_SCHEMA = "mosaic.qspace.q_normalization"
Q_NORMALIZATION_SIDECAR_SCHEMA_VERSION = 1

# Two DELIBERATELY-DISTINCT normalization axes (do not conflate in any future edit):
#   * reciprocal-space normalization == ``reciprocal_point_count`` == accepted x
#     multiplicity (this contract). This is the reciprocal half-space weighting.
#   * real-space coverage == ``expected_point_count`` == sum_i prod(grid_shape_nd[i])
#     (commit.py::_expected_point_count_from_grid_shape). This is the count of
#     real-space samples the inverse NUFFT emits per chunk.
# They answer different questions (how reciprocal points are weighted vs. how many
# real-space samples a chunk covers) and must never be unified.


@dataclass(frozen=True)
class QNormalizationContract:
    """Authoritative reciprocal-space normalization for one interval (or tile).

    ``planned_count`` and ``accepted_count`` are BOTH multiplicity-free so they are
    directly comparable; ``accepted_count <= planned_count`` always (a mask can only drop
    points). Multiplicity is applied once, in :attr:`reciprocal_point_count`.
    """

    planned_count: int
    accepted_count: int
    multiplicity: int
    half_space_role: str = "full"
    interval_id: int | None = None
    q_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_count", int(self.planned_count))
        object.__setattr__(self, "accepted_count", int(self.accepted_count))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))
        if self.planned_count < 0 or self.accepted_count < 0:
            raise ValueError("q-normalization counts must be non-negative.")
        if self.multiplicity < 1:
            raise ValueError("half-space multiplicity must be >= 1.")
        if self.accepted_count > self.planned_count:
            raise ValueError(
                "accepted masked count cannot exceed the planned interval-bound count "
                f"(accepted={self.accepted_count} > planned={self.planned_count})."
            )

    @property
    def mask_rejected(self) -> int:
        """Points the mask dropped (planned - accepted). Makes masking explicit."""
        return self.planned_count - self.accepted_count

    @property
    def reciprocal_point_count(self) -> int:
        """The persisted-equivalent count: accepted * multiplicity (applied ONCE).

        This is the RECIPROCAL-SPACE normalization axis. It is deliberately distinct
        from real-space coverage (``expected_point_count = sum_i prod(grid_shape_nd[i])``,
        commit.py::_expected_point_count_from_grid_shape): one weights reciprocal points
        by half-space multiplicity, the other counts emitted real-space samples. See the
        module-level note; never conflate the two.
        """
        return self.accepted_count * self.multiplicity

    def to_payload(self) -> dict[str, Any]:
        return {
            "interval_id": None if self.interval_id is None else int(self.interval_id),
            "planned_count": int(self.planned_count),
            "accepted_count": int(self.accepted_count),
            "multiplicity": int(self.multiplicity),
            "half_space_role": str(self.half_space_role),
            "q_digest": None if self.q_digest is None else str(self.q_digest),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "QNormalizationContract":
        interval_id = payload.get("interval_id")
        q_digest = payload.get("q_digest")
        return cls(
            planned_count=int(payload["planned_count"]),
            accepted_count=int(payload["accepted_count"]),
            multiplicity=int(payload["multiplicity"]),
            half_space_role=str(payload.get("half_space_role", "full")),
            interval_id=None if interval_id is None else int(interval_id),
            q_digest=None if q_digest is None else str(q_digest),
        )


def estimate_attempt_output_bytes(
    *,
    real_space_sample_count: int,
    grid_rows: int,
    grid_cols: int,
    complex_bytes: int = _COMPLEX128_BYTES,
    int_bytes: int = _INT64_BYTES,
) -> int:
    """Estimate the durable per-attempt OUTPUT payload size in bytes (planning only).

    Mirrors the chunk payload: ``amplitudes_delta`` + ``amplitudes_average`` (complex128,
    one per real-space sample), ``point_ids`` (int64 per sample), and ``grid_shape_nd``
    (int64, rows x cols). This is the quantity a byte-budgeted WorkUnit splits against; it
    is an upper-bound estimate and performs no I/O.
    """
    samples = max(int(real_space_sample_count), 0)
    rows = max(int(grid_rows), 0)
    cols = max(int(grid_cols), 0)
    amplitudes = samples * 2 * int(complex_bytes)
    point_ids = samples * int(int_bytes)
    grid_shape = rows * cols * int(int_bytes)
    return amplitudes + point_ids + grid_shape


def estimate_qgrid_input_bytes(
    *,
    accepted_count: int,
    dim: int,
    float_bytes: int = _FLOAT64_BYTES,
) -> int:
    """Estimate the reciprocal q_grid INPUT size in bytes (accepted points x dim x float64)."""
    return max(int(accepted_count), 0) * max(int(dim), 0) * int(float_bytes)


def _q_normalization_sidecar_path(output_dir: "str | Path", run_digest: str) -> "Path":
    """Derive the sidecar path the SAME way the qspace plan path is derived.

    Lives at ``<output_dir>/.mosaic/runs/<run_digest>/q_normalization.json`` -- i.e.
    in the same run-scoped directory as ``qspace_plan.json``, the file whose bytes are
    the identity-bearing ``qspace_plan_digest``. The storage import is function-local so
    this module stays a pure import-time leaf (no orchestration/storage edge at import).
    """
    from core.storage.attempt_store import run_root

    return run_root(output_dir, run_digest) / Q_NORMALIZATION_SIDECAR_NAME


def q_normalization_sidecar_payload(
    contracts: "list[QNormalizationContract] | tuple[QNormalizationContract, ...]",
) -> dict[str, Any]:
    """Build the JSON-serializable sidecar payload (schema + ordered interval contracts)."""
    ordered = sorted(
        contracts,
        key=lambda contract: (-1 if contract.interval_id is None else int(contract.interval_id)),
    )
    return {
        "schema": Q_NORMALIZATION_SIDECAR_SCHEMA,
        "schema_version": Q_NORMALIZATION_SIDECAR_SCHEMA_VERSION,
        "intervals": [contract.to_payload() for contract in ordered],
    }


def write_q_normalization_sidecar(
    output_dir: "str | Path",
    run_digest: str,
    contracts: "list[QNormalizationContract] | tuple[QNormalizationContract, ...]",
) -> "Path":
    """Persist all intervals' q-normalization contracts to the sidecar (atomic JSON).

    Reuses the existing atomic JSON writer (``core.storage.atomic.atomic_write_json``);
    NEVER touches ``qspace_plan.json``. ``run_digest`` scopes the path exactly as the
    qspace plan path is scoped. Imports are function-local to keep this module a leaf.
    """
    from core.storage.atomic import atomic_write_json

    path = _q_normalization_sidecar_path(output_dir, run_digest)
    atomic_write_json(
        path,
        q_normalization_sidecar_payload(contracts),
        output_dir=output_dir,
    )
    return path


@functools.lru_cache(maxsize=32)
def _load_sidecar_index(path_str: str, mtime_ns: int) -> "dict[int, dict[str, Any]]":
    """Parse the sidecar ONCE per (path, mtime) and index its contracts by interval_id.

    ``mtime_ns`` is part of the cache key so a rewritten sidecar is re-read. This exists
    because the commit-time reconciliation calls :func:`load_q_normalization_contract`
    once PER ATTEMPT; without this cache each call re-opened and fully re-parsed the whole
    sidecar (O(N**2) over a run's attempts at scattering scale).
    """
    import json

    with open(path_str, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    index: "dict[int, dict[str, Any]]" = {}
    for entry in payload.get("intervals", []):
        interval_id = entry.get("interval_id")
        if interval_id is not None:
            index[int(interval_id)] = entry
    return index


def load_q_normalization_contract(
    output_dir: "str | Path",
    run_digest: str,
    interval_id: int,
) -> "QNormalizationContract | None":
    """Load one interval's contract from the sidecar, or ``None`` if absent.

    Returns ``None`` when the sidecar file does not exist (older runs predate the
    feature) or contains no entry for ``interval_id`` -- callers SKIP their assertion in
    that case for back-compatibility. The sidecar is parsed once per (path, mtime) and
    cached, so repeated per-attempt lookups during a run are O(1).
    """
    path = _q_normalization_sidecar_path(output_dir, run_digest)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    entry = _load_sidecar_index(str(path), mtime_ns).get(int(interval_id))
    if entry is None:
        return None
    return QNormalizationContract.from_payload(entry)


__all__ = [
    "QNormalizationContract",
    "Q_NORMALIZATION_SIDECAR_NAME",
    "Q_NORMALIZATION_SIDECAR_SCHEMA",
    "Q_NORMALIZATION_SIDECAR_SCHEMA_VERSION",
    "estimate_attempt_output_bytes",
    "estimate_qgrid_input_bytes",
    "load_q_normalization_contract",
    "q_normalization_sidecar_payload",
    "write_q_normalization_sidecar",
]
