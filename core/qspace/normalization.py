"""
core/qspace/normalization.py

Single authoritative q-normalization record (W6.2 #1 — additive foundation).

Today three reciprocal-space counts are computed independently and never reconciled
(see plan_phase2_w6_scaling_design.md, W6.0):

* the PLANNED interval-bound count (mask-blind, dense) -- folds multiplicity in;
* the ACCEPTED masked q_grid count (multiplicity-free, currently telemetry-only);
* the PERSISTED ``contribution_reciprocal_points = accepted * multiplicity``.

This module provides one immutable record that holds the multiplicity-FREE planned and
accepted counts side by side, applies half-space multiplicity in EXACTLY one place
(``reciprocal_point_count``), and makes mask rejection explicit (``mask_rejected``). It is
a pure leaf (imports only the stdlib) so it can be referenced from any layer without an
import cycle.

It is intentionally NOT yet wired into the durable qspace plan / commit identity: this is
the additive first increment of the byte-budgeted-tiling roadmap. Wiring it into
``build_qspace_plan`` (where both counts are already in scope, planning.py:511-549) and
asserting the persisted ``contribution_reciprocal_points`` against it is a later,
digest-affecting increment that must be gated separately.
"""
from __future__ import annotations

from dataclasses import dataclass


# Output payload element sizes (the durable residual/scattering chunk payload).
_COMPLEX128_BYTES = 16
_INT64_BYTES = 8
_FLOAT64_BYTES = 8


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
        """The persisted-equivalent count: accepted * multiplicity (applied ONCE)."""
        return self.accepted_count * self.multiplicity


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


__all__ = [
    "QNormalizationContract",
    "estimate_attempt_output_bytes",
    "estimate_qgrid_input_bytes",
]
