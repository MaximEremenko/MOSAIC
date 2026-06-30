"""Byte-budgeted tiling plans for scattering work units (additive, default-off).

A scattering ``(interval, chunk)`` work unit emits one durable amplitude pair per
real-space sample and consumes one reciprocal q-row per accepted q-point. Two
independent axes can therefore be byte-budget split:

* the OUTPUT (real-space sample) axis -- ``plan_point_tiles`` -- bounds the durable
  per-attempt payload (``estimate_attempt_output_bytes``).
* the INPUT (reciprocal q-row) axis -- ``plan_qspace_tiles`` -- bounds the q_grid input
  the inverse NUFFT sums over (``estimate_qgrid_input_bytes``). Because the inverse NUFFT
  SUMS over q-points, the full result equals the sum of per-tile partials -- but with a
  DIFFERENT floating-point reduction order, so per-tile partials agree with the un-tiled
  result only within the PREDICTED forward-error tolerance
  (``core.storage.agreement.predict_agreement_rtol``), never bitwise.

Both planners are PURE planning leaves: they compute tile boundaries only. They perform
no I/O, build no work units, and (by themselves) change no execution. The budgets are read
from ``MOSAIC_WORKUNIT_BYTE_BUDGET`` (output) / ``MOSAIC_QSPACE_BYTE_BUDGET`` (q-input) and
default to UNLIMITED (unset) -- in which case the planners always return a SINGLE tile
covering the whole range, i.e. the exact pre-existing behaviour.

Imports are restricted to stdlib plus ``estimate_attempt_output_bytes`` /
``estimate_qgrid_input_bytes`` (both already leaves) so the module stays an import-time
leaf with no orchestration/storage edges.
"""
from __future__ import annotations

import math
import os

from core.qspace.normalization import (
    estimate_attempt_output_bytes,
    estimate_qgrid_input_bytes,
)

__all__ = [
    "resolve_workunit_byte_budget",
    "plan_point_tiles",
    "resolve_qspace_byte_budget",
    "plan_qspace_tiles",
]

_BYTE_BUDGET_ENV = "MOSAIC_WORKUNIT_BYTE_BUDGET"
_QSPACE_BYTE_BUDGET_ENV = "MOSAIC_QSPACE_BYTE_BUDGET"


def resolve_workunit_byte_budget() -> int | None:
    """Resolve the per-work-unit OUTPUT byte budget from the environment.

    Returns ``None`` (UNLIMITED -> never split) when ``MOSAIC_WORKUNIT_BYTE_BUDGET`` is
    unset, empty, non-integer, or ``<= 0``. Otherwise returns the positive integer budget.
    """
    raw = os.environ.get(_BYTE_BUDGET_ENV)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    return budget


def _output_byte_terms(*, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Split the affine output estimate into (per_sample, fixed) byte terms.

    ``estimate_attempt_output_bytes`` is affine in the sample count:
    ``bytes(n) = per_sample * n + fixed``. The fixed term is the ``grid_shape_nd`` payload
    (independent of the sample count); EVERY tile pays it, so it is what makes the split
    coarser than a naive ``ceil(whole / budget)``. We recover both terms by evaluating the
    estimator at one sample and at zero samples.
    """
    one = estimate_attempt_output_bytes(
        real_space_sample_count=1, grid_rows=grid_rows, grid_cols=grid_cols
    )
    zero = estimate_attempt_output_bytes(
        real_space_sample_count=0, grid_rows=grid_rows, grid_cols=grid_cols
    )
    per_sample = int(one) - int(zero)
    fixed = int(zero)
    return per_sample, fixed


def plan_point_tiles(
    *,
    total_samples: int,
    grid_rows: int,
    grid_cols: int,
    byte_budget: int | None,
) -> tuple[tuple[int, int], ...]:
    """Plan contiguous, non-overlapping ``(start, stop)`` point-tiles over the sample axis.

    The returned tiles EXACTLY cover ``[0, total_samples)`` (their ranges are contiguous,
    non-overlapping, and union to the whole interval). The contract:

      * ``total_samples == 0`` -> a single empty tile ``((0, 0),)``.
      * ``byte_budget is None`` (UNLIMITED) -> a single tile ``((0, total_samples),)``.
      * whole-unit estimated output bytes ``<= byte_budget`` -> a single tile (no split).
      * otherwise -> the MINIMUM number of equal-ish contiguous tiles such that every tile's
        estimated output stays ``<= byte_budget``. ``tiles = ceil(whole_bytes / byte_budget)``,
        samples distributed as evenly as possible with the remainder on the EARLY tiles.

    When even a single sample's output exceeds the budget, each sample becomes its own tile
    (``tiles == total_samples``); the planner never emits an empty interior tile.
    """
    total = max(int(total_samples), 0)
    if total == 0:
        return ((0, 0),)

    rows = max(int(grid_rows), 0)
    cols = max(int(grid_cols), 0)

    whole_bytes = estimate_attempt_output_bytes(
        real_space_sample_count=total, grid_rows=rows, grid_cols=cols
    )

    if byte_budget is None or whole_bytes <= byte_budget:
        return ((0, total),)

    # Each tile's output is per_sample*span + fixed (the fixed grid_shape_nd term is paid by
    # EVERY tile). The largest span keeping a tile within budget is therefore
    # floor((budget - fixed) / per_sample); the +fixed is what makes the split coarser than
    # a naive ceil(whole / budget).
    per_sample, fixed = _output_byte_terms(grid_rows=rows, grid_cols=cols)
    if per_sample <= 0:
        # No sample-dependent payload to bound: a single tile already fit the <= check above,
        # so this is unreachable in practice; keep it total-safe.
        max_samples_per_tile = total
    else:
        usable = byte_budget - fixed
        max_samples_per_tile = usable // per_sample if usable >= per_sample else 0

    if max_samples_per_tile < 1:
        # Even one sample plus the fixed grid term exceeds the budget: splitting cannot help,
        # so each sample becomes its own tile (never an empty interior tile).
        tile_count = total
    else:
        tile_count = int(math.ceil(total / max_samples_per_tile))

    tile_count = max(1, min(tile_count, total))

    # Distribute `total` samples across `tile_count` contiguous tiles as evenly as possible,
    # placing the remainder on the early tiles (sizes differ by at most one).
    base = total // tile_count
    remainder = total % tile_count
    tiles: list[tuple[int, int]] = []
    start = 0
    for index in range(tile_count):
        span = base + (1 if index < remainder else 0)
        stop = start + span
        tiles.append((start, stop))
        start = stop
    return tuple(tiles)


def resolve_qspace_byte_budget() -> int | None:
    """Resolve the per-work-unit reciprocal q_grid INPUT byte budget from the environment.

    Returns ``None`` (UNLIMITED -> never split) when ``MOSAIC_QSPACE_BYTE_BUDGET`` is
    unset, empty, non-integer, or ``<= 0``. Otherwise returns the positive integer budget.
    Mirrors :func:`resolve_workunit_byte_budget` but for the q-input axis.
    """
    raw = os.environ.get(_QSPACE_BYTE_BUDGET_ENV)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    return budget


def plan_qspace_tiles(
    *,
    accepted_count: int,
    dim: int,
    byte_budget: int | None,
) -> tuple[tuple[int, int], ...]:
    """Plan contiguous, non-overlapping ``(start, stop)`` q-row tiles over the q-input axis.

    The returned tiles EXACTLY cover ``[0, accepted_count)`` (their ranges are contiguous,
    non-overlapping, and union to the whole interval). Splitting the reciprocal q_grid into
    sub-ranges and SUMMING the inverse-NUFFT partials reproduces the full inverse result
    (the inverse sums over q-points), but with a different floating-point reduction order,
    so the per-tile sum agrees with the un-tiled result only within the PREDICTED tolerance
    (``core.storage.agreement``), never bitwise. The contract:

      * ``accepted_count == 0`` -> a single empty tile ``((0, 0),)``.
      * ``byte_budget is None`` (UNLIMITED) -> a single tile ``((0, accepted_count),)``.
      * whole q_grid input bytes ``<= byte_budget`` -> a single tile (no split).
      * otherwise -> the MINIMUM number of equal-ish contiguous tiles such that every tile's
        estimated q-input stays ``<= byte_budget``. The q-input estimate is purely affine in
        the q-row count with NO fixed per-tile overhead (``bytes(n) = per_row * n``), so the
        largest tile span within budget is ``floor(budget / per_row)`` and the tile count is
        ``ceil(accepted_count / max_rows_per_tile)``; rows are distributed as evenly as
        possible with the remainder on the EARLY tiles (sizes differ by at most one).

    When even a single q-row's input exceeds the budget, each q-row becomes its own tile
    (``tiles == accepted_count``); the planner never emits an empty interior tile. Mirrors
    the affine/cover/contiguity discipline of :func:`plan_point_tiles`.
    """
    total = max(int(accepted_count), 0)
    if total == 0:
        return ((0, 0),)

    d = max(int(dim), 0)

    whole_bytes = estimate_qgrid_input_bytes(accepted_count=total, dim=d)

    if byte_budget is None or whole_bytes <= byte_budget:
        return ((0, total),)

    # The q-input estimate is affine with a ZERO fixed term: bytes(n) = per_row * n where
    # per_row = dim * float64_bytes. Recover per_row from a single q-row (no fixed term to
    # subtract, unlike the output estimator's grid_shape overhead).
    per_row = estimate_qgrid_input_bytes(accepted_count=1, dim=d)
    if per_row <= 0:
        # No q-input payload to bound (dim == 0): the whole unit already fit the <= check
        # above, so this is unreachable in practice; keep it total-safe.
        max_rows_per_tile = total
    else:
        max_rows_per_tile = byte_budget // per_row

    if max_rows_per_tile < 1:
        # Even one q-row exceeds the budget: splitting cannot help, so each q-row becomes
        # its own tile (never an empty interior tile).
        tile_count = total
    else:
        tile_count = int(math.ceil(total / max_rows_per_tile))

    tile_count = max(1, min(tile_count, total))

    # Distribute `total` q-rows across `tile_count` contiguous tiles as evenly as possible,
    # placing the remainder on the early tiles (sizes differ by at most one).
    base = total // tile_count
    remainder = total % tile_count
    tiles: list[tuple[int, int]] = []
    start = 0
    for index in range(tile_count):
        span = base + (1 if index < remainder else 0)
        stop = start + span
        tiles.append((start, stop))
        start = stop
    return tuple(tiles)
