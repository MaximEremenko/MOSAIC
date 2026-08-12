"""Byte-budgeted point-tiling plan (optional by default).

Pins the contract of ``core.scattering.work_unit_tiling``:
  * the env-driven budget resolver (unset/empty/<=0 -> UNLIMITED -> None),
  * ``plan_point_tiles`` always producing contiguous, non-overlapping tiles that EXACTLY
    cover ``[0, total_samples)``, with the no-split default preserved when unlimited or
    under budget, and the minimum equal-ish split when over budget.
"""
from __future__ import annotations

import math

import pytest

from core.qspace.normalization import estimate_attempt_output_bytes
from core.scattering.work_unit_tiling import (
    plan_point_tiles,
    resolve_workunit_byte_budget,
)


_GRID_ROWS = 1
_GRID_COLS = 3


def _whole_bytes(total_samples: int) -> int:
    return estimate_attempt_output_bytes(
        real_space_sample_count=total_samples,
        grid_rows=_GRID_ROWS,
        grid_cols=_GRID_COLS,
    )


def _tile_bytes(start: int, stop: int) -> int:
    return estimate_attempt_output_bytes(
        real_space_sample_count=stop - start,
        grid_rows=_GRID_ROWS,
        grid_cols=_GRID_COLS,
    )


def _assert_exact_cover(tiles, total_samples: int) -> None:
    # contiguous + non-overlapping + covers [0, total_samples) exactly
    assert tiles[0][0] == 0
    assert tiles[-1][1] == total_samples
    for (start, stop), (next_start, _next_stop) in zip(tiles, tiles[1:]):
        assert start <= stop
        assert stop == next_start
    covered = sum(stop - start for start, stop in tiles)
    assert covered == total_samples


# ---- resolve_workunit_byte_budget ----------------------------------------

def test_budget_unset_is_unlimited(monkeypatch):
    monkeypatch.delenv("MOSAIC_WORKUNIT_BYTE_BUDGET", raising=False)
    assert resolve_workunit_byte_budget() is None


@pytest.mark.parametrize("raw", ["", "   ", "0", "-1", "-1000", "not-an-int", "12.5"])
def test_budget_empty_or_nonpositive_or_garbage_is_unlimited(monkeypatch, raw):
    monkeypatch.setenv("MOSAIC_WORKUNIT_BYTE_BUDGET", raw)
    assert resolve_workunit_byte_budget() is None


def test_budget_positive_int_is_returned(monkeypatch):
    monkeypatch.setenv("MOSAIC_WORKUNIT_BYTE_BUDGET", "4096")
    assert resolve_workunit_byte_budget() == 4096


# ---- plan_point_tiles: no-split paths ------------------------------------

def test_unlimited_budget_single_tile():
    tiles = plan_point_tiles(
        total_samples=1000, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=None
    )
    assert tiles == ((0, 1000),)


def test_budget_at_or_above_whole_estimate_single_tile():
    total = 1000
    whole = _whole_bytes(total)
    # exactly the whole estimate -> still a single tile (<=)
    assert plan_point_tiles(
        total_samples=total, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=whole
    ) == ((0, total),)
    # comfortably above -> single tile
    assert plan_point_tiles(
        total_samples=total,
        grid_rows=_GRID_ROWS,
        grid_cols=_GRID_COLS,
        byte_budget=whole * 10,
    ) == ((0, total),)


def test_total_samples_zero_single_empty_tile():
    assert plan_point_tiles(
        total_samples=0, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=None
    ) == ((0, 0),)
    assert plan_point_tiles(
        total_samples=0, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=8
    ) == ((0, 0),)


# ---- plan_point_tiles: split paths ---------------------------------------

def test_budget_half_yields_two_contiguous_tiles():
    # Use a zero-overhead grid so the per-tile fixed grid_shape term is 0 and the
    # half-the-whole budget splits exactly into two tiles (the plan's canonical example).
    total = 1000
    whole = estimate_attempt_output_bytes(
        real_space_sample_count=total, grid_rows=0, grid_cols=0
    )
    budget = whole // 2
    tiles = plan_point_tiles(
        total_samples=total, grid_rows=0, grid_cols=0, byte_budget=budget
    )
    assert len(tiles) == 2
    _assert_exact_cover(tiles, total)
    # equal-ish split (sizes differ by at most one)
    spans = sorted(stop - start for start, stop in tiles)
    assert spans[-1] - spans[0] <= 1
    for start, stop in tiles:
        assert (
            estimate_attempt_output_bytes(
                real_space_sample_count=stop - start, grid_rows=0, grid_cols=0
            )
            <= budget
        )


def test_budget_half_with_grid_overhead_splits_within_budget():
    # With a non-zero grid_shape term, every tile pays the fixed overhead, so a half-budget
    # may need more than two tiles -- but each tile must still stay within budget.
    total = 1000
    whole = _whole_bytes(total)
    budget = whole // 2
    tiles = plan_point_tiles(
        total_samples=total, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=budget
    )
    assert len(tiles) >= 2
    _assert_exact_cover(tiles, total)
    spans = sorted(stop - start for start, stop in tiles)
    assert spans[-1] - spans[0] <= 1
    for start, stop in tiles:
        assert _tile_bytes(start, stop) <= budget


def test_tiny_budget_yields_one_tile_per_sample():
    total = 7
    # per-sample payload is 40 bytes (2*complex128 + int64); a budget below that forces
    # one sample per tile.
    budget = 8
    tiles = plan_point_tiles(
        total_samples=total, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=budget
    )
    assert len(tiles) == total
    _assert_exact_cover(tiles, total)
    assert tiles == tuple((i, i + 1) for i in range(total))


@pytest.mark.parametrize("total", [1, 2, 3, 17, 100, 999, 1000, 4096])
@pytest.mark.parametrize("budget_fraction", [2, 3, 5, 8, 64])
def test_split_invariants_contiguous_cover_and_under_budget(total, budget_fraction):
    whole = _whole_bytes(total)
    budget = max(1, whole // budget_fraction)
    tiles = plan_point_tiles(
        total_samples=total, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=budget
    )
    _assert_exact_cover(tiles, total)
    # no empty interior tiles
    for start, stop in tiles:
        assert stop - start >= 1
    # every tile within budget, EXCEPT the unavoidable single-sample case where one
    # sample's payload (plus the fixed grid overhead) already exceeds the budget.
    for start, stop in tiles:
        span = stop - start
        if span == 1:
            continue
        assert _tile_bytes(start, stop) <= budget
    # equal-ish: tile spans differ by at most one
    spans = [stop - start for start, stop in tiles]
    assert max(spans) - min(spans) <= 1
    # MINIMALITY: using one fewer tile would over-fill at least one tile beyond budget
    # (unless every tile is already a single forced sample -- the irreducible case).
    if len(tiles) > 1 and max(spans) > 1:
        fewer = len(tiles) - 1
        biggest = math.ceil(total / fewer)  # largest tile if we used one fewer tile
        assert _tile_bytes(0, biggest) > budget


def test_minimum_number_of_tiles():
    # Pick a budget that fits exactly 100 samples' payload; expect ceil(total/100) tiles.
    per_sample = _whole_bytes(1) - _whole_bytes(0)
    total = 1000
    budget = per_sample * 100 + (_whole_bytes(0))  # 100 samples + fixed grid overhead
    tiles = plan_point_tiles(
        total_samples=total, grid_rows=_GRID_ROWS, grid_cols=_GRID_COLS, byte_budget=budget
    )
    _assert_exact_cover(tiles, total)
    for start, stop in tiles:
        assert _tile_bytes(start, stop) <= budget
    # 1000 samples, ~100 per tile -> 10 tiles
    assert len(tiles) == 10
