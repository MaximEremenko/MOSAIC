"""C3a: byte-budgeted q-space tiling plan (additive, default-off).

Pins the contract of ``core.scattering.work_unit_tiling.plan_qspace_tiles`` and its env
resolver ``resolve_qspace_byte_budget``:
  * the env-driven budget resolver (unset/empty/<=0 -> UNLIMITED -> None),
  * ``plan_qspace_tiles`` always producing contiguous, non-overlapping q-row tiles that
    EXACTLY cover ``[0, accepted_count)``, with the no-split default preserved when
    unlimited or under budget, and the minimum equal-ish split when over budget.

The q-input estimate (``estimate_qgrid_input_bytes``) is affine in the q-row count with a
ZERO fixed per-tile term, so a half budget splits cleanly into two tiles and a tiny budget
forces one q-row per tile.
"""
from __future__ import annotations

import math

import pytest

from core.qspace.normalization import estimate_qgrid_input_bytes
from core.scattering.work_unit_tiling import (
    plan_qspace_tiles,
    resolve_qspace_byte_budget,
)


_DIM = 3


def _whole_bytes(accepted_count: int, dim: int = _DIM) -> int:
    return estimate_qgrid_input_bytes(accepted_count=accepted_count, dim=dim)


def _tile_bytes(start: int, stop: int, dim: int = _DIM) -> int:
    return estimate_qgrid_input_bytes(accepted_count=stop - start, dim=dim)


def _assert_exact_cover(tiles, accepted_count: int) -> None:
    # contiguous + non-overlapping + covers [0, accepted_count) exactly
    assert tiles[0][0] == 0
    assert tiles[-1][1] == accepted_count
    for (start, stop), (next_start, _next_stop) in zip(tiles, tiles[1:]):
        assert start <= stop
        assert stop == next_start
    covered = sum(stop - start for start, stop in tiles)
    assert covered == accepted_count


# ---- resolve_qspace_byte_budget ------------------------------------------

def test_qspace_budget_unset_is_unlimited(monkeypatch):
    monkeypatch.delenv("MOSAIC_QSPACE_BYTE_BUDGET", raising=False)
    assert resolve_qspace_byte_budget() is None


@pytest.mark.parametrize("raw", ["", "   ", "0", "-1", "-1000", "not-an-int", "12.5"])
def test_qspace_budget_empty_or_nonpositive_or_garbage_is_unlimited(monkeypatch, raw):
    monkeypatch.setenv("MOSAIC_QSPACE_BYTE_BUDGET", raw)
    assert resolve_qspace_byte_budget() is None


def test_qspace_budget_positive_int_is_returned(monkeypatch):
    monkeypatch.setenv("MOSAIC_QSPACE_BYTE_BUDGET", "4096")
    assert resolve_qspace_byte_budget() == 4096


def test_qspace_budget_independent_of_workunit_budget(monkeypatch):
    # The two budgets are read from DIFFERENT env vars; the q-space one must not pick up
    # the output (work-unit) budget.
    monkeypatch.setenv("MOSAIC_WORKUNIT_BYTE_BUDGET", "111")
    monkeypatch.delenv("MOSAIC_QSPACE_BYTE_BUDGET", raising=False)
    assert resolve_qspace_byte_budget() is None


# ---- plan_qspace_tiles: no-split paths ------------------------------------

def test_unlimited_budget_single_tile():
    tiles = plan_qspace_tiles(accepted_count=500, dim=_DIM, byte_budget=None)
    assert tiles == ((0, 500),)


def test_budget_at_or_above_whole_estimate_single_tile():
    total = 500
    whole = _whole_bytes(total)
    # exactly the whole estimate -> still a single tile (<=)
    assert plan_qspace_tiles(accepted_count=total, dim=_DIM, byte_budget=whole) == (
        (0, total),
    )
    # comfortably above -> single tile
    assert plan_qspace_tiles(
        accepted_count=total, dim=_DIM, byte_budget=whole * 10
    ) == ((0, total),)


def test_accepted_count_zero_single_empty_tile():
    assert plan_qspace_tiles(accepted_count=0, dim=_DIM, byte_budget=None) == ((0, 0),)
    assert plan_qspace_tiles(accepted_count=0, dim=_DIM, byte_budget=8) == ((0, 0),)


# ---- plan_qspace_tiles: split paths ---------------------------------------

def test_budget_half_yields_two_contiguous_tiles():
    # The q-input estimate has NO fixed per-tile term, so half the whole budget splits
    # exactly into two tiles (the canonical example).
    total = 500
    whole = _whole_bytes(total)
    budget = whole // 2
    tiles = plan_qspace_tiles(accepted_count=total, dim=_DIM, byte_budget=budget)
    assert len(tiles) == 2
    _assert_exact_cover(tiles, total)
    # equal-ish split (sizes differ by at most one)
    spans = sorted(stop - start for start, stop in tiles)
    assert spans[-1] - spans[0] <= 1
    for start, stop in tiles:
        assert _tile_bytes(start, stop) <= budget


def test_tiny_budget_yields_one_tile_per_qrow():
    total = 7
    # per-q-row payload is dim*8 = 24 bytes; a budget below that forces one q-row per tile.
    budget = 8
    tiles = plan_qspace_tiles(accepted_count=total, dim=_DIM, byte_budget=budget)
    assert len(tiles) == total
    _assert_exact_cover(tiles, total)
    assert tiles == tuple((i, i + 1) for i in range(total))


def test_minimum_number_of_tiles():
    # Pick a budget that fits exactly 100 q-rows; expect ceil(total/100) tiles.
    per_row = _whole_bytes(1)
    total = 1000
    budget = per_row * 100
    tiles = plan_qspace_tiles(accepted_count=total, dim=_DIM, byte_budget=budget)
    _assert_exact_cover(tiles, total)
    for start, stop in tiles:
        assert _tile_bytes(start, stop) <= budget
    assert len(tiles) == 10


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("total", [1, 2, 3, 17, 100, 499, 500, 1024])
@pytest.mark.parametrize("budget_fraction", [2, 3, 5, 8, 64])
def test_split_invariants_contiguous_cover_and_under_budget(dim, total, budget_fraction):
    whole = _whole_bytes(total, dim)
    budget = max(1, whole // budget_fraction)
    tiles = plan_qspace_tiles(accepted_count=total, dim=dim, byte_budget=budget)
    _assert_exact_cover(tiles, total)
    # no empty interior tiles
    for start, stop in tiles:
        assert stop - start >= 1
    # every tile within budget, EXCEPT the unavoidable single-q-row case where one q-row's
    # payload already exceeds the budget.
    for start, stop in tiles:
        span = stop - start
        if span == 1:
            continue
        assert _tile_bytes(start, stop, dim) <= budget
    # equal-ish: tile spans differ by at most one
    spans = [stop - start for start, stop in tiles]
    assert max(spans) - min(spans) <= 1
    # MINIMALITY: using one fewer tile would over-fill at least one tile beyond budget
    # (unless every tile is already a single forced q-row -- the irreducible case).
    if len(tiles) > 1 and max(spans) > 1:
        fewer = len(tiles) - 1
        biggest = math.ceil(total / fewer)  # largest tile if we used one fewer tile
        assert _tile_bytes(0, biggest, dim) > budget
