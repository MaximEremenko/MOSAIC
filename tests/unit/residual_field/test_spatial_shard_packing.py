"""Shard packing must bound the dense lattice grid of EVERY shard.

Interval ids are emitted one reciprocal axis at a time (k-fastest for 3D), so
contiguous-id slicing produces shards whose bounding box spans nearly the full
q-volume — a 31 GiB dense grid at hkl40 scale. The spatial packer must keep
each shard's bounding-box grid inside the byte budget regardless of the id
ordering, while covering every interval exactly once.
"""
from __future__ import annotations

import math

import numpy as np

from core.residual_field.planning import _batch_interval_chunks


def _hkl40_like_geometry(n_per_axis: int = 8, step: float = 5.0):
    """Interval blocks in the id order the q-space planner uses: k fastest,
    then h, then l — a contiguous id run is a thin full-length k-column."""
    geometry: dict[int, dict] = {}
    interval_id = 0
    for l_block in range(n_per_axis):
        for h_block in range(n_per_axis):
            for k_block in range(n_per_axis):
                interval_id += 1
                geometry[interval_id] = {
                    "h_start": -step * (h_block + 1),
                    "h_end": -step * h_block,
                    "k_start": -step * (k_block + 1),
                    "k_end": -step * k_block,
                    "l_start": step * l_block,
                    "l_end": step * (l_block + 1),
                }
    return geometry


def _bbox_grid_bytes(shard, geometry, supercell) -> int:
    pitch = 1.0 / np.asarray(supercell, dtype=float)
    lo = np.min(
        [[geometry[i][f"{a}_start"] for a in ("h", "k", "l")] for i in shard], axis=0
    )
    hi = np.max(
        [[geometry[i][f"{a}_end"] for a in ("h", "k", "l")] for i in shard], axis=0
    )
    points = 1
    for axis in range(3):
        points *= max(1, int(math.floor((hi[axis] - lo[axis]) / pitch[axis] + 0.5)) + 1)
    return points * 16 * 2


def test_packed_shards_respect_grid_budget_and_cover_all_intervals():
    geometry = _hkl40_like_geometry()
    supercell = (16, 16, 16)
    budget = 1 << 30  # 1 GiB
    pairs = [(interval_id, 0) for interval_id in geometry]

    batches = _batch_interval_chunks(
        pairs,
        max_intervals_per_shard=147,  # the adaptive fold's hkl40-scale answer
        interval_geometry=geometry,
        supercell=supercell,
        grid_budget_bytes=budget,
    )

    seen: list[int] = []
    for _chunk, shard in batches:
        seen.extend(shard)
        assert _bbox_grid_bytes(shard, geometry, supercell) <= budget, (
            f"shard of {len(shard)} intervals exceeds the {budget >> 20} MiB "
            "dense-grid budget"
        )
        assert len(shard) <= 147
    assert sorted(seen) == sorted(geometry), "packing lost or duplicated intervals"


def test_contiguous_id_slicing_would_violate_budget():
    """Sanity check that the scenario is real: contiguous-id slicing at the
    same shard size DOES blow the budget on k-fastest ordering."""
    geometry = _hkl40_like_geometry()
    supercell = (16, 16, 16)
    budget = 1 << 30
    ids = sorted(geometry)
    worst = max(
        _bbox_grid_bytes(tuple(ids[start : start + 147]), geometry, supercell)
        for start in range(0, len(ids), 147)
    )
    assert worst > budget


def test_packing_deterministic_and_worker_count_independent():
    geometry = _hkl40_like_geometry()
    supercell = (16, 16, 16)
    pairs = [(interval_id, chunk) for interval_id in geometry for chunk in (0, 1)]
    kwargs = dict(
        max_intervals_per_shard=64,
        interval_geometry=geometry,
        supercell=supercell,
        grid_budget_bytes=1 << 28,
    )
    first = _batch_interval_chunks(pairs, **kwargs)
    second = _batch_interval_chunks(list(reversed(pairs)), **kwargs)
    assert first == second, "shard identity must not depend on input order"
    per_chunk = {}
    for chunk, shard in first:
        per_chunk.setdefault(chunk, []).append(shard)
    assert per_chunk[0] == per_chunk[1], "chunks must share one shard split"


def test_missing_geometry_falls_back_to_contiguous_slicing():
    geometry = _hkl40_like_geometry()
    pairs = [(interval_id, 0) for interval_id in geometry]
    del geometry[1]  # one unknown id disables packing for safety
    batches = _batch_interval_chunks(
        pairs,
        max_intervals_per_shard=100,
        interval_geometry=geometry,
        supercell=(16, 16, 16),
        grid_budget_bytes=1 << 30,
    )
    sizes = sorted({len(shard) for _chunk, shard in batches}, reverse=True)
    assert sizes[0] == 100, "fallback should be plain contiguous slicing"
