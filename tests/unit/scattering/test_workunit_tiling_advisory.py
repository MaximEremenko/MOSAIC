"""C2 advisory wiring: prove the byte-budget tiling advisory is a no-op by default and
only emits INFO logs (never altering execution) when the budget is configured.
"""
from __future__ import annotations

import logging

import pytest

from core.scattering import execution as scattering_execution
from core.scattering.contracts import ScatteringWorkUnit


def _work_units():
    return [
        ScatteringWorkUnit.interval_chunk(
            interval_id=0, chunk_id=0, dimension=3, output_dir="out"
        ),
        ScatteringWorkUnit.interval_chunk(
            interval_id=1, chunk_id=0, dimension=3, output_dir="out"
        ),
        ScatteringWorkUnit.interval_chunk(
            interval_id=0, chunk_id=1, dimension=3, output_dir="out"
        ),
    ]


def _point_data_list(samples_chunk0=4000, samples_chunk1=10):
    rows = []
    for _ in range(samples_chunk0):
        rows.append({"chunk_id": 0, "coordinates": (0.0, 0.0, 0.0)})
    for _ in range(samples_chunk1):
        rows.append({"chunk_id": 1, "coordinates": (0.0, 0.0, 0.0)})
    return rows


def test_advisory_is_noop_when_budget_unset(monkeypatch, caplog):
    monkeypatch.delenv("MOSAIC_WORKUNIT_BYTE_BUDGET", raising=False)
    caplog.set_level(logging.INFO, logger="core.scattering.execution")
    scattering_execution._log_workunit_byte_budget_tiling_advisory(
        _work_units(), _point_data_list()
    )
    assert not any(
        "byte-budget advisory" in record.getMessage() for record in caplog.records
    )


def test_advisory_logs_recommended_tiles_when_budget_set(monkeypatch, caplog):
    # Budget small enough to force a split on the 4000-sample chunk but never changes
    # which work units run. Per-sample output is ~40 bytes; 4000 samples ~= 160 KB.
    monkeypatch.setenv("MOSAIC_WORKUNIT_BYTE_BUDGET", "4096")
    caplog.set_level(logging.INFO, logger="core.scattering.execution")
    scattering_execution._log_workunit_byte_budget_tiling_advisory(
        _work_units(), _point_data_list()
    )
    messages = [r.getMessage() for r in caplog.records if "byte-budget advisory" in r.getMessage()]
    # chunk 0 (4000 samples) must trigger; chunk 1 (10 samples) fits under budget -> no log.
    assert any("chunk=0" in m and "recommended point-tiles=" in m for m in messages)
    assert not any("chunk=1" in m for m in messages)


def test_advisory_never_raises_on_bad_point_data(monkeypatch):
    # Even with malformed point data, the advisory must swallow errors (pure advisory).
    monkeypatch.setenv("MOSAIC_WORKUNIT_BYTE_BUDGET", "4096")
    scattering_execution._log_workunit_byte_budget_tiling_advisory(
        _work_units(), [{"chunk_id": 0}]  # missing "coordinates"
    )
