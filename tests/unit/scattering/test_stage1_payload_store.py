"""Durable stage-1 payload store: round-trip fidelity and loader integration."""
from __future__ import annotations

import numpy as np
import pytest

from core.scattering.grid import IntervalTask
from core.scattering.streaming import (
    _STORE_MISS,
    clear_streaming_payload_memo,
    read_stored_interval_payload,
    stage1_store_has,
    write_stored_interval_payload,
)


def _task(seed: int = 3, n: int = 1000) -> IntervalTask:
    rng = np.random.default_rng(seed)
    return IntervalTask(
        irecip_id=41,
        element="O",
        q_grid=rng.uniform(-5, 5, size=(n, 3)),
        q_amp=(rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex128
        ),
        q_amp_av=(rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex128
        ),
        q_grid_digest="abc123",
        half_space_role="positive_half",
        reciprocal_multiplicity=2,
    )


def test_store_round_trip_is_exact_and_memmapped(tmp_path):
    store = str(tmp_path / "store")
    task = _task()
    assert read_stored_interval_payload(store, 41) is _STORE_MISS
    write_stored_interval_payload(store, 41, task)
    assert stage1_store_has(store, 41)

    loaded = read_stored_interval_payload(store, 41)
    assert loaded is not _STORE_MISS and loaded is not None
    assert loaded.irecip_id == 41
    assert loaded.element == "O"
    assert loaded.q_grid_digest == "abc123"
    assert loaded.half_space_role == "positive_half"
    assert loaded.reciprocal_multiplicity == 2
    np.testing.assert_array_equal(loaded.q_grid, task.q_grid)
    np.testing.assert_array_equal(loaded.q_amp, task.q_amp)
    np.testing.assert_array_equal(loaded.q_amp_av, task.q_amp_av)
    # arrays come back as in-place maps: page cache, not anonymous RSS
    assert isinstance(loaded.q_grid, np.memmap)
    assert isinstance(loaded.q_amp, np.memmap)


def test_store_records_mask_empty_intervals(tmp_path):
    store = str(tmp_path / "store")
    write_stored_interval_payload(store, 7, None)
    assert stage1_store_has(store, 7)
    assert read_stored_interval_payload(store, 7) is None


def test_store_write_is_idempotent(tmp_path):
    store = str(tmp_path / "store")
    write_stored_interval_payload(store, 5, _task(seed=1))
    before = read_stored_interval_payload(store, 5)
    write_stored_interval_payload(store, 5, _task(seed=99))  # ignored: exists
    after = read_stored_interval_payload(store, 5)
    np.testing.assert_array_equal(before.q_amp, after.q_amp)


def test_lazy_loader_prefers_store_over_compute(tmp_path, monkeypatch):
    """A loader must consume the store and never invoke stage-1 compute."""
    from core.scattering import streaming as streaming_mod

    clear_streaming_payload_memo()
    store = str(tmp_path / "store")
    task = _task(seed=8)
    write_stored_interval_payload(store, 41, task)

    def _explode(*args, **kwargs):
        raise AssertionError("stage-1 compute must not run on a store hit")

    monkeypatch.setattr(
        streaming_mod, "compute_scattering_interval_payload", _explode
    )

    class _Ctx:
        cache_token = "test-token"
        interval_lookup = {41: {"h_range": (0, 1)}}
        nufft_eps = None
        nufft_prefer_cpu = None
        nufft_gpu_only = None
        payload_store_dir = store

    (loader,) = streaming_mod.lazy_streamed_interval_loaders((41,), _Ctx())
    loaded = loader()
    assert loaded is not None
    np.testing.assert_array_equal(loaded.q_amp, task.q_amp)
    clear_streaming_payload_memo()
