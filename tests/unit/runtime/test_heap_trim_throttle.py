"""Verify _PerTaskHeapTrim.transition fires only every Nth released task."""
from __future__ import annotations

import importlib


def _reload_wh(monkeypatch, *, every):
    if every is None:
        monkeypatch.delenv("MOSAIC_HEAP_TRIM_EVERY", raising=False)
    else:
        monkeypatch.setenv("MOSAIC_HEAP_TRIM_EVERY", every)
    import core.runtime.worker_hooks as wh
    return importlib.reload(wh)


def test_transition_fires_only_every_nth(monkeypatch):
    wh = _reload_wh(monkeypatch, every="4")
    calls = {"trim": 0}
    monkeypatch.setattr(
        wh, "_malloc_trim",
        lambda: calls.__setitem__("trim", calls["trim"] + 1),
        raising=True,
    )
    plugin = wh._PerTaskHeapTrim()
    for _ in range(3):
        plugin.transition("k", "memory", "released")
    assert calls["trim"] == 0
    plugin.transition("k", "memory", "released")
    assert calls["trim"] == 1
    for _ in range(3):
        plugin.transition("k", "memory", "released")
    assert calls["trim"] == 1
    plugin.transition("k", "memory", "released")
    assert calls["trim"] == 2


def test_non_released_transitions_never_fire(monkeypatch):
    wh = _reload_wh(monkeypatch, every="1")
    calls = {"trim": 0}
    monkeypatch.setattr(
        wh, "_malloc_trim",
        lambda: calls.__setitem__("trim", calls["trim"] + 1),
        raising=True,
    )
    plugin = wh._PerTaskHeapTrim()
    plugin.transition("k", "waiting", "processing")
    plugin.transition("k", "processing", "memory")
    plugin.transition("k", "memory", "erred")
    assert calls["trim"] == 0


def test_default_throttle_is_one(monkeypatch):
    wh = _reload_wh(monkeypatch, every=None)
    plugin = wh._PerTaskHeapTrim()
    # Default flipped from 32 → 1: GPU pool retention dominates with concurrent
    # tasks; trim every task to keep VRAM bounded.
    assert plugin._every == 1


def test_invalid_env_falls_back_to_default(monkeypatch):
    wh = _reload_wh(monkeypatch, every="not-an-int")
    plugin = wh._PerTaskHeapTrim()
    assert plugin._every == 1


def test_teardown_still_trims(monkeypatch):
    wh = _reload_wh(monkeypatch, every="32")
    calls = {"trim": 0}
    monkeypatch.setattr(
        wh, "_malloc_trim",
        lambda: calls.__setitem__("trim", calls["trim"] + 1),
        raising=True,
    )
    plugin = wh._PerTaskHeapTrim()
    plugin.teardown(worker=None)
    assert calls["trim"] == 1


def test_pressure_overrides_throttle(monkeypatch):
    """Adaptive trim: when reported VRAM pressure exceeds the threshold,
    the trim runs even on tasks that fall in the throttle gap."""
    wh = _reload_wh(monkeypatch, every="100")  # would never fire on counter
    calls = {"trim": 0}
    monkeypatch.setattr(
        wh, "_malloc_trim",
        lambda: calls.__setitem__("trim", calls["trim"] + 1),
        raising=True,
    )
    plugin = wh._PerTaskHeapTrim()
    # Force pressure detection True regardless of CuPy availability.
    monkeypatch.setattr(plugin, "_under_gpu_pressure", lambda: True, raising=True)
    plugin.transition("k", "memory", "released")
    assert calls["trim"] == 1


def test_no_pressure_respects_throttle(monkeypatch):
    wh = _reload_wh(monkeypatch, every="100")
    calls = {"trim": 0}
    monkeypatch.setattr(
        wh, "_malloc_trim",
        lambda: calls.__setitem__("trim", calls["trim"] + 1),
        raising=True,
    )
    plugin = wh._PerTaskHeapTrim()
    monkeypatch.setattr(plugin, "_under_gpu_pressure", lambda: False, raising=True)
    for _ in range(99):
        plugin.transition("k", "memory", "released")
    assert calls["trim"] == 0
    plugin.transition("k", "memory", "released")
    assert calls["trim"] == 1
