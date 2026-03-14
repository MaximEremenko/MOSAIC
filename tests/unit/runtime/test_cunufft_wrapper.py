from __future__ import annotations

import builtins

import numpy as np
import pytest

import core.adapters.cunufft_wrapper as cunufft_wrapper


def _block_finufft_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "finufft":
            raise ModuleNotFoundError("No module named 'finufft'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cunufft_wrapper, "_FINUFFT3", {})
    monkeypatch.setattr(cunufft_wrapper, "_DIRECT_CPU_FALLBACK_WARNED", False)


def test_cpu_fallback_uses_direct_forward_path_when_finufft_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_finufft_import(monkeypatch)
    real_coords = np.array([[0.0], [0.5]], dtype=np.float64)
    weights = np.array([1.0 + 0.0j, 2.0 - 1.0j], dtype=np.complex128)
    q_coords = np.array([[0.0], [1.25]], dtype=np.float64)
    expected = np.exp(1j * (q_coords @ real_coords.T)) @ weights

    with pytest.warns(RuntimeWarning, match="slow direct CPU fallback"):
        result = cunufft_wrapper._cpu_fallback(
            real_coords,
            weights,
            q_coords,
            eps=1e-12,
            inverse=False,
            batch=8,
        )

    np.testing.assert_allclose(result, expected)


def test_cpu_fallback_uses_direct_inverse_path_when_finufft_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_finufft_import(monkeypatch)
    real_coords = np.array([[0.0], [0.5]], dtype=np.float64)
    weights = np.array([1.0 + 0.0j, 0.5 + 0.25j], dtype=np.complex128)
    q_coords = np.array([[0.0], [1.25]], dtype=np.float64)
    expected = np.exp(-1j * (real_coords @ q_coords.T)) @ weights

    with pytest.warns(RuntimeWarning, match="slow direct CPU fallback"):
        result = cunufft_wrapper._cpu_fallback(
            real_coords,
            weights,
            q_coords,
            eps=1e-12,
            inverse=True,
            batch=8,
        )

    np.testing.assert_allclose(result, expected)
