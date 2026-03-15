from __future__ import annotations

import builtins
from types import SimpleNamespace

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


def test_max_chunk_size_returns_zero_when_min_chunk_does_not_fit():
    assert (
        cunufft_wrapper._max_chunk_size(
            free_bytes=100,
            baseline_bytes=60,
            per_target_bytes=10,
            mem_frac=1.0,
            user_cap=None,
            min_chunk=5,
        )
        == 0
    )


def test_resolve_budget_policy_preserves_explicit_override():
    reserve, value, source = cunufft_wrapper._resolve_budget_policy(
        mem_frac=0.42,
        free_bytes=8 << 30,
        resident_bytes=1 << 30,
    )
    assert reserve == 0
    assert source == "explicit"
    assert value == pytest.approx(0.42)


def test_adaptive_default_reserve_is_relatively_more_permissive_on_large_vram():
    small_reserve, small_frac, small_source = cunufft_wrapper._resolve_budget_policy(
        mem_frac=None,
        free_bytes=1 << 30,
        resident_bytes=128 << 20,
    )
    large_reserve, large_frac, large_source = cunufft_wrapper._resolve_budget_policy(
        mem_frac=None,
        free_bytes=32 << 30,
        resident_bytes=128 << 20,
    )
    assert small_source == "adaptive-reserve-default"
    assert large_source == "adaptive-reserve-default"
    assert small_frac < large_frac
    assert small_reserve < (1 << 30)
    assert large_reserve >= 2 << 30


def test_adaptive_default_remains_bounded_on_small_vram():
    reserve, value, source = cunufft_wrapper._resolve_budget_policy(
        mem_frac=None,
        free_bytes=1 << 30,
        resident_bytes=512 << 20,
    )
    assert source == "adaptive-reserve-default"
    assert reserve >= 256 << 20
    assert 0.20 <= value <= 0.30


def test_plan_target_chunk_does_not_double_count_resident_bytes():
    chunk_without_resident_charge, _ = cunufft_wrapper._plan_target_chunk(
        resident_coords=np.zeros((4, 1), dtype=np.float64),
        target_coords=np.zeros((100, 1), dtype=np.float64),
        start=0,
        free_bytes=1 << 30,
        budget_fraction=1.0,
        min_chunk=1,
        max_chunk=100,
        incremental_launch_baseline_bytes=0,
        n_trans=1,
    )
    chunk_with_old_double_charge, _ = cunufft_wrapper._plan_target_chunk(
        resident_coords=np.zeros((4, 1), dtype=np.float64),
        target_coords=np.zeros((100, 1), dtype=np.float64),
        start=0,
        free_bytes=1 << 30,
        budget_fraction=1.0,
        min_chunk=1,
        max_chunk=100,
        incremental_launch_baseline_bytes=256 << 20,
        n_trans=1,
    )
    assert chunk_without_resident_charge >= chunk_with_old_double_charge


def test_plan_target_chunk_allows_small_total_problem_below_min_chunk():
    chunk, _ = cunufft_wrapper._plan_target_chunk(
        resident_coords=np.zeros((4, 1), dtype=np.float64),
        target_coords=np.zeros((8, 1), dtype=np.float64),
        start=0,
        free_bytes=10**9,
        budget_fraction=1.0,
        min_chunk=32_000,
        max_chunk=None,
        incremental_launch_baseline_bytes=0,
    )

    assert chunk == 8


def test_select_type3_sides_is_direction_aware():
    real_coords = np.array([[0.0], [1.0]])
    q_coords = np.array([[2.0], [3.0], [4.0]])
    weights = np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])

    f_sources, f_weights, f_targets = cunufft_wrapper._select_type3_sides(
        real_coords,
        weights[:2],
        q_coords,
        inverse=False,
    )
    i_sources, i_weights, i_targets = cunufft_wrapper._select_type3_sides(
        real_coords,
        weights,
        q_coords,
        inverse=True,
    )

    np.testing.assert_allclose(f_sources, real_coords)
    np.testing.assert_allclose(f_targets, q_coords)
    np.testing.assert_allclose(i_sources, q_coords)
    np.testing.assert_allclose(i_targets, real_coords)
    np.testing.assert_allclose(f_weights, weights[:2])
    np.testing.assert_allclose(i_weights, weights)


def test_adaptive_gpu_launch_no_longer_needs_device_get(monkeypatch: pytest.MonkeyPatch):
    class _NoGet:
        def __getattr__(self, name):
            if name == "get":
                raise AssertionError("planning should not call .get() on device arrays")
            raise AttributeError(name)

    monkeypatch.setattr(
        cunufft_wrapper,
        "_launch_once",
        lambda *args, **kwargs: np.array([1.0 + 0.0j]),
    )
    result = cunufft_wrapper._adaptive_gpu_launch(
        1,
        [_NoGet()],
        _NoGet(),
        [_NoGet()],
        1e-12,
        False,
    )
    np.testing.assert_allclose(result, np.array([1.0 + 0.0j]))


def test_launch_once_is_direction_agnostic_after_side_selection(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_kernel(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return np.array([2.0 + 0.0j])

    monkeypatch.setattr(cunufft_wrapper, "_KER", {1: fake_kernel})
    result = cunufft_wrapper._launch_once(
        1,
        [np.array([1.0])],
        np.array([1.0 + 0.0j]),
        [np.array([3.0])],
        1e-12,
        -1,
        {"gpu_method": 1},
    )

    np.testing.assert_allclose(result, np.array([2.0 + 0.0j]))
    assert len(calls["args"]) == 3
    assert calls["kwargs"]["isign"] == -1


def test_resident_bytes_include_source_side_scratch():
    resident_coords = np.zeros((10, 2), dtype=np.float64)
    resident_weights = np.zeros(10, dtype=np.complex128)
    expected = (10 * 2 * 8) + (10 * 16) + int(10 * cunufft_wrapper._SCRATCH_ALPHA * 16)
    assert cunufft_wrapper._resident_bytes(resident_coords, resident_weights) == expected


def test_per_target_bytes_scales_with_n_trans():
    assert cunufft_wrapper._per_target_bytes(2, 2) > cunufft_wrapper._per_target_bytes(2, 1)


def test_execute_inverse_cunufft_batch_stacks_cpu_results(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_cpu(real_coords, weights, q_coords, eps, inverse):
        calls.append(np.asarray(weights).copy())
        return np.array([weights.sum() + 0.0j, weights.sum() + 1.0j], dtype=np.complex128)

    monkeypatch.setattr(cunufft_wrapper, "_CPU_ONLY", True)
    monkeypatch.setattr(cunufft_wrapper, "_cpu_fallback", fake_cpu)

    result = cunufft_wrapper.execute_inverse_cunufft_batch(
        q_coords=np.array([[0.0], [1.0]], dtype=np.float64),
        weights=np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]]),
        real_coords=np.array([[0.0], [0.5]], dtype=np.float64),
    )

    assert len(calls) == 2
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], np.array([3.0 + 0.0j, 3.0 + 1.0j]))
    np.testing.assert_allclose(result[1], np.array([7.0 + 0.0j, 7.0 + 1.0j]))


def test_execute_inverse_cunufft_batch_device_materializes_once(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=RuntimeError),
            driver=SimpleNamespace(CUDADriverError=RuntimeError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    asnumpy_calls = {"count": 0}
    fake_cp.asnumpy = lambda x: asnumpy_calls.__setitem__("count", asnumpy_calls["count"] + 1) or np.asarray(x)

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_execute_inverse_batch_gpu",
        lambda **kwargs: np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
    )

    result = cunufft_wrapper._execute_inverse_cunufft_batch_device(
        q_coords=np.array([[0.0]], dtype=np.float64),
        weights=np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
        real_coords=np.array([[0.0]], dtype=np.float64),
        eps=1e-12,
    )

    np.testing.assert_allclose(result, np.array([[1.0 + 0.0j], [2.0 + 0.0j]]))
    assert asnumpy_calls["count"] == 1


def test_execute_inverse_cunufft_batch_device_falls_back_safely(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_as_device",
        lambda arr, allow_fail=False: None if allow_fail else np.asarray(arr),
    )
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda real_coords, weights, q_coords, eps, inverse: np.array(
            [weights.sum() + 0.0j],
            dtype=np.complex128,
        ),
    )

    result = cunufft_wrapper._execute_inverse_cunufft_batch_device(
        q_coords=np.array([[0.0]], dtype=np.float64),
        weights=np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
        real_coords=np.array([[0.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(result, np.array([[1.0 + 0.0j], [2.0 + 0.0j]]))


def test_execute_inverse_cunufft_super_batch_preserves_shape(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cunufft_wrapper,
        "_execute_inverse_cunufft_batch",
        lambda **kwargs: np.asarray(kwargs["weights_arr"], dtype=np.complex128),
    )

    result = cunufft_wrapper.execute_inverse_cunufft_super_batch(
        q_coords=np.array([[0.0], [1.0]], dtype=np.float64),
        weights=np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j],
                [3.0 + 0.0j, 4.0 + 0.0j],
                [5.0 + 0.0j, 6.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
        real_coords=np.array([[0.0], [0.5]], dtype=np.float64),
    )

    assert result.shape == (3, 2)


def test_execute_inverse_cunufft_super_batch_reduces_width_on_runtime_error(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_exec(**kwargs):
        width = kwargs["weights_arr"].shape[0]
        calls.append(width)
        if width > 2:
            raise RuntimeError("out of memory")
        return np.asarray(kwargs["weights_arr"], dtype=np.complex128)

    monkeypatch.setattr(cunufft_wrapper, "_execute_inverse_cunufft_batch", fake_exec)

    result = cunufft_wrapper.execute_inverse_cunufft_super_batch(
        q_coords=np.array([[0.0], [1.0]], dtype=np.float64),
        weights=np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j],
                [3.0 + 0.0j, 4.0 + 0.0j],
                [5.0 + 0.0j, 6.0 + 0.0j],
                [7.0 + 0.0j, 8.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
        real_coords=np.array([[0.0], [0.5]], dtype=np.float64),
    )

    assert calls[0] == 4
    assert calls[1:] == [2, 2]
    assert result.shape == (4, 2)


def test_execute_inverse_cunufft_super_batch_does_not_retry_non_memory_runtime_error(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_exec(**kwargs):
        calls.append(kwargs["weights_arr"].shape[0])
        raise RuntimeError("bad-shape")

    monkeypatch.setattr(cunufft_wrapper, "_execute_inverse_cunufft_batch", fake_exec)

    with pytest.raises(RuntimeError, match="bad-shape"):
        cunufft_wrapper.execute_inverse_cunufft_super_batch(
            q_coords=np.array([[0.0], [1.0]], dtype=np.float64),
            weights=np.array(
                [
                    [1.0 + 0.0j, 2.0 + 0.0j],
                    [3.0 + 0.0j, 4.0 + 0.0j],
                    [5.0 + 0.0j, 6.0 + 0.0j],
                ],
                dtype=np.complex128,
            ),
            real_coords=np.array([[0.0], [0.5]], dtype=np.float64),
        )

    assert calls == [3]


def test_batched_type3_retries_resource_like_runtime_error(monkeypatch: pytest.MonkeyPatch):
    class FakeOOM(MemoryError):
        pass

    class FakeCUDARuntimeError(RuntimeError):
        pass

    class FakeCUDADriverError(RuntimeError):
        pass

    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=FakeOOM,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=FakeCUDARuntimeError),
            driver=SimpleNamespace(CUDADriverError=FakeCUDADriverError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    calls = []

    def fake_launch(dim, resident_cols, d_w, target_cols, eps, inverse):
        calls.append(len(target_cols[0]))
        if len(calls) == 1:
            raise RuntimeError("shared memory exhausted")
        return np.full(len(target_cols[0]), 13.0 + 0.0j, dtype=np.complex128)

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(cunufft_wrapper, "_adaptive_gpu_launch", fake_launch)

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=4,
    )

    np.testing.assert_allclose(result, np.full(4, 13.0 + 0.0j, dtype=np.complex128))
    assert calls[0] == 4
    assert calls[1:] == [2, 2]


def test_batched_type3_reraises_non_retryable_runtime_error(monkeypatch: pytest.MonkeyPatch):
    class FakeOOM(MemoryError):
        pass

    class FakeCUDARuntimeError(RuntimeError):
        pass

    class FakeCUDADriverError(RuntimeError):
        pass

    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=FakeOOM,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=FakeCUDARuntimeError),
            driver=SimpleNamespace(CUDADriverError=FakeCUDADriverError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_adaptive_gpu_launch",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad-shape")),
    )
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )

    with pytest.raises(RuntimeError, match="bad-shape"):
        cunufft_wrapper.execute_cunufft(
            np.array([[0.0]], dtype=np.float64),
            np.array([1.0 + 0.0j], dtype=np.complex128),
            np.array([[1.0]], dtype=np.float64),
            min_chunk=1,
            max_chunk=1,
        )


def test_inverse_batch_reraises_non_retryable_runtime_error(monkeypatch: pytest.MonkeyPatch):
    class FakeOOM(MemoryError):
        pass

    class FakeCUDARuntimeError(RuntimeError):
        pass

    class FakeCUDADriverError(RuntimeError):
        pass

    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=FakeOOM,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=FakeCUDARuntimeError),
            driver=SimpleNamespace(CUDADriverError=FakeCUDADriverError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_execute_inverse_batch_gpu",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad-shape")),
    )
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )

    with pytest.raises(RuntimeError, match="bad-shape"):
        cunufft_wrapper.execute_inverse_cunufft_batch(
            q_coords=np.array([[0.0]], dtype=np.float64),
            weights=np.array([[1.0 + 0.0j]], dtype=np.complex128),
            real_coords=np.array([[0.0]], dtype=np.float64),
            min_chunk=1,
            max_chunk=1,
        )


def test_set_cpu_only_false_reprobes_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cunufft_wrapper, "_CPU_ONLY", True)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", False)
    monkeypatch.setattr(cunufft_wrapper, "cp", None)

    def fake_probe():
        cunufft_wrapper.cp = "gpu"
        cunufft_wrapper._GPU_AVAILABLE = True

    monkeypatch.setattr(cunufft_wrapper, "_probe_gpu_backend", fake_probe)

    cunufft_wrapper.set_cpu_only(False)

    assert cunufft_wrapper._CPU_ONLY is False
    assert cunufft_wrapper._GPU_AVAILABLE is True
    assert cunufft_wrapper.cp == "gpu"


def test_execute_cunufft_prefer_cpu_uses_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: np.array([7.0 + 0.0j]),
    )
    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j]),
        np.array([[1.0]], dtype=np.float64),
        prefer_cpu=True,
    )
    np.testing.assert_allclose(result, np.array([7.0 + 0.0j]))


def test_execute_inverse_cunufft_prefer_cpu_uses_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: np.array([9.0 + 0.0j]),
    )
    result = cunufft_wrapper.execute_inverse_cunufft(
        np.array([[1.0]], dtype=np.float64),
        np.array([1.0 + 0.0j]),
        np.array([[0.0]], dtype=np.float64),
        prefer_cpu=True,
    )
    np.testing.assert_allclose(result, np.array([9.0 + 0.0j]))


def test_batched_type3_calls_cleanup_before_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=RuntimeError),
            driver=SimpleNamespace(CUDADriverError=RuntimeError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    cleanup_calls = {"count": 0}
    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(cunufft_wrapper, "_adaptive_gpu_launch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cunufft_wrapper, "free_gpu_memory", lambda: cleanup_calls.__setitem__("count", cleanup_calls["count"] + 1))
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: np.array([11.0 + 0.0j]),
    )

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j]),
        np.array([[1.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=1,
    )

    np.testing.assert_allclose(result, np.array([11.0 + 0.0j]))
    assert cleanup_calls["count"] >= 1


def test_batched_type3_does_not_flush_on_every_successful_chunk(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=RuntimeError),
            driver=SimpleNamespace(CUDADriverError=RuntimeError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    cleanup_calls = {"count": 0}

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_adaptive_gpu_launch",
        lambda *args, **kwargs: np.array([13.0 + 0.0j], dtype=np.complex128),
    )
    monkeypatch.setattr(
        cunufft_wrapper,
        "free_gpu_memory",
        lambda: cleanup_calls.__setitem__("count", cleanup_calls["count"] + 1),
    )

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([[1.0], [2.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=1,
    )

    np.testing.assert_allclose(result, np.array([13.0 + 0.0j, 13.0 + 0.0j]))
    assert cleanup_calls["count"] == 1


def test_build_gpu_launch_kwargs_uses_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOSAIC_NUFFT_GPU_METHOD", raising=False)
    monkeypatch.delenv("MOSAIC_NUFFT_GPU_KEREVALMETH", raising=False)
    monkeypatch.delenv("MOSAIC_NUFFT_GPU_MAXBATCHSIZE", raising=False)
    monkeypatch.delenv("MOSAIC_NUFFT_GPU_SPREADINTERPONLY", raising=False)
    monkeypatch.delenv("MOSAIC_NUFFT_GPU_STREAM", raising=False)

    kwargs = cunufft_wrapper._build_gpu_launch_kwargs(gpu_maxsubprobsize=8)

    assert kwargs["gpu_method"] == 1
    assert kwargs["gpu_kerevalmeth"] == 1
    assert kwargs["gpu_maxsubprobsize"] == 8
    assert kwargs["gpu_maxbatchsize"] == 1
    assert kwargs["gpu_spreadinterponly"] == 1
    assert "gpu_stream" not in kwargs


def test_build_gpu_launch_kwargs_accepts_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_NUFFT_GPU_METHOD", "2")
    monkeypatch.setenv("MOSAIC_NUFFT_GPU_KEREVALMETH", "0")
    monkeypatch.setenv("MOSAIC_NUFFT_GPU_MAXBATCHSIZE", "3")
    monkeypatch.setenv("MOSAIC_NUFFT_GPU_SPREADINTERPONLY", "0")

    kwargs = cunufft_wrapper._build_gpu_launch_kwargs(gpu_maxsubprobsize=4)

    assert kwargs["gpu_method"] == 2
    assert kwargs["gpu_kerevalmeth"] == 0
    assert kwargs["gpu_maxsubprobsize"] == 4
    assert kwargs["gpu_maxbatchsize"] == 3
    assert kwargs["gpu_spreadinterponly"] == 0


def test_build_gpu_launch_kwargs_can_include_current_stream(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(ptr=12345),
        )
    )
    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setenv("MOSAIC_NUFFT_GPU_STREAM", "current")

    kwargs = cunufft_wrapper._build_gpu_launch_kwargs(gpu_maxsubprobsize=4)

    assert kwargs["gpu_stream"] == 12345


def test_experimental_overlap_flag_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP", "1")
    assert cunufft_wrapper._experimental_overlap_enabled() is True
    monkeypatch.setenv("MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP", "0")
    assert cunufft_wrapper._experimental_overlap_enabled() is False


def test_copy_device_to_host_falls_back_when_pinned_unavailable(monkeypatch: pytest.MonkeyPatch):
    class FakeArray:
        shape = (2,)
        dtype = np.dtype(np.complex128)

    fake_cp = SimpleNamespace(
        asnumpy=lambda x, out=None: np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128),
    )
    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setenv("MOSAIC_NUFFT_PINNED_HOST", "1")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupyx":
            raise ImportError("no cupyx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    host = cunufft_wrapper._copy_device_to_host(FakeArray())
    np.testing.assert_allclose(host, np.array([1.0 + 0.0j, 2.0 + 0.0j]))


def test_experimental_overlap_logs_diagnostics(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=RuntimeError),
            driver=SimpleNamespace(CUDADriverError=RuntimeError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    debug_messages = []

    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_adaptive_gpu_launch",
        lambda *args, **kwargs: np.array([13.0 + 0.0j], dtype=np.complex128),
    )
    monkeypatch.setattr(
        cunufft_wrapper.logger,
        "debug",
        lambda msg, *args: debug_messages.append(msg % args if args else msg),
    )
    monkeypatch.setenv("MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP", "1")

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([[1.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=1,
    )

    np.testing.assert_allclose(result, np.array([13.0 + 0.0j]))
    assert any("Experimental overlap requested" in msg for msg in debug_messages)
    assert any("timings" in msg for msg in debug_messages)


def test_wrapper_telemetry_captures_chunk_summary(monkeypatch: pytest.MonkeyPatch):
    fake_cp = SimpleNamespace(
        cuda=SimpleNamespace(
            memory=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
            ),
            runtime=SimpleNamespace(CUDARuntimeError=RuntimeError),
            driver=SimpleNamespace(CUDADriverError=RuntimeError),
        ),
        asnumpy=lambda x: np.asarray(x),
    )
    monkeypatch.setattr(cunufft_wrapper, "cp", fake_cp)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10**9)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_adaptive_gpu_launch",
        lambda *args, **kwargs: np.array([13.0 + 0.0j], dtype=np.complex128),
    )
    monkeypatch.setenv("MOSAIC_NUFFT_CAPTURE_TELEMETRY", "1")

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([[1.0], [2.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=1,
    )

    np.testing.assert_allclose(result, np.array([13.0 + 0.0j, 13.0 + 0.0j]))
    telemetry = cunufft_wrapper.get_last_nufft_telemetry()
    assert telemetry["chunk_count"] == 2
    assert telemetry["full_target_fit_in_one_chunk"] is False
    assert telemetry["n_sources"] == 1
    assert telemetry["n_targets"] == 2
    assert telemetry["resident_bytes"] > 0
    assert telemetry["final_d2h_bytes"] == np.asarray(result).nbytes


def test_wrapper_telemetry_records_fallback_reason(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_NUFFT_CAPTURE_TELEMETRY", "1")
    monkeypatch.setattr(cunufft_wrapper, "_CPU_ONLY", False)
    monkeypatch.setattr(cunufft_wrapper, "_GPU_AVAILABLE", True)
    monkeypatch.setattr(cunufft_wrapper, "_ensure_gpu_kernels", lambda: None)
    monkeypatch.setattr(cunufft_wrapper, "_as_device", lambda arr, allow_fail=False: np.asarray(arr))
    monkeypatch.setattr(cunufft_wrapper, "_contig", lambda x: x)
    monkeypatch.setattr(cunufft_wrapper, "_free_mem_bytes", lambda: 10)
    monkeypatch.setattr(
        cunufft_wrapper,
        "_cpu_fallback",
        lambda *args, **kwargs: np.array([1.0 + 0.0j]),
    )

    result = cunufft_wrapper.execute_cunufft(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0 + 0.0j], dtype=np.complex128),
        np.array([[1.0]], dtype=np.float64),
        min_chunk=1,
        max_chunk=1,
    )

    np.testing.assert_allclose(result, np.array([1.0 + 0.0j]))
    telemetry = cunufft_wrapper.get_last_nufft_telemetry()
    assert telemetry["fallback_reason"] == "budget-exhausted"


@pytest.mark.skipif(
    not cunufft_wrapper._GPU_AVAILABLE,
    reason="Requires GPU/cuFINUFFT to validate gpu_spreadinterponly numerically.",
)
def test_inverse_batch_matches_cpu_reference_with_current_gpu_spreadinterponly():
    q_coords = np.array([[0.0], [1.25]], dtype=np.float64)
    real_coords = np.array([[0.0], [0.5]], dtype=np.float64)
    weights = np.array(
        [[1.0 + 0.0j, 0.5 + 0.25j], [0.75 + 0.0j, 0.25 + 0.5j]],
        dtype=np.complex128,
    )
    expected = np.stack(
        [
            cunufft_wrapper._direct_cpu_fallback(
                real_coords,
                weights[index],
                q_coords,
                inverse=True,
                batch=8,
            )
            for index in range(weights.shape[0])
        ],
        axis=0,
    )
    result = cunufft_wrapper.execute_inverse_cunufft_batch(
        q_coords=q_coords,
        weights=weights,
        real_coords=real_coords,
        eps=1e-12,
        gpu_only=True,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)
