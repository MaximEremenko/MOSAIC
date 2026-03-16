import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.qspace.masking.mask_strategies as mask_strategies
from core.qspace.masking.mask_strategies import (
    CustomReciprocalSpacePointsStrategy,
    DefaultMaskStrategy,
    EqBasedStrategy,
)
from core.qspace.masking.shape_strategies import (
    CircleShapeStrategy,
    EllipsoidShapeStrategy,
    IntervalShapeStrategy,
    SphereShapeStrategy,
)
from core.scattering.grid import generate_q_space_grid, get_last_qspace_grid_telemetry


def test_eq_based_strategy_masks_points():
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mask = strategy.generate_mask(mesh)
    assert mask.tolist() == [True, False]


def test_eq_based_strategy_pickle_round_trip_rebuilds_lazy_callables():
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0], [-1.0]], dtype=np.float64)

    first_mask = strategy.generate_mask(mesh)
    assert strategy._f_cpu is not None

    restored = pickle.loads(pickle.dumps(strategy))

    assert restored._f_cpu is None
    assert restored._f_gpu is None
    np.testing.assert_array_equal(restored.generate_mask(mesh), first_mask)
    assert restored._f_cpu is not None


def test_resolve_mask_backend_prefers_cpu_for_small_workload():
    decision = mask_strategies._resolve_mask_backend(
        point_count=1_000,
        dim=3,
        gpu_available=True,
        free_bytes=8 << 30,
        total_bytes=8 << 30,
        min_points=250_000,
        backend_override="auto",
    )

    assert decision["backend"] == "cpu"
    assert decision["reason"] == "below-min-points"


def test_resolve_mask_backend_allows_gpu_for_large_workload_with_headroom():
    decision = mask_strategies._resolve_mask_backend(
        point_count=500_000,
        dim=3,
        gpu_available=True,
        free_bytes=8 << 30,
        total_bytes=8 << 30,
        min_points=250_000,
        backend_override="auto",
    )

    assert decision["backend"] == "gpu"
    assert decision["reason"] == "gpu-eligible"
    assert decision["estimated_bytes"] > 0


def test_resolve_mask_backend_respects_reserve_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_MASK_EQUATION_GPU_RESERVE_BYTES", str(3 << 30))

    decision = mask_strategies._resolve_mask_backend(
        point_count=500_000,
        dim=3,
        gpu_available=True,
        free_bytes=8 << 30,
        total_bytes=8 << 30,
        min_points=250_000,
        backend_override="auto",
    )

    assert decision["reserve_bytes"] == 3 << 30


def test_eq_based_strategy_uses_cpu_when_gpu_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0], [-1.0]], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_MASK_EQUATION_BACKEND", "gpu")
    monkeypatch.setattr(mask_strategies, "_load_cupy", lambda: None)

    mask = strategy.generate_mask(mesh)

    assert mask.tolist() == [True, False]


def test_eq_based_strategy_gpu_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch):
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0], [-1.0]], dtype=np.float64)

    monkeypatch.setattr(
        strategy,
        "_backend_decision",
        lambda point_count, dim: (
            {
                "backend": "gpu",
                "reason": "gpu-eligible",
                "free_bytes": 8 << 30,
                "reserve_bytes": 1 << 30,
                "estimated_bytes": 1 << 20,
            },
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        strategy,
        "_ensure_gpu_callable",
        lambda cp_mod: (_ for _ in ()).throw(RuntimeError("gpu lambdify failed")),
    )

    mask = strategy.generate_mask(mesh)

    assert mask.tolist() == [True, False]


def test_eq_based_strategy_telemetry_records_backend_and_reason(monkeypatch: pytest.MonkeyPatch):
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0], [-1.0]], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_MASK_CAPTURE_TELEMETRY", "1")
    monkeypatch.setattr(mask_strategies, "_load_cupy", lambda: None)

    mask = strategy.generate_mask(mesh)
    telemetry = mask_strategies.get_last_eq_mask_telemetry()

    assert mask.tolist() == [True, False]
    assert telemetry["backend_used"] == "cpu"
    assert telemetry["decision_reason"] == "gpu-unavailable"
    assert telemetry["point_count"] == 2
    assert telemetry["estimated_bytes_per_point"] >= 32
    assert telemetry["duration_seconds"] >= 0.0


def test_generate_q_space_grid_keeps_numpy_output_with_equation_strategy():
    strategy = EqBasedStrategy("h >= 0")
    q_grid = generate_q_space_grid(
        interval={"h_start": 0.0, "h_end": 0.5},
        B_=np.array([[1.0]], dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([2.0]),
    )

    assert isinstance(q_grid, np.ndarray)
    assert q_grid.ndim == 2
    assert q_grid.shape[1] == 1


def test_generate_q_space_grid_telemetry_full_path(monkeypatch: pytest.MonkeyPatch):
    strategy = EqBasedStrategy("h >= 0")

    monkeypatch.setenv("MOSAIC_QSPACE_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    q_grid = generate_q_space_grid(
        interval={"h_start": 0.0, "h_end": 0.5},
        B_=np.array([[1.0]], dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([2.0]),
    )
    telemetry = get_last_qspace_grid_telemetry()

    assert isinstance(q_grid, np.ndarray)
    assert telemetry["mode"] == "full"
    assert telemetry["decision_reason"] == "below-block-threshold"
    assert telemetry["mask_strategy"] == "EqBasedStrategy"
    assert telemetry["blockwise_safe"] is True
    assert telemetry["block_count"] == 1
    assert telemetry["accepted_points"] == len(q_grid)
    assert telemetry["accepted_fraction"] == pytest.approx(1.0)
    assert telemetry["mesh_build_seconds"] >= 0.0
    assert telemetry["mask_seconds"] >= 0.0
    assert telemetry["q_conversion_seconds"] >= 0.0


def test_generate_q_space_grid_blockwise_matches_full_for_equation_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EqBasedStrategy("(h + k) <= 0.75")
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
    }
    B_ = np.eye(2, dtype=np.float64)
    supercell = np.array([4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "3")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)


def test_generate_q_space_grid_telemetry_blockwise_path(monkeypatch: pytest.MonkeyPatch):
    strategy = EqBasedStrategy("(h + k) <= 0.75")
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
    }

    monkeypatch.setenv("MOSAIC_QSPACE_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_MASK_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "3")
    q_grid = generate_q_space_grid(
        interval=interval,
        B_=np.eye(2, dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([4.0, 4.0], dtype=np.float64),
    )
    telemetry = get_last_qspace_grid_telemetry()

    assert isinstance(q_grid, np.ndarray)
    assert telemetry["mode"] == "blockwise"
    assert telemetry["decision_reason"] == "blockwise-safe-strategy"
    assert telemetry["mask_strategy"] == "EqBasedStrategy"
    assert telemetry["blockwise_safe"] is True
    assert telemetry["block_count"] > 1
    assert telemetry["accepted_points"] == len(q_grid)
    assert telemetry["accepted_fraction"] < 1.0
    assert telemetry["mesh_build_seconds"] >= 0.0
    assert telemetry["mask_seconds"] >= 0.0
    assert telemetry["q_conversion_seconds"] >= 0.0
    assert telemetry["total_seconds"] >= 0.0
    assert isinstance(telemetry["blocks"], list)
    assert len(telemetry["blocks"]) == telemetry["block_count"]
    assert telemetry["mask_backend_counts"]["cpu"] >= 1
    assert telemetry["blocks"][0]["mask_strategy"] == "EqBasedStrategy"


def test_generate_q_space_grid_blockwise_calls_strategy_multiple_times(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EqBasedStrategy("h >= 0")
    interval = {"h_start": 0.0, "h_end": 0.875}
    B_ = np.array([[1.0]], dtype=np.float64)
    supercell = np.array([8.0], dtype=np.float64)
    calls = {"count": 0}
    original_generate_mask = strategy.generate_mask

    def counted_generate_mask(hkl_mesh):
        calls["count"] += 1
        return original_generate_mask(hkl_mesh)

    monkeypatch.setattr(strategy, "generate_mask", counted_generate_mask)
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")

    q_grid = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    assert calls["count"] > 1
    assert isinstance(q_grid, np.ndarray)
    assert q_grid.shape[1] == 1


def test_generate_q_space_grid_blockwise_matches_full_for_default_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = DefaultMaskStrategy()
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
    }
    B_ = np.eye(2, dtype=np.float64)
    supercell = np.array([4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "3")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)


def test_generate_q_space_grid_default_strategy_reports_cpu_backend(monkeypatch: pytest.MonkeyPatch):
    strategy = DefaultMaskStrategy()

    monkeypatch.setenv("MOSAIC_QSPACE_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")
    q_grid = generate_q_space_grid(
        interval={"h_start": 0.0, "h_end": 0.875},
        B_=np.array([[1.0]], dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([8.0], dtype=np.float64),
    )
    telemetry = get_last_qspace_grid_telemetry()

    assert isinstance(q_grid, np.ndarray)
    assert telemetry["mask_strategy"] == "DefaultMaskStrategy"
    assert telemetry["mask_backend_counts"]["cpu"] >= 1
    assert telemetry["blocks"][0]["mask_reason"] == "strategy-mask"


def test_generate_q_space_grid_blockwise_matches_full_for_ellipsoid_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EllipsoidShapeStrategy(
        spetial_points=np.array([[0.25, 0.25, 0.25]], dtype=np.float64),
        axes=np.array([0.30, 0.30, 0.30], dtype=np.float64),
        theta=0.0,
        phi=0.0,
    )
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
        "l_start": 0.0,
        "l_end": 0.75,
    }
    B_ = np.eye(3, dtype=np.float64)
    supercell = np.array([4.0, 4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "5")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)


def test_generate_q_space_grid_blockwise_matches_full_for_interval_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = IntervalShapeStrategy(
        {
            "specialPoints": [
                {
                    "radius": 0.25,
                    "coordinate": 0.25,
                }
            ]
        }
    )
    interval = {"h_start": 0.0, "h_end": 0.875}
    B_ = np.array([[1.0]], dtype=np.float64)
    supercell = np.array([8.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")

    q_grid = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )
    telemetry = get_last_qspace_grid_telemetry()

    np.testing.assert_allclose(q_grid, expected)
    assert telemetry["mode"] == "blockwise"
    assert telemetry["decision_reason"] == "blockwise-safe-strategy"


def test_generate_q_space_grid_blockwise_matches_full_for_circle_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = CircleShapeStrategy(
        {
            "specialPoints": [
                {
                    "radius": 0.30,
                    "coordinate": [0.25, 0.25],
                }
            ]
        }
    )
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
    }
    B_ = np.eye(2, dtype=np.float64)
    supercell = np.array([4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "3")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)


def test_generate_q_space_grid_blockwise_matches_full_for_sphere_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = SphereShapeStrategy(
        {
            "specialPoints": [
                {
                    "radius": 0.35,
                    "coordinate": [0.25, 0.25, 0.25],
                }
            ]
        }
    )
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
        "l_start": 0.0,
        "l_end": 0.75,
    }
    B_ = np.eye(3, dtype=np.float64)
    supercell = np.array([4.0, 4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "7")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)


def test_custom_reciprocal_space_points_strategy_blockwise_matches_full_and_caches_reads(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = CustomReciprocalSpacePointsStrategy(
        file_path="ignored.txt",
        ih=np.array([0.0, 1.0], dtype=np.float64),
        ik=np.array([0.0, 1.0], dtype=np.float64),
        il=np.array([0.0, 1.0], dtype=np.float64),
    )
    interval = {
        "h_start": 0.0,
        "h_end": 0.5,
        "k_start": 0.0,
        "k_end": 0.5,
        "l_start": 0.0,
        "l_end": 0.5,
    }
    B_ = np.eye(3, dtype=np.float64)
    supercell = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    calls = {"count": 0}
    reflections_df = pd.DataFrame(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [1.5, 1.5, 1.5],
            ],
            dtype=np.float64,
        )
    )

    def fake_read_table(*args, **kwargs):
        calls["count"] += 1
        return reflections_df

    monkeypatch.setattr(mask_strategies.pd, "read_table", fake_read_table)
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)
    assert calls["count"] == 1


def test_custom_reciprocal_space_points_strategy_reports_generic_cpu_telemetry(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = CustomReciprocalSpacePointsStrategy(
        file_path="ignored.txt",
        ih=np.array([0.0, 1.0], dtype=np.float64),
        ik=np.array([0.0, 1.0], dtype=np.float64),
        il=np.array([0.0, 1.0], dtype=np.float64),
    )
    reflections_df = pd.DataFrame(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )

    monkeypatch.setattr(mask_strategies.pd, "read_table", lambda *args, **kwargs: reflections_df)
    monkeypatch.setenv("MOSAIC_QSPACE_CAPTURE_TELEMETRY", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")

    q_grid = generate_q_space_grid(
        interval={
            "h_start": 0.0,
            "h_end": 0.5,
            "k_start": 0.0,
            "k_end": 0.0,
            "l_start": 0.0,
            "l_end": 0.0,
        },
        B_=np.eye(3, dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([2.0, 1.0, 1.0], dtype=np.float64),
    )
    telemetry = get_last_qspace_grid_telemetry()

    assert isinstance(q_grid, np.ndarray)
    assert telemetry["mask_strategy"] == "CustomReciprocalSpacePointsStrategy"
    assert telemetry["mask_backend_counts"]["cpu"] >= 1
    assert telemetry["blocks"][0]["mask_reason"] == "strategy-mask"


def test_generate_q_space_grid_blockwise_preserves_empty_result(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EqBasedStrategy("h < 0")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "2")

    q_grid = generate_q_space_grid(
        interval={"h_start": 0.0, "h_end": 0.875},
        B_=np.array([[1.0]], dtype=np.float64),
        mask_parameters={},
        mask_strategy=strategy,
        supercell=np.array([8.0], dtype=np.float64),
    )

    assert isinstance(q_grid, np.ndarray)
    assert q_grid.shape == (0, 1)


@pytest.mark.skipif(
    mask_strategies._load_cupy() is None,
    reason="Requires CuPy with a usable GPU runtime.",
)
def test_eq_based_strategy_gpu_matches_cpu_for_representative_equation(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EqBasedStrategy(
        "((Mod(h,1.0) - 0.5)**2 <= 0.25) & (Heaviside(k, 0.0) >= 0.0)"
    )
    mesh = np.array(
        [
            [0.0, -0.5, 0.0],
            [0.25, 0.0, 0.0],
            [0.75, 0.5, 0.0],
            [1.25, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    h_vals, k_vals, l_vals = strategy._split_components(mesh)
    expected = strategy._cpu_mask(h_vals, k_vals, l_vals)

    monkeypatch.setenv("MOSAIC_MASK_EQUATION_BACKEND", "gpu")
    monkeypatch.setenv("MOSAIC_MASK_EQUATION_GPU_MIN_POINTS", "1")

    result = strategy.generate_mask(mesh)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(
    mask_strategies._load_cupy() is None,
    reason="Requires CuPy with a usable GPU runtime.",
)
def test_generate_q_space_grid_blockwise_gpu_matches_full_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    strategy = EqBasedStrategy(
        "((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= 0.5"
    )
    interval = {
        "h_start": 0.0,
        "h_end": 0.75,
        "k_start": 0.0,
        "k_end": 0.75,
    }
    B_ = np.eye(2, dtype=np.float64)
    supercell = np.array([4.0, 4.0], dtype=np.float64)

    monkeypatch.setenv("MOSAIC_MASK_EQUATION_BACKEND", "cpu")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "1000")
    expected = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    monkeypatch.setenv("MOSAIC_MASK_EQUATION_BACKEND", "gpu")
    monkeypatch.setenv("MOSAIC_MASK_EQUATION_GPU_MIN_POINTS", "1")
    monkeypatch.setenv("MOSAIC_QSPACE_BLOCK_POINTS", "3")
    result = generate_q_space_grid(
        interval=interval,
        B_=B_,
        mask_parameters={},
        mask_strategy=strategy,
        supercell=supercell,
    )

    np.testing.assert_allclose(result, expected)
