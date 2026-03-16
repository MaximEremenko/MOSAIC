from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.qspace.masking.mask_strategies import EqBasedStrategy, get_last_eq_mask_telemetry
from core.scattering.grid import (
    generate_q_space_grid,
    get_last_qspace_grid_telemetry,
    reciprocal_space_points_counter,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dimension: int
    supercell: tuple[int, ...]
    interval: dict[str, float]
    equation: str


def _case_2d() -> BenchmarkCase:
    return BenchmarkCase(
        name="2d_equation",
        dimension=2,
        supercell=(256, 256),
        interval={
            "h_start": 0.0,
            "h_end": 1.0,
            "k_start": 0.0,
            "k_end": 1.0,
        },
        equation=(
            "((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= (0.22)**2"
        ),
    )


def _case_3d() -> BenchmarkCase:
    return BenchmarkCase(
        name="3d_equation",
        dimension=3,
        supercell=(64, 64, 64),
        interval={
            "h_start": 0.0,
            "h_end": 1.0,
            "k_start": 0.0,
            "k_end": 1.0,
            "l_start": 0.0,
            "l_end": 1.0,
        },
        equation=(
            "(((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= (0.1876)**2) & "
            "(((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= (0.2501)**2)"
        ),
    )


def _cases(name: str) -> list[BenchmarkCase]:
    if name == "2d":
        return [_case_2d()]
    if name == "3d":
        return [_case_3d()]
    return [_case_2d(), _case_3d()]


def _forced_block_points(total_points: int) -> int:
    return max(1, min(50_000, max(total_points // 8, 1)))


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    original = {key: os.getenv(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _path_env(total_points: int, path_mode: str) -> dict[str, str | None]:
    if path_mode == "full":
        return {"MOSAIC_QSPACE_BLOCK_POINTS": str(total_points + 1)}
    if path_mode == "blockwise":
        return {"MOSAIC_QSPACE_BLOCK_POINTS": str(_forced_block_points(total_points))}
    return {}


def _backend_env(backend_mode: str) -> dict[str, str | None]:
    if backend_mode == "auto":
        return {"MOSAIC_MASK_EQUATION_BACKEND": None}
    return {"MOSAIC_MASK_EQUATION_BACKEND": backend_mode}


def _run_once(case: BenchmarkCase, *, backend_mode: str, path_mode: str) -> dict:
    supercell = np.asarray(case.supercell, dtype=np.float64)
    total_points = reciprocal_space_points_counter(case.interval, supercell)
    env = {
        "MOSAIC_MASK_CAPTURE_TELEMETRY": "1",
        "MOSAIC_QSPACE_CAPTURE_TELEMETRY": "1",
        **_backend_env(backend_mode),
        **_path_env(total_points, path_mode),
    }
    B_ = np.eye(case.dimension, dtype=np.float64)

    with _temporary_env(env):
        strategy = EqBasedStrategy(case.equation)
        started = time.perf_counter()
        q_grid = generate_q_space_grid(
            interval=case.interval,
            B_=B_,
            mask_parameters={},
            mask_strategy=strategy,
            supercell=supercell,
        )
        wall_seconds = time.perf_counter() - started
        qspace = get_last_qspace_grid_telemetry()
        mask = get_last_eq_mask_telemetry()

    return {
        "case": case.name,
        "dimension": case.dimension,
        "supercell": list(case.supercell),
        "requested_backend": backend_mode,
        "requested_path": path_mode,
        "total_points": int(total_points),
        "q_points": int(len(q_grid)),
        "wall_seconds": float(wall_seconds),
        "qspace": qspace,
        "mask": mask,
    }


def _variants() -> list[tuple[str, str]]:
    return [
        ("cpu", "full"),
        ("cpu", "blockwise"),
        ("auto", "blockwise"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded q-space masking benchmark for representative 2D/3D equation-mask cases.",
    )
    parser.add_argument(
        "--case",
        choices=("2d", "3d", "all"),
        default="all",
        help="Representative masking case to run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per case/variant before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Timed runs per case/variant.",
    )
    args = parser.parse_args()

    results = []
    for case in _cases(args.case):
        for backend_mode, path_mode in _variants():
            for _ in range(max(args.warmup, 0)):
                _run_once(case, backend_mode=backend_mode, path_mode=path_mode)
            for _ in range(max(args.repeat, 1)):
                results.append(
                    _run_once(case, backend_mode=backend_mode, path_mode=path_mode)
                )

    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
