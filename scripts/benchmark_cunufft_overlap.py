from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from statistics import median

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.adapters import cunufft_wrapper


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _configure_case_env(*, experimental: bool) -> None:
    for key in (
        "MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP",
        "MOSAIC_NUFFT_CAPTURE_TELEMETRY",
        "MOSAIC_NUFFT_PINNED_HOST",
        "MOSAIC_NUFFT_GPU_STREAM",
    ):
        os.environ.pop(key, None)
    os.environ["MOSAIC_NUFFT_CAPTURE_TELEMETRY"] = "1"
    if experimental:
        os.environ["MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP"] = "1"
        os.environ["MOSAIC_NUFFT_PINNED_HOST"] = "1"
        os.environ["MOSAIC_NUFFT_GPU_STREAM"] = "current"


def _telemetry_summary(telemetry: dict | None) -> dict[str, object] | None:
    if telemetry is None:
        return None
    upload = float(telemetry.get("total_upload_seconds", 0.0))
    launch = float(telemetry.get("total_launch_seconds", 0.0))
    download = float(telemetry.get("total_download_seconds", 0.0))
    transfer = upload + download
    if launch > transfer:
        dominant = "compute-bound"
    elif transfer > launch:
        dominant = "transfer-bound"
    else:
        dominant = "balanced"
    return {
        "mode": telemetry.get("mode"),
        "n_sources": telemetry.get("n_sources"),
        "n_targets": telemetry.get("n_targets"),
        "n_trans": telemetry.get("n_trans"),
        "effective_mem_frac": telemetry.get("effective_mem_frac"),
        "mem_policy_source": telemetry.get("mem_policy_source"),
        "resident_bytes": telemetry.get("resident_bytes"),
        "reserve_bytes": telemetry.get("reserve_bytes"),
        "incremental_launch_baseline_bytes": telemetry.get("incremental_launch_baseline_bytes"),
        "chunk_count": telemetry.get("chunk_count"),
        "full_target_fit_in_one_chunk": telemetry.get("full_target_fit_in_one_chunk"),
        "final_d2h_bytes": telemetry.get("final_d2h_bytes"),
        "used_pinned_host_copy": telemetry.get("used_pinned_host_copy"),
        "fallback_reason": telemetry.get("fallback_reason"),
        "total_upload_seconds": upload,
        "total_launch_seconds": launch,
        "total_download_seconds": download,
        "transfer_vs_launch_ratio": (transfer / launch) if launch > 0 else None,
        "dominant_phase": dominant,
    }


def _benchmark_sizes() -> tuple[int, int]:
    # For inverse type-3 the source side (q_coords + weights) stays resident
    # while the real-space targets are chunked. To stress chunking and any
    # overlap-related diagnostics meaningfully, keep q moderate and scale real.
    q_points = _env_int("MOSAIC_NUFFT_BENCH_Q_POINTS", 4_096 * 1_024)
    real_points = _env_int("MOSAIC_NUFFT_BENCH_REAL_POINTS", 8_192 * 1_024)
    if q_points <= 0 or real_points <= 0:
        raise SystemExit("Benchmark point counts must be positive integers.")
    return q_points, real_points


def _estimate_inverse_resident_bytes(*, q_points: int, n_trans: int = 2, dim: int = 1) -> int:
    coord_bytes = q_points * dim * 8
    weight_bytes = q_points * n_trans * 16
    source_scratch_bytes = int(q_points * cunufft_wrapper._SCRATCH_ALPHA * 16 * n_trans)
    return int(coord_bytes + weight_bytes + source_scratch_bytes)


def _guard_benchmark_geometry(*, q_points: int, real_points: int) -> None:
    free_vram = cunufft_wrapper._free_mem_bytes()
    if free_vram <= 0:
        return
    resident_bytes = _estimate_inverse_resident_bytes(q_points=q_points)
    if resident_bytes >= int(free_vram * 0.75):
        gib = 1024 ** 3
        raise SystemExit(
            "Benchmark geometry is not meaningful for the current inverse wrapper path: "
            f"the resident q/weight side alone is about {resident_bytes / gib:.2f} GiB "
            f"while free VRAM is about {free_vram / gib:.2f} GiB. "
            "For this harness, reduce MOSAIC_NUFFT_BENCH_Q_POINTS and increase "
            "MOSAIC_NUFFT_BENCH_REAL_POINTS instead, because the inverse path keeps "
            "q_coords resident and chunks real_coords."
        )
    if real_points <= q_points:
        raise SystemExit(
            "This harness is intended to stress chunked real-space targets. "
            "Set MOSAIC_NUFFT_BENCH_REAL_POINTS significantly larger than "
            "MOSAIC_NUFFT_BENCH_Q_POINTS."
        )


def _run_once(*, experimental: bool) -> dict[str, object]:
    _configure_case_env(experimental=experimental)

    q_points, real_points = _benchmark_sizes()
    q_coords = np.linspace(-4.0, 4.0, q_points, dtype=np.float64)[:, None]
    real_coords = np.linspace(-1.0, 1.0, real_points, dtype=np.float64)[:, None]
    weights = np.vstack(
        [
            np.exp(-0.1 * q_coords[:, 0] ** 2).astype(np.complex128),
            np.exp(-0.2 * q_coords[:, 0] ** 2).astype(np.complex128),
        ]
    )

    t0 = time.perf_counter()
    out = cunufft_wrapper.execute_inverse_cunufft_batch(
        q_coords=q_coords,
        weights=weights,
        real_coords=real_coords,
        eps=1e-12,
        gpu_only=True,
    )
    elapsed = time.perf_counter() - t0
    telemetry = cunufft_wrapper.get_last_nufft_telemetry()
    return {
        "experimental_overlap": experimental,
        "elapsed_seconds": elapsed,
        "shape": list(out.shape),
        "norm": float(np.linalg.norm(out)),
        "telemetry": _telemetry_summary(telemetry),
    }


def _run_case(*, experimental: bool, repeats: int = 5) -> dict[str, object]:
    # Warm up each mode once so we do not compare first-run CUDA/cuFINUFFT setup
    # against a warmed second run.
    warmup = _run_once(experimental=experimental)
    runs: list[dict[str, object]] = []
    for _ in range(repeats):
        runs.append(_run_once(experimental=experimental))
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    return {
        "experimental_overlap": experimental,
        "elapsed_seconds_median": median(elapsed),
        "elapsed_seconds_min": min(elapsed),
        "elapsed_seconds_max": max(elapsed),
        "elapsed_seconds_runs": elapsed,
        "shape": warmup["shape"],
        "norm": warmup["norm"],
        "warmup_elapsed_seconds": warmup["elapsed_seconds"],
    }


def main() -> int:
    if not cunufft_wrapper._GPU_AVAILABLE:
        raise SystemExit("GPU/cuFINUFFT is required for this benchmark harness.")
    q_points, real_points = _benchmark_sizes()
    _guard_benchmark_geometry(q_points=q_points, real_points=real_points)

    # Warm up global CUDA/cuFINUFFT state in both modes first, then benchmark.
    _run_once(experimental=False)
    _run_once(experimental=True)
    stable = _run_case(experimental=False)
    experimental = _run_case(experimental=True)
    print(
        json.dumps(
            {
                "stable": stable,
                "experimental": experimental,
                "interpretation": (
                    "Compare elapsed_seconds_median after warmup. The previous "
                    "single-run comparison was dominated by first-run CUDA/cuFINUFFT "
                    "setup cost. If experimental remains faster with matching "
                    "norm/shape across repeated warm runs, the experimental path is "
                    "worth studying further. If results are similar, the current "
                    "path is still effectively serialized."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
