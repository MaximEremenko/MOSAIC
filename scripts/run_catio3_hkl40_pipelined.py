#!/usr/bin/env python3
"""Pipelined hkl40 launcher: keep GPUs AND CPUs loaded simultaneously.

Each MOSAIC case is internally phase-serial: a GPU-heavy segment (stage-1
prewarm + residual transforms) followed by a CPU/IO-heavy segment (finalize,
publish, decoder, extraction) during which the GPUs idle. The four hkl40
cases (all / sphere / rod / rest) are scientifically independent, so this
launcher overlaps them: when a case's residual phase finishes (its GPUs go
quiet), the NEXT case's GPU phase starts immediately — the finished case's
CPU tail runs concurrently with the new case's GPU work. Node utilization
approaches the sum of the streams instead of their sequence.

GPU-phase serialization is by construction: at most one case is in its
residual segment at a time (the next case is released on the
"Residual-field finished" log line). Cases already residual-complete (pure
resume) are released immediately — they have no GPU segment at all.

This is the single-node shape of the intended HPC layout, where cases and
configs fan out across nodes against a shared stage-1 payload store.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ["all", "sphere", "rod", "rest"]
RESIDUAL_DONE_MARKER = "Residual-field finished"


def build_env() -> dict[str, str]:
    site = ROOT / ".venv" / "lib" / "python3.11" / "site-packages"
    cuda_libs = ":".join(
        str(site / "nvidia" / pkg / "lib")
        for pkg in ("cuda_runtime", "cufft", "nvjitlink", "cublas", "cuda_nvrtc")
    )
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = cuda_libs + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    # No device literals: cases size themselves (one worker per visible
    # GPU); a SLURM allocation constrains visibility via its cgroup.
    scratch = ROOT / ".dask-local"
    env.setdefault("DASK_LOCAL_DIR", str(scratch))
    for name, sub in (
        ("MOSAIC_RESIDUAL_LATTICE_SCRATCH", "hkl40-lattice"),
        ("MOSAIC_WORKER_SCRATCH_ROOT", "hkl40-worker"),
        ("MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT", "hkl40-shard"),
    ):
        path = scratch / sub
        path.mkdir(parents=True, exist_ok=True)
        env.setdefault(name, str(path))
    env.setdefault("MOSAIC_NUFFT_EXECUTION_POLICY", "gpu-required")
    # Intra-worker double-buffering: two in-flight units per worker (2 task
    # threads x nufft:2). One unit's CPU fold overlaps the sibling's GPU
    # transform; the wrapper's VRAM ledger queues the transforms themselves.
    env.setdefault("MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER", "1")
    # Concurrent cases share the box: cap the per-case extraction fork
    # pool so a CPU tail cannot starve a co-resident GPU case of RAM
    # (measured: 96-way extraction + prewarm OOM-killed the CPU case).
    env.setdefault("MOSAIC_DECODE_PARALLEL", str(max(4, (os.cpu_count() or 8) // 3)))
    env.setdefault("MOSAIC_NUFFT_SLOTS_PER_WORKER", "2")
    env.setdefault("MOSAIC_DASK_GPU_RESOURCE", "2")
    env.setdefault("MOSAIC_SCATTERING_STAGE2_STREAMING", "1")
    env.setdefault("MOSAIC_RESIDUAL_PREFETCH_FACTOR", "8")
    env.setdefault("MOSAIC_RESIDUAL_LATTICE_RAM_FRACTION", "0.2")
    env.setdefault("MOSAIC_RESIDUAL_SHARD_GRID_BUDGET_BYTES", str(4 << 30))
    env.setdefault("MOSAIC_RESIDUAL_CHECKPOINT_CADENCE_BATCHES", "8")
    env.setdefault("MOSAIC_SCATTERING_INTERVAL_PAYLOAD_CACHE_MAX_BYTES", str(256 << 20))
    env.setdefault("MOSAIC_STREAMING_PAYLOAD_MEMO_MAX_BYTES", str(256 << 20))
    env.setdefault("MOSAIC_RESIDUAL_RIFFT_PAYLOAD_CACHE_MAX_BYTES", str(2 << 30))
    env.setdefault("MALLOC_TRIM_THRESHOLD_", str(64 << 20))
    env.setdefault("MALLOC_ARENA_MAX", "2")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def launch_case(case: str, log_dir: Path, env: dict[str, str]) -> tuple[subprocess.Popen, Path]:
    log_path = log_dir / f"{case}.log"
    handle = open(log_path, "w")
    proc = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "core.main",
            str(
                ROOT
                / "examples"
                / "config_3D"
                / "displacement"
                / f"run_parameters_hkl40_{case}.json"
            ),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_path


def wait_for_residual_done(proc: subprocess.Popen, log_path: Path) -> None:
    """Block until the case's GPU segment ends (marker line) or it exits."""
    while proc.poll() is None:
        try:
            if RESIDUAL_DONE_MARKER in log_path.read_text(errors="ignore"):
                return
        except OSError:
            pass
        time.sleep(3)


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return float("inf")


def wait_for_ram(min_gb: float, context: str) -> None:
    """Case-level RAM admission: do not start a case into a box that cannot
    hold it alongside the residents (the orchestrator-level analogue of the
    workers' budget discipline)."""
    waited = 0
    while mem_available_gb() < min_gb:
        if waited == 0:
            print(
                f"[pipeline] waiting for {min_gb:.0f} GB free before {context} "
                f"(now {mem_available_gb():.0f} GB)",
                flush=True,
            )
        time.sleep(10)
        waited += 10


def main() -> int:
    log_dir = Path(
        os.getenv("MOSAIC_PIPELINE_LOG_DIR", str(ROOT / ".dask-local" / "pipeline-logs"))
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    env = build_env()
    procs: dict[str, subprocess.Popen] = {}
    results: dict[str, int] = {}

    def reap(case: str, proc: subprocess.Popen) -> None:
        results[case] = proc.wait()
        print(
            f"[pipeline] case={case} exited rc={results[case]}",
            flush=True,
        )

    reapers = []
    print(f"[pipeline] logs -> {log_dir}", flush=True)

    def start(case: str):
        proc, log_path = launch_case(case, log_dir, env)
        procs[case] = proc
        print(f"[pipeline] started case={case} pid={proc.pid}", flush=True)
        reaper = threading.Thread(target=reap, args=(case, proc), daemon=True)
        reaper.start()
        reapers.append(reaper)
        return proc, log_path

    # Cases whose residual phase is already durable have no GPU segment: run
    # them ungated, fully overlapped with the gated chain's GPU work.
    ungated = [case for case in CASES if os.getenv("MOSAIC_PIPELINE_UNGATED", "all") and case in os.getenv("MOSAIC_PIPELINE_UNGATED", "all").split(",")]
    gated = [case for case in CASES if case not in ungated]
    default_floor = max(8.0, 0.3 * (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1073741824))
    min_free_gb = float(os.getenv("MOSAIC_PIPELINE_MIN_FREE_GB", str(round(default_floor))))
    for case in ungated:
        wait_for_ram(min_free_gb, f"ungated case {case}")
        start(case)
    for index, case in enumerate(gated):
        wait_for_ram(min_free_gb, f"case {case}")
        proc, log_path = start(case)
        if index < len(gated) - 1:
            wait_for_residual_done(proc, log_path)
            print(
                f"[pipeline] case={case} GPU segment done -> releasing next case "
                "(its CPU tail continues in parallel)",
                flush=True,
            )
    for reaper in reapers:
        reaper.join()
    failed = sorted(case for case, rc in results.items() if rc != 0)
    if failed:
        # Overlap casualties (OOM etc.) rerun SERIALLY with the box to
        # themselves; all their durable state resumes, so retries are cheap.
        print(f"[pipeline] retrying failed cases serially: {failed}", flush=True)
        for case in failed:
            wait_for_ram(min_free_gb, f"retry of {case}")
            proc, _log = start(case)
            results[case] = proc.wait()
            print(f"[pipeline] retry case={case} rc={results[case]}", flush=True)
    still_failed = {case: rc for case, rc in results.items() if rc != 0}
    if still_failed:
        print(f"[pipeline] FAILED cases: {still_failed}", flush=True)
        return 1
    print("[pipeline] all cases complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
