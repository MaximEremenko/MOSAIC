"""Shared MOSAIC launcher for the example notebooks.

Every demo runs cases the same way: a fresh ``python -m core.main``
subprocess with the environment the current codebase expects —
CUDA wheel libraries on LD_LIBRARY_PATH (cufinufft links the venv's
libcudart/libcufft, not the system CUDA), scratch on real disk (worker
scratch must NEVER land on a tmpfs /tmp: "spill" to RAM feeds the OOM
killer), the streaming stage-2 architecture enabled, and no device
pinning — worker count auto-sizes to visible GPUs (one worker per GPU),
so the same notebook runs on a 1-GPU laptop and a multi-GPU node.

Subprocess (not in-process ``main()``) on purpose: repeated in-process
runs share dask client/event-loop state across cases, and
LD_LIBRARY_PATH cannot be set after the kernel has started.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_tmpfs(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
        best_match, best_type = "", ""
        with open("/proc/mounts") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fs_type = parts[1], parts[2]
                if resolved == mount_point or resolved.startswith(
                    mount_point.rstrip("/") + "/"
                ):
                    if len(mount_point) > len(best_match):
                        best_match, best_type = mount_point, fs_type
        return best_type in ("tmpfs", "ramfs")
    except OSError:
        return False


def mosaic_environment(repo_root: Path | None = None) -> dict[str, str]:
    root = Path(repo_root or _REPO_ROOT).resolve()
    env = os.environ.copy()

    site = Path(sysconfig.get_paths()["purelib"])
    cuda_libs = [
        str(site / "nvidia" / pkg / "lib")
        for pkg in ("cuda_runtime", "cufft", "nvjitlink", "cublas", "cuda_nvrtc")
        if (site / "nvidia" / pkg / "lib").is_dir()
    ]
    if cuda_libs:
        env["LD_LIBRARY_PATH"] = ":".join(cuda_libs) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )

    scratch = root / ".dask-local"
    env.setdefault("DASK_LOCAL_DIR", str(scratch / "examples-dask"))
    for name, sub in (
        ("MOSAIC_RESIDUAL_LATTICE_SCRATCH", "examples-lattice"),
        ("MOSAIC_WORKER_SCRATCH_ROOT", "examples-worker"),
        ("MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT", "examples-shard"),
    ):
        path = scratch / sub
        path.mkdir(parents=True, exist_ok=True)
        env.setdefault(name, str(path))
        if _is_tmpfs(Path(env[name])):
            raise RuntimeError(
                f"{name}={env[name]} is on tmpfs (RAM-backed). Point it at "
                "real disk before running the examples."
            )

    # Streaming computes intervals inside residual work units and writes NO
    # per-interval artifacts. Demos that analyze interval_*.hdf5 declare it
    # by setting MOSAIC_SAVE_SCATTERING_INTERVAL_ARTIFACTS=1 (their first
    # cell) and get the legacy artifact-writing path instead.
    if env.get("MOSAIC_SAVE_SCATTERING_INTERVAL_ARTIFACTS") != "1":
        env.setdefault("MOSAIC_SCATTERING_STAGE2_STREAMING", "1")
    env.setdefault("MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER", "1")
    env.setdefault("MOSAIC_NUFFT_SLOTS_PER_WORKER", "2")
    env.setdefault("MOSAIC_DASK_GPU_RESOURCE", "2")
    env.setdefault("MOSAIC_RESIDUAL_PREFETCH_FACTOR", "8")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("MALLOC_TRIM_THRESHOLD_", str(64 << 20))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _die_with_parent():
    # Linux: SIGTERM the child if the notebook kernel dies mid-run.
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def run_mosaic_case(run_file, *, repo_root: Path | None = None, tag: str | None = None) -> None:
    """Run one MOSAIC case (a run_parameters JSON) to completion."""
    root = Path(repo_root or _REPO_ROOT).resolve()
    run_path = Path(run_file).resolve()
    label = tag if tag is not None else run_path.stem
    print("=" * 72)
    print(f"Running MOSAIC -- {str(label).upper()}")
    print(f"  run file: {run_path}")
    print("=" * 72, flush=True)
    subprocess.run(
        [sys.executable, "-m", "core.main", str(run_path)],
        cwd=str(root),
        env=mosaic_environment(root),
        check=True,
        preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
    )
    print(f"[{label}] completed OK", flush=True)
