#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.dask_resources import scheduler_resource_capacity
from core.runtime.fs_capability import profile_output_filesystem
from core.runtime.gpu_admission import require_gpu_admission
from core.storage.performance import (
    evaluate_performance_budget,
    load_performance_budget,
    performance_metrics_path,
    write_performance_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MOSAIC HPC runtime smoke checks.")
    parser.add_argument("--scheduler", default=os.getenv("DASK_BACKEND", "local"))
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-digest", default="hpc_smoke_validate")
    parser.add_argument("--chaos", action="store_true")
    parser.add_argument("--gpu-policy", default=os.getenv("MOSAIC_NUFFT_EXECUTION_POLICY", "auto"))
    parser.add_argument("--perf-budget-file", default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("DASK_BACKEND", args.scheduler)
    os.environ.setdefault("DASK_MAX_WORKERS", str(max(1, int(args.nodes))))

    from core.runtime.dask_client import get_client

    client = get_client()
    fs_manifest = profile_output_filesystem(
        Path(args.output_dir),
        run_digest=args.run_digest,
        client=client,
        require_cross_host=int(args.nodes) > 1,
    )
    nufft_capacity = scheduler_resource_capacity(client, "nufft")
    gpu_report = None
    if str(args.gpu_policy).strip().lower().replace("_", "-") in {
        "gpu-required",
        "allow-fallback",
    }:
        gpu_report = require_gpu_admission(
            client,
            policy=args.gpu_policy,
            required_gpu_tasks=1,
        )
    metrics = write_performance_metrics(
        output_dir=args.output_dir,
        run_digest=args.run_digest,
    )
    budget_violations: list[str] = []
    if args.perf_budget_file is not None:
        budget = load_performance_budget(args.perf_budget_file)
        budget_violations = evaluate_performance_budget(metrics, budget)
    ok = not budget_violations
    print(
        json.dumps(
            {
                "ok": ok,
                "scheduler": args.scheduler,
                "nodes": int(args.nodes),
                "chaos_requested": bool(args.chaos),
                "fs_capability_digest": fs_manifest.capability_digest,
                "nufft_slots": int(nufft_capacity.effective_runnable_slots),
                "gpu_policy": args.gpu_policy,
                "gpu_admitted": gpu_report is not None,
                "performance_metrics_path": performance_metrics_path(
                    args.output_dir,
                    args.run_digest,
                ).as_posix(),
                "performance_budget_violations": budget_violations,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
