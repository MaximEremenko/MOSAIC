from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Mapping

from core.storage.attempt_store import run_root
from core.storage.digests import digest_dict
from core.storage.manifest import read_manifest, write_manifest


PERFORMANCE_METRICS_SCHEMA = "mosaic.performance_metrics"
PERFORMANCE_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PerformanceMetricsManifest:
    run_digest: str
    generated_at_utc: str
    run_file_count: int
    stage_file_counts: dict[str, int]
    chunk_file_counts: dict[str, int]
    attempt_leaf_fanout: tuple[dict[str, Any], ...]
    max_attempt_leaf_entries: int
    commit_scan_seconds: dict[str, float]
    recovery_scan_seconds: float | None
    performance_metrics_digest: str

    schema: ClassVar[str] = PERFORMANCE_METRICS_SCHEMA
    schema_version: ClassVar[int] = PERFORMANCE_METRICS_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "run_digest": self.run_digest,
            "generated_at_utc": self.generated_at_utc,
            "run_file_count": int(self.run_file_count),
            "stage_file_counts": dict(self.stage_file_counts),
            "chunk_file_counts": dict(self.chunk_file_counts),
            "attempt_leaf_fanout": [dict(item) for item in self.attempt_leaf_fanout],
            "max_attempt_leaf_entries": int(self.max_attempt_leaf_entries),
            "commit_scan_seconds": {
                str(key): float(value) for key, value in self.commit_scan_seconds.items()
            },
            "recovery_scan_seconds": (
                None if self.recovery_scan_seconds is None else float(self.recovery_scan_seconds)
            ),
            "performance_metrics_digest": self.performance_metrics_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PerformanceMetricsManifest":
        return cls(
            run_digest=str(payload["run_digest"]),
            generated_at_utc=str(payload["generated_at_utc"]),
            run_file_count=int(payload["run_file_count"]),
            stage_file_counts={
                str(key): int(value)
                for key, value in dict(payload["stage_file_counts"]).items()
            },
            chunk_file_counts={
                str(key): int(value)
                for key, value in dict(payload["chunk_file_counts"]).items()
            },
            attempt_leaf_fanout=tuple(
                {
                    "stage": str(item["stage"]),
                    "chunk_id": int(item["chunk_id"]),
                    "shard": str(item["shard"]),
                    "work_unit_digest": str(item["work_unit_digest"]),
                    "attempt_count": int(item["attempt_count"]),
                }
                for item in payload["attempt_leaf_fanout"]
            ),
            max_attempt_leaf_entries=int(payload["max_attempt_leaf_entries"]),
            commit_scan_seconds={
                str(key): float(value)
                for key, value in dict(payload["commit_scan_seconds"]).items()
            },
            recovery_scan_seconds=(
                None
                if payload.get("recovery_scan_seconds") is None
                else float(payload["recovery_scan_seconds"])
            ),
            performance_metrics_digest=str(payload["performance_metrics_digest"]),
        )


def performance_metrics_path(output_dir: str | Path, run_digest: str) -> Path:
    return run_root(output_dir, run_digest) / "performance_metrics.json"


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _stage_file_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage in ("scattering", "residual_field", "decoding"):
        stage_path = root / stage
        if stage_path.exists():
            counts[stage] = _file_count(stage_path)
    return counts


def _chunk_file_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        chunks_root = stage_dir / "chunks"
        if not chunks_root.exists():
            continue
        for chunk_dir in sorted(path for path in chunks_root.iterdir() if path.is_dir()):
            if not chunk_dir.name.startswith("chunk_"):
                continue
            counts[f"{stage_dir.name}/{chunk_dir.name}"] = _file_count(chunk_dir)
    return counts


def _attempt_leaf_fanout(root: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        chunks_root = stage_dir / "chunks"
        if not chunks_root.exists():
            continue
        for chunk_dir in sorted(path for path in chunks_root.iterdir() if path.is_dir()):
            attempts_root = chunk_dir / "attempts"
            if not attempts_root.exists():
                continue
            try:
                chunk_id = int(chunk_dir.name.removeprefix("chunk_"))
            except ValueError:
                continue
            for shard_dir in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
                for work_unit_dir in sorted(path for path in shard_dir.iterdir() if path.is_dir()):
                    attempt_count = sum(
                        1
                        for attempt_dir in work_unit_dir.iterdir()
                        if attempt_dir.is_dir() and attempt_dir.name.startswith("attempt_")
                    )
                    records.append(
                        {
                            "stage": stage_dir.name,
                            "chunk_id": int(chunk_id),
                            "shard": shard_dir.name,
                            "work_unit_digest": work_unit_dir.name,
                            "attempt_count": int(attempt_count),
                        }
                    )
    return tuple(records)


def _digest_for_payload(payload: Mapping[str, Any]) -> str:
    digest_input = {
        key: value
        for key, value in payload.items()
        if key != "performance_metrics_digest"
    }
    return digest_dict(digest_input, domain="mosaic.performance_metrics.v1")


def collect_run_performance_metrics(
    *,
    output_dir: str | Path,
    run_digest: str,
    commit_scan_seconds: Mapping[str, float] | None = None,
    recovery_scan_seconds: float | None = None,
) -> PerformanceMetricsManifest:
    root = run_root(output_dir, run_digest)
    stage_counts = _stage_file_counts(root) if root.exists() else {}
    chunk_counts = _chunk_file_counts(root) if root.exists() else {}
    fanout = _attempt_leaf_fanout(root) if root.exists() else ()
    max_attempt_leaf_entries = max(
        (int(item["attempt_count"]) for item in fanout),
        default=0,
    )
    payload = {
        "schema": PERFORMANCE_METRICS_SCHEMA,
        "schema_version": PERFORMANCE_METRICS_SCHEMA_VERSION,
        "run_digest": str(run_digest),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_file_count": _file_count(root),
        "stage_file_counts": stage_counts,
        "chunk_file_counts": chunk_counts,
        "attempt_leaf_fanout": [dict(item) for item in fanout],
        "max_attempt_leaf_entries": int(max_attempt_leaf_entries),
        "commit_scan_seconds": {
            str(key): float(value) for key, value in dict(commit_scan_seconds or {}).items()
        },
        "recovery_scan_seconds": (
            None if recovery_scan_seconds is None else float(recovery_scan_seconds)
        ),
    }
    payload["performance_metrics_digest"] = _digest_for_payload(payload)
    return PerformanceMetricsManifest.from_payload(payload)


def write_performance_metrics(
    *,
    output_dir: str | Path,
    run_digest: str,
    commit_scan_seconds: Mapping[str, float] | None = None,
    recovery_scan_seconds: float | None = None,
) -> PerformanceMetricsManifest:
    target = performance_metrics_path(output_dir, run_digest)
    merged_commit_seconds: dict[str, float] = {}
    merged_recovery_seconds = recovery_scan_seconds
    if target.exists():
        existing = read_manifest(
            target,
            codec=PerformanceMetricsManifest,
            output_dir=output_dir,
        )
        merged_commit_seconds.update(existing.commit_scan_seconds)
        if merged_recovery_seconds is None:
            merged_recovery_seconds = existing.recovery_scan_seconds
    merged_commit_seconds.update(
        {str(key): float(value) for key, value in dict(commit_scan_seconds or {}).items()}
    )
    metrics = collect_run_performance_metrics(
        output_dir=output_dir,
        run_digest=run_digest,
        commit_scan_seconds=merged_commit_seconds,
        recovery_scan_seconds=merged_recovery_seconds,
    )
    write_manifest(target, metrics, output_dir=output_dir)
    return metrics


def load_performance_budget(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Performance budget file must contain a JSON object.")
    return {str(key): float(value) for key, value in raw.items()}


def evaluate_performance_budget(
    metrics: PerformanceMetricsManifest | Mapping[str, Any],
    budget: Mapping[str, float],
) -> list[str]:
    payload = metrics.to_payload() if hasattr(metrics, "to_payload") else dict(metrics)
    violations: list[str] = []

    max_attempt_leaf_entries = budget.get("max_attempt_leaf_entries")
    if max_attempt_leaf_entries is not None and int(payload["max_attempt_leaf_entries"]) > float(max_attempt_leaf_entries):
        violations.append(
            "max_attempt_leaf_entries exceeded: "
            f"{payload['max_attempt_leaf_entries']} > {max_attempt_leaf_entries}"
        )

    max_commit_scan = budget.get("max_commit_scan_seconds_per_chunk")
    if max_commit_scan is not None:
        commit_values = [float(value) for value in dict(payload["commit_scan_seconds"]).values()]
        observed = max(commit_values, default=0.0)
        if observed > float(max_commit_scan):
            violations.append(
                "max_commit_scan_seconds_per_chunk exceeded: "
                f"{observed:.6f} > {float(max_commit_scan):.6f}"
            )

    max_recovery_scan = budget.get("max_recovery_scan_seconds")
    recovery_scan_seconds = payload.get("recovery_scan_seconds")
    if (
        max_recovery_scan is not None
        and recovery_scan_seconds is not None
        and float(recovery_scan_seconds) > float(max_recovery_scan)
    ):
        violations.append(
            "max_recovery_scan_seconds exceeded: "
            f"{float(recovery_scan_seconds):.6f} > {float(max_recovery_scan):.6f}"
        )

    max_run_file_count = budget.get("max_run_file_count")
    if max_run_file_count is not None and int(payload["run_file_count"]) > float(max_run_file_count):
        violations.append(
            f"max_run_file_count exceeded: {payload['run_file_count']} > {max_run_file_count}"
        )

    return violations


__all__ = [
    "PERFORMANCE_METRICS_SCHEMA",
    "PerformanceMetricsManifest",
    "collect_run_performance_metrics",
    "evaluate_performance_budget",
    "load_performance_budget",
    "performance_metrics_path",
    "write_performance_metrics",
]
