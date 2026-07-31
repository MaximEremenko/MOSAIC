"""CPU-count detection that survives HPC allocations and containers.

``os.cpu_count()`` reports every core on the physical host. Inside a SLURM
cpuset, a cgroup-limited container, or any shared node, that number is a
lie — sizing a fork pool or a BLAS pool from it oversubscribes the
allocation (96 threads pinned onto 6 granted cores) and can OOM the
job's memory budget. This module answers "how many CPUs may THIS process
actually use" by taking the minimum of every signal that is present:

- scheduler grants (``SLURM_CPUS_PER_TASK`` / ``SLURM_CPUS_ON_NODE``),
- the process CPU affinity mask (cpusets — how SLURM and Docker
  ``cpuset`` limits manifest),
- the cgroup-v2/v1 CPU quota (how ``docker --cpus`` / Kubernetes
  requests manifest, invisible to the affinity mask),
- ``os.cpu_count()`` as the ceiling.
"""
from __future__ import annotations

import math
import os


def _slurm_cpus() -> int | None:
    for env in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.getenv(env)
        if raw:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def _affinity_cpus() -> int | None:
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        return None


def _cgroup_quota_cpus() -> int | None:
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="ascii") as handle:
            quota_raw, period_raw = handle.read().split()
        if quota_raw != "max":
            quota, period = int(quota_raw), int(period_raw)
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    try:
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="ascii"
        ) as handle:
            quota = int(handle.read())
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="ascii"
        ) as handle:
            period = int(handle.read())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def available_cpu_count() -> int:
    signals = [
        value
        for value in (
            _slurm_cpus(),
            _affinity_cpus(),
            _cgroup_quota_cpus(),
            os.cpu_count(),
        )
        if value is not None and value > 0
    ]
    return min(signals) if signals else 1


# --- memory ---------------------------------------------------------------
#
# /proc/meminfo describes the PHYSICAL HOST even inside a memory-limited
# cgroup (Docker, Kubernetes, SLURM's cgroup plugin): a job granted 6 GB on
# a 512 GB node reads MemTotal=512G / MemAvailable=400G and budgets itself
# straight into an OOM kill. The cgroup interface is the allocation's truth.

_CGROUP_NONE = (1 << 60)  # v1 "unlimited" sentinel is PAGE-rounded 2**63-ish


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, "r", encoding="ascii") as handle:
            raw = handle.read().strip()
        if raw == "max":
            return None
        value = int(raw)
    except (OSError, ValueError):
        return None
    return value if 0 < value < _CGROUP_NONE else None


def _slurm_memory_bytes() -> int | None:
    raw = os.getenv("SLURM_MEM_PER_NODE")
    if raw:
        try:
            value = int(str(raw).strip())
            if value > 0:
                return value << 20  # SLURM reports MB
        except (TypeError, ValueError):
            pass
    raw = os.getenv("SLURM_MEM_PER_CPU")
    if raw:
        try:
            value = int(str(raw).strip())
            if value > 0:
                return (value << 20) * available_cpu_count()
        except (TypeError, ValueError):
            pass
    return None


def _cgroup_memory_limit_bytes() -> int | None:
    limit = _read_int_file("/sys/fs/cgroup/memory.max")  # v2
    if limit is None:
        limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return limit


def _cgroup_memory_available_bytes() -> int | None:
    limit = _cgroup_memory_limit_bytes()
    if limit is None:
        return None
    current = _read_int_file("/sys/fs/cgroup/memory.current")
    if current is None:
        current = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if current is None:
        return None
    # Page-cache pages inside the cgroup are reclaimable, same as the
    # kernel's MemAvailable heuristic; count the inactive file pages back.
    reclaimable = 0
    for stat_path in ("/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat"):
        try:
            with open(stat_path, "r", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("inactive_file "):
                        reclaimable = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
        break
    return max(0, limit - current + reclaimable)


def total_memory_bytes() -> int:
    try:
        host_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        host_total = 0
    signals = [
        value
        for value in (
            _slurm_memory_bytes(),
            _cgroup_memory_limit_bytes(),
            host_total if host_total > 0 else None,
        )
        if value is not None and value > 0
    ]
    return min(signals) if signals else 0


def available_memory_bytes() -> int | None:
    """MemAvailable clamped by the cgroup allocation's own headroom."""
    host_available: int | None = None
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    host_available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        host_available = None
    cgroup_available = _cgroup_memory_available_bytes()
    signals = [v for v in (host_available, cgroup_available) if v is not None]
    return min(signals) if signals else None
