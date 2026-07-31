"""Allocation-aware CPU/RAM detection (SLURM, cgroups, containers)."""
import os

import pytest

from core.runtime import cpu_resources


def test_unrestricted_matches_host(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert 1 <= cpu_resources.available_cpu_count() <= (os.cpu_count() or 1)


def test_slurm_grant_wins_when_smaller(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert cpu_resources.available_cpu_count() == 3


def test_slurm_garbage_ignored(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    assert cpu_resources.available_cpu_count() >= 1


def test_affinity_respected():
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pytest.skip("no sched_getaffinity")
    assert cpu_resources.available_cpu_count() <= allowed


def test_slurm_mem_per_node(monkeypatch):
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")  # MB
    assert cpu_resources.total_memory_bytes() == 4096 << 20


def test_slurm_mem_per_cpu(monkeypatch):
    monkeypatch.delenv("SLURM_MEM_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert cpu_resources.total_memory_bytes() == 2 * (1024 << 20)


def test_total_memory_positive_unrestricted(monkeypatch):
    for env in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(env, raising=False)
    assert cpu_resources.total_memory_bytes() > 0


def test_available_memory_bounded_by_total():
    available = cpu_resources.available_memory_bytes()
    if available is None:
        pytest.skip("no /proc/meminfo")
    assert 0 < available <= cpu_resources.total_memory_bytes()
