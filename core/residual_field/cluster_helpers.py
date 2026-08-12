"""Dask cluster / worker / memory helper utilities for the residual-field stage.

Leaf utilities that query and act on Dask scheduler/worker state.
None of these import from core.residual_field.execution.
"""

from __future__ import annotations

import logging

from core.runtime import is_sync_client
# Private alias kept for the residual-field call sites; the guarded helper
# (a sync client must report no workers) now lives in core.runtime.
from core.runtime import current_worker_addresses as _current_worker_addresses
from core.residual_field.runtime_policy import _memory_backpressure_threshold
from core.residual_field.tasks import clear_residual_rifft_payload_cache

__all__ = [
    "_cap_async_max_inflight",
    "_clear_worker_rifft_payload_caches",
    "_cluster_host_memory_pressure",
    "_current_worker_addresses",
    "_resolve_owner_address",
    "_scheduler_nufft_capacity",
    "_trim_workers_for_memory_pressure",
]

logger = logging.getLogger(__name__)


def _scheduler_nufft_capacity(client) -> tuple[int, int] | None:
    if client is None or is_sync_client(client):
        return None
    try:
        scheduler_info = client.scheduler_info()
        workers = scheduler_info.get("workers", {})
    except Exception:
        workers = {}
    if not workers:
        return (1, 0)
    nufft_slots = sum(
        int(worker.get("resources", {}).get("nufft", 0))
        for worker in workers.values()
    )
    if int(nufft_slots) <= 0:
        raise RuntimeError(
            "Dask scheduler reports zero total 'nufft' resource capacity; "
            "residual-field NUFFT tasks would stay queued or overbook. "
            "Configure worker resources with nufft=N."
        )
    return max(1, int(nufft_slots)), len(workers)


def _cap_async_max_inflight(
    *,
    client,
    requested: int,
    prefetch_factor: int = 2,
) -> int:
    requested = max(1, int(requested))
    capacity_info = _scheduler_nufft_capacity(client)
    if capacity_info is None:
        return requested
    nufft_slots, worker_count = capacity_info
    # prefetch_factor arrives fully resolved (env/config precedence lives in
    # runtime_policy._residual_nufft_prefetch_factor — one knob, one place).
    factor = max(1, min(8, int(prefetch_factor)))
    capacity = max(1, int(nufft_slots) * int(factor))
    capped = min(requested, capacity)
    logger.info(
        "Residual-field local queue cap | requested=%d | effective=%d | workers=%d | nufft_slots=%d | prefetch_factor=%d",
        requested,
        capped,
        int(worker_count),
        int(nufft_slots),
        int(factor),
    )
    return capped


def _worker_host_available_memory_fraction() -> float:
    """Runs ON a worker: MemAvailable/MemTotal for THIS host, cgroup-aware."""
    from core.runtime.cpu_resources import available_memory_bytes, total_memory_bytes

    total = total_memory_bytes()
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, available_memory_bytes() / float(total)))


def _cluster_host_memory_pressure(
    client,
    *,
    threshold: float | None = None,
) -> bool:
    """True when any worker HOST is low on memory.

    Probes host truth (cgroup-clamped MemAvailable/MemTotal via client.run)
    rather than Dask's per-worker memory_limit: the architecture
    deliberately sets memory_limit=0 on every default backend (memmap pages
    count into RSS and the nanny killed healthy workers), which made the
    old scheduler-metrics probe skip every worker — this valve was a
    permanent no-op and MOSAIC_RESIDUAL_MEMORY_BACKPRESSURE_PCT a dead
    knob. Pressure = host available fraction below (1 - threshold), the
    host-side reading of "RSS at threshold percent"."""
    if client is None or is_sync_client(client):
        return False
    threshold = _memory_backpressure_threshold() if threshold is None else float(threshold)
    if threshold <= 0.0:
        return False
    run = getattr(client, "run", None)
    if not callable(run):
        return False
    try:
        readings = run(_worker_host_available_memory_fraction)
    except Exception:
        return False
    available_floor = max(0.0, 1.0 - threshold)
    return any(
        float(fraction) < available_floor
        for fraction in (readings or {}).values()
    )


def _trim_workers_for_memory_pressure(client) -> None:
    if client is None or is_sync_client(client):
        return
    try:
        from core.runtime.worker_hooks import trim_worker_memory

        client.run(trim_worker_memory)
    except Exception:
        pass


def _clear_worker_rifft_payload_caches(client) -> None:
    # The lattice grid cache is the largest per-process RAM consumer and had
    # no production release path at all — grids pinned RAM past the stage
    # end, starving every later admission. All three caches are stage-scoped
    # scratch and must not outlive the residual stage; the streaming memo
    # additionally holds stage-1 payloads.
    from core.residual_field.tasks import clear_residual_lattice_cache

    for clear in (clear_residual_rifft_payload_cache, clear_residual_lattice_cache):
        try:
            clear()
        except Exception:
            pass
    try:
        from core.scattering.streaming import clear_streaming_payload_memo

        clear_streaming_payload_memo()
    except Exception:
        pass
    if client is None or is_sync_client(client):
        return
    run = getattr(client, "run", None)
    if not callable(run):
        return
    for remote_clear in (
        clear_residual_rifft_payload_cache,
        clear_residual_lattice_cache,
    ):
        try:
            run(remote_clear)
        except Exception:
            pass
    try:
        from core.scattering.streaming import clear_streaming_payload_memo

        run(clear_streaming_payload_memo)
    except Exception:
        pass


def _resolve_owner_address(
    *,
    target_key: tuple[int, int | None],
    target_owners: dict[tuple[int, int | None], str],
    worker_addresses: list[str],
) -> str | None:
    current_owner = target_owners.get(target_key)
    if current_owner in worker_addresses:
        return current_owner
    if not worker_addresses:
        return current_owner
    live_owner_loads = {address: 0 for address in worker_addresses}
    for other_target_key, owner_address in target_owners.items():
        if other_target_key == target_key:
            continue
        if owner_address in live_owner_loads:
            live_owner_loads[owner_address] += 1
    replacement_owner = min(
        worker_addresses,
        key=lambda address: (live_owner_loads[address], address),
    )
    target_owners[target_key] = replacement_owner
    if current_owner is not None and current_owner != replacement_owner:
        logger.warning(
            "Residual-field owner remap | target=%s | previous=%s | replacement=%s",
            target_key,
            current_owner,
            replacement_owner,
        )
    return replacement_owner
