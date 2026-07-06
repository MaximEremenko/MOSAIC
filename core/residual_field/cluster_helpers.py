"""Dask cluster / worker / memory helper utilities for the residual-field stage.

Leaf utilities that query and act on Dask scheduler/worker state.
None of these import from core.residual_field.execution.
"""

from __future__ import annotations

import logging

from core.runtime import is_sync_client
from core.residual_field.backend import is_same_node_local_client
from core.residual_field.runtime_policy import _memory_backpressure_threshold
from core.residual_field.tasks import clear_residual_rifft_payload_cache

__all__ = [
    "_cap_async_max_inflight",
    "_clear_worker_rifft_payload_caches",
    "_cluster_host_memory_pressure",
    "_current_worker_addresses",
    "_resolve_owner_address",
    "_same_node_local_nufft_capacity",
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


def _same_node_local_nufft_capacity(client) -> tuple[int, int] | None:
    if client is None or is_sync_client(client):
        return None
    if not is_same_node_local_client(client):
        return None
    return _scheduler_nufft_capacity(client)


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


def _cluster_host_memory_pressure(
    client,
    *,
    threshold: float | None = None,
) -> bool:
    if client is None or is_sync_client(client):
        return False
    threshold = _memory_backpressure_threshold() if threshold is None else float(threshold)
    if threshold <= 0.0:
        return False
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        return False
    for worker in workers.values():
        try:
            memory_limit = int(worker.get("memory_limit") or 0)
            if memory_limit <= 0:
                continue
            metrics = worker.get("metrics", {}) or {}
            rss = int(metrics.get("memory") or worker.get("memory") or 0)
            if rss > 0 and float(rss) >= (float(memory_limit) * threshold):
                return True
        except Exception:
            continue
    return False


def _trim_workers_for_memory_pressure(client) -> None:
    if client is None or is_sync_client(client):
        return
    try:
        from core.runtime.worker_hooks import trim_worker_memory

        client.run(trim_worker_memory)
    except Exception:
        pass


def _clear_worker_rifft_payload_caches(client) -> None:
    try:
        clear_residual_rifft_payload_cache()
    except Exception:
        pass
    # Streaming (fused stage-1) mode additionally leaves per-process
    # scattering payload memos behind; they are stage-scoped scratch and must
    # not outlive the residual stage.
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
    try:
        run(clear_residual_rifft_payload_cache)
    except Exception:
        pass
    try:
        from core.scattering.streaming import clear_streaming_payload_memo

        run(clear_streaming_payload_memo)
    except Exception:
        pass


def _current_worker_addresses(client) -> list[str]:
    if client is None or is_sync_client(client):
        return []
    try:
        workers = client.scheduler_info().get("workers", {})
    except Exception:
        workers = {}
    return sorted(workers)


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
