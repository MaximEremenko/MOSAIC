from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
import time


@dataclass(frozen=True)
class SchedulerResourceCapacity:
    resource_name: str
    worker_count: int
    eligible_worker_addresses: tuple[str, ...]
    effective_runnable_slots: int
    total_declared_capacity: float
    required_per_task: float = 1.0

    @property
    def has_runnable_slots(self) -> bool:
        return self.effective_runnable_slots > 0


def _scheduler_workers(client) -> Mapping[object, object]:
    try:
        scheduler_info = client.scheduler_info()
    except Exception:
        return {}
    workers = scheduler_info.get("workers", {})
    return workers if isinstance(workers, Mapping) else {}


def _worker_resource_capacity(worker: object, resource_name: str) -> float:
    if not isinstance(worker, Mapping):
        return 0.0
    resources = worker.get("resources", {}) or {}
    if not isinstance(resources, Mapping):
        return 0.0
    try:
        return float(resources.get(resource_name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _worker_runnable_slots(
    worker: object,
    resource_name: str,
    *,
    required_per_task: float,
) -> int:
    capacity = _worker_resource_capacity(worker, resource_name)
    if capacity <= 0.0 or required_per_task <= 0.0:
        return 0
    return max(0, int(math.floor(capacity / required_per_task)))


def _resource_readiness_deferred() -> bool:
    return os.getenv("MOSAIC_DASK_RESOURCE_READINESS", "").strip().lower() == "deferred"


def scheduler_resource_capacity(
    client,
    resource_name: str,
    *,
    required_per_task: float = 1.0,
) -> SchedulerResourceCapacity:
    workers = _scheduler_workers(client)
    worker_count = len(workers)
    slots_by_address = {
        str(address): _worker_runnable_slots(
            worker,
            resource_name,
            required_per_task=float(required_per_task),
        )
        for address, worker in workers.items()
    }
    eligible_addresses = tuple(
        sorted(address for address, slots in slots_by_address.items() if int(slots) > 0)
    )
    return SchedulerResourceCapacity(
        resource_name=str(resource_name),
        worker_count=worker_count,
        eligible_worker_addresses=eligible_addresses,
        effective_runnable_slots=sum(int(slots) for slots in slots_by_address.values()),
        total_declared_capacity=sum(
            _worker_resource_capacity(worker, resource_name)
            for worker in workers.values()
        ),
        required_per_task=float(required_per_task),
    )


def _capacity_error_message(
    capacity: SchedulerResourceCapacity,
    *,
    context: str,
    minimum_slots: int,
) -> str:
    if capacity.worker_count == 0:
        return (
            f"Dask scheduler reports zero workers for {context}; "
            f"cannot prove {capacity.resource_name!r} resource capacity. "
            "NUFFT tasks would stay queued. Wait for workers to register, or set "
            "MOSAIC_DASK_RESOURCE_READINESS=deferred only when a scheduler is "
            "expected to receive workers after task submission."
        )
    return (
        f"Dask scheduler reports no runnable {capacity.resource_name!r} resource "
        f"slots for {context}; observed worker count={capacity.worker_count}, "
        f"total declared capacity={capacity.total_declared_capacity:g}, "
        f"required per task={capacity.required_per_task:g}, "
        f"minimum slots={int(minimum_slots)}. "
        "Fractional worker resources cannot be aggregated across workers for "
        "resources={'nufft': 1}. Create Dask workers with "
        f"resources={{{capacity.resource_name!r}: N}} or use MOSAIC's Dask client factory."
    )


def require_scheduler_resource_capacity(
    client,
    resource_name: str,
    *,
    context: str,
    minimum_slots: int = 1,
    required_per_task: float = 1.0,
) -> SchedulerResourceCapacity:
    capacity = scheduler_resource_capacity(
        client,
        resource_name,
        required_per_task=required_per_task,
    )
    if capacity.worker_count == 0 and _resource_readiness_deferred():
        return wait_for_scheduler_resource_capacity(
            client,
            resource_name,
            context=context,
            minimum_slots=minimum_slots,
            required_per_task=required_per_task,
        )
    if capacity.worker_count == 0 or capacity.effective_runnable_slots < int(minimum_slots):
        raise RuntimeError(
            _capacity_error_message(
                capacity,
                context=context,
                minimum_slots=int(minimum_slots),
            )
        )
    return capacity


def _resource_wait_timeout_seconds() -> float:
    raw = os.getenv("MOSAIC_DASK_RESOURCE_WAIT_TIMEOUT")
    if raw is None or str(raw).strip() == "":
        return 120.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 120.0


def wait_for_scheduler_resource_capacity(
    client,
    resource_name: str,
    *,
    context: str,
    minimum_slots: int = 1,
    required_per_task: float = 1.0,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 1.0,
) -> SchedulerResourceCapacity:
    timeout = (
        _resource_wait_timeout_seconds()
        if timeout_seconds is None
        else max(0.0, float(timeout_seconds))
    )
    poll_interval = max(0.01, float(poll_interval_seconds))
    deadline = time.monotonic() + timeout
    last_capacity = scheduler_resource_capacity(
        client,
        resource_name,
        required_per_task=required_per_task,
    )
    while (
        last_capacity.worker_count == 0
        or last_capacity.effective_runnable_slots < int(minimum_slots)
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                _capacity_error_message(
                    last_capacity,
                    context=f"{context} after waiting {timeout:g}s",
                    minimum_slots=int(minimum_slots),
                )
            )
        time.sleep(min(poll_interval, max(0.01, deadline - time.monotonic())))
        last_capacity = scheduler_resource_capacity(
            client,
            resource_name,
            required_per_task=required_per_task,
        )
    return last_capacity


__all__ = [
    "SchedulerResourceCapacity",
    "require_scheduler_resource_capacity",
    "scheduler_resource_capacity",
    "wait_for_scheduler_resource_capacity",
]
