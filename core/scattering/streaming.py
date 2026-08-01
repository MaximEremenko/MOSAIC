"""Streaming (fused stage-1 -> stage-2) scattering support.

In streaming mode the scattering stage computes NO interval payloads and
writes NO durable interval store. Instead it publishes a
:class:`StreamingComputeContext` — everything a residual work unit needs to
compute its interval batch's scattering payloads in-task — and the residual
stage's work units run stage-1 kernels themselves, fold the results into
worker-local subchunk accumulators, and discard the payloads. Amplitudes
exist once, in the RAM of the worker that computed them, for the lifetime of
one fold; resume recomputes only batches the checkpoint ledger has not made
durable.

Enable with ``runtime_info.scattering_stage2_mode = "streaming"`` or
``MOSAIC_SCATTERING_STAGE2_STREAMING=1``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from core.runtime.budgeted_cache import BudgetedLRU
from core.scattering.interval_payload import (
    PAYLOAD_MISS,
    read_interval_payload,
    write_interval_payload,
)
from core.scattering.kernels import IntervalTask, streaming_lattice_default
from core.scattering.tasks import compute_scattering_interval_payload

logger = logging.getLogger(__name__)


def stage2_streaming_enabled(parameters: Mapping[str, Any]) -> bool:
    runtime_info = parameters.get("runtime_info") or {}
    if hasattr(runtime_info, "to_mapping"):
        runtime_info = runtime_info.to_mapping()
    mode = runtime_info.get("scattering_stage2_mode")
    if mode is None:
        mode = runtime_info.get("stage2_mode")
    if mode is not None:
        return str(mode).strip().lower().replace("-", "_") == "streaming"
    enabled = runtime_info.get("scattering_stage2_streaming")
    if enabled is None:
        enabled = os.getenv("MOSAIC_SCATTERING_STAGE2_STREAMING")
    if isinstance(enabled, str):
        return enabled.strip().lower() in {"1", "true", "yes", "on", "streaming"}
    return bool(enabled)


@dataclass(frozen=True)
class StreamingComputeContext:
    """Inputs for in-task scattering interval computation (fused stage-1).

    One instance is built per run by the scattering stage and shipped to
    every residual work unit (dask-scattered once, broadcast). All members
    are picklable; the structure arrays are the same objects the legacy
    stage-1 tasks receive. ``cache_token`` scopes the per-process payload
    memo to this run's scattering identity."""

    cache_token: str
    interval_lookup: Dict[int, dict]
    B_: np.ndarray
    mask_params: dict
    MaskStrategy: Any
    supercell: np.ndarray
    original_coords: np.ndarray
    cells_origin: np.ndarray
    elements_arr: np.ndarray
    charge: float
    use_coeff: bool
    coeff_val: np.ndarray | None
    unique_elements: tuple
    ff_factory: Any
    # The SCATTERING stage's NUFFT execution settings. Payload computation
    # must use these (not the residual stage's), or streaming amplitudes
    # silently diverge from durable-mode amplitudes whenever the two stages
    # are configured differently (scattering_nufft_* vs residual_nufft_*).
    nufft_eps: float | None = None
    nufft_prefer_cpu: bool | None = None
    nufft_gpu_only: bool | None = None
    # Durable stage-1 payload store (directory path, already scoped to this
    # run's scattering identity). When set, lazy loaders read computed
    # payloads from disk before recomputing, and write fresh computes back —
    # so stage-1 runs ONCE per interval per store lifetime instead of once
    # per (shard, owner, restart). ``None`` keeps the pure in-RAM behaviour.
    payload_store_dir: str | None = None


def streaming_slot_map(
    batch_interval_ids, n_slots: int
) -> dict[tuple[int, ...], int]:
    """Deterministic subchunk slot per interval batch: rank in canonical
    batch order, mod n_slots.

    Replaces sha256(interval_ids) mod n_slots (retired with digest schema
    3): the hash gave binomially skewed — possibly empty — slots, which
    concentrated the rod case's scattering volume in ONE slot and forced
    the per-(chunk,slot) owner spread as a compensating layer. Rank order
    is exactly as stable as the hash was: the batch list is a
    config-deterministic, worker-count-invariant function of the plan, and
    resume rebuilds the identical batch universe per pending chunk
    (interval saves are all-or-nothing at chunk finalize) — while balancing
    batch counts across slots exactly, with no empty slots whenever
    batches >= slots."""
    ordered = sorted(
        {
            tuple(int(interval_id) for interval_id in interval_ids)
            for interval_ids in batch_interval_ids
        }
    )
    slots = max(1, int(n_slots))
    return {interval_ids: rank % slots for rank, interval_ids in enumerate(ordered)}


# Per-process memo of computed interval payloads: consecutive work units on
# the same worker (the same batch folded into different chunks) reuse one
# stage-1 computation instead of recomputing per chunk. Entries are
# (IntervalTask | None); None records a mask-empty interval so its emptiness
# is not re-derived either — _MEMO_MISS distinguishes a miss from that
# memoized None.
_MEMO_MISS = object()


def _stream_memo_max_bytes() -> int:
    raw = os.getenv("MOSAIC_STREAMING_PAYLOAD_MEMO_MAX_BYTES")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return 1 << 30


def _interval_task_nbytes(task: IntervalTask | None) -> int:
    if task is None:
        return 64
    total = 0
    for name in ("q_grid", "q_amp", "q_amp_av"):
        value = getattr(task, name, None)
        if value is not None and not isinstance(value, np.memmap):
            total += int(np.asarray(value).nbytes)
    return max(64, total)


# min_entries=1: the memo always keeps at least the most recent payload so
# back-to-back folds of the same batch never recompute it.
_STREAM_MEMO = BudgetedLRU(
    max_bytes=_stream_memo_max_bytes,
    size_fn=_interval_task_nbytes,
    min_entries=1,
)


def clear_streaming_payload_memo() -> None:
    _STREAM_MEMO.clear()


def _memo_get(key: tuple[str, int]):
    """Return the memoized payload (possibly None) or ``_MEMO_MISS``."""
    return _STREAM_MEMO.get(key, _MEMO_MISS)


def _memo_store(key: tuple[str, int], task: IntervalTask | None) -> None:
    if _stream_memo_max_bytes() <= 0:
        return
    _STREAM_MEMO.store(key, task)


# ---------------------------------------------------------------------------
# Durable stage-1 payload store
# ---------------------------------------------------------------------------
# One interval payload per file under the context's payload_store_dir, in the
# SAME format the precompute mode's interval artifacts use (see
# scattering/interval_payload) — one writer, one reader, one commit protocol
# for both durable modes. Array members are read back as in-place memmaps
# (page cache, not RSS), so a store hit costs no stage-1 compute AND almost no
# anonymous memory. Mask-empty intervals are recorded too, so their emptiness
# is durable and never re-derived.

_STORE_MISS = object()


def _store_payload_path(store_dir: str, interval_id: int) -> "Path":
    from pathlib import Path

    return Path(store_dir) / f"interval_{int(interval_id):06d}.h5"


def _legacy_npz_store_path(store_dir: str, interval_id: int) -> "Path":
    """Pre-consolidation entries. Still READ so an existing store keeps its
    value across the format change; nothing writes this shape any more."""
    from pathlib import Path

    return Path(store_dir) / f"interval_{int(interval_id):06d}.npz"


def stage1_store_has(store_dir: str, interval_id: int) -> bool:
    return (
        _store_payload_path(store_dir, interval_id).exists()
        or _legacy_npz_store_path(store_dir, interval_id).exists()
    )


def _read_legacy_npz_payload(path):
    import json

    from core.scattering.interval_payload import mmap_is_safe_for_gpu_transfer
    from core.storage.npz_mmap import mmap_npz_member

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(np.asarray(data["meta"]).item()))
    if meta.get("empty"):
        return None
    # Same constraint as the current format: never hand CUDA a mapping of a
    # network-filesystem file (see mmap_is_safe_for_gpu_transfer).
    allow_mmap = mmap_is_safe_for_gpu_transfer(path)
    arrays = {}
    for member in ("q_grid", "q_amp", "q_amp_av"):
        mapped = mmap_npz_member(path, member) if allow_mmap else None
        if mapped is None:
            with np.load(path, allow_pickle=False) as data:
                mapped = np.asarray(data[member])
        arrays[member] = mapped
    return IntervalTask(
        irecip_id=int(meta["irecip_id"]),
        element=str(meta["element"]),
        q_grid=arrays["q_grid"],
        q_amp=arrays["q_amp"],
        q_amp_av=arrays["q_amp_av"],
        q_grid_digest=meta.get("q_grid_digest"),
        half_space_role=str(meta["half_space_role"]),
        reciprocal_multiplicity=int(meta.get("reciprocal_multiplicity", 1)),
    )


def read_stored_interval_payload(store_dir: str, interval_id: int):
    """Return the stored payload, ``None`` for a recorded mask-empty interval,
    or the ``_STORE_MISS`` sentinel when nothing durable exists yet.

    Reads are memory-mapped: a store hit costs page cache, not the anonymous
    RSS that materializing multi-GB members would."""
    path = _store_payload_path(store_dir, interval_id)
    legacy_path = _legacy_npz_store_path(store_dir, interval_id)
    try:
        if path.exists():
            payload = read_interval_payload(path, mmap=True)
            return _STORE_MISS if payload is PAYLOAD_MISS else payload
        if legacy_path.exists():
            return _read_legacy_npz_payload(legacy_path)
        return _STORE_MISS
    except Exception:
        logger.warning(
            "Unreadable stage-1 store entry %s; recomputing.", path, exc_info=True
        )
        return _STORE_MISS


def write_stored_interval_payload(
    store_dir: str, interval_id: int, task: "IntervalTask | None"
) -> None:
    """Persist one interval payload (idempotent, race-safe).

    Same format and same commit protocol as the precompute-mode interval
    artifact — temp file, fsync, reopen-and-validate, rename — so the store
    inherits the artifact writer's durability instead of the weaker
    savez+replace it used to have."""
    path = _store_payload_path(store_dir, interval_id)
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_interval_payload(path, task, interval_id=int(interval_id))
    except Exception:
        logger.warning(
            "Failed to persist stage-1 store entry %s; run continues without it.",
            path,
            exc_info=True,
        )


def prewarm_stage1_payload_store(
    interval_ids,
    context: StreamingComputeContext,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> int:
    """Compute-and-persist a batch of interval payloads (a prewarm task).

    Runs on any worker — prewarm batches are NOT slot-pinned, which is what
    spreads stage-1 across every GPU up front instead of serializing it into
    each owner's first fold of a shard. Payloads are dropped as soon as they
    are durable; peak memory is one payload (plus the bounded memo)."""
    computed = 0
    for loader in lazy_streamed_interval_loaders(
        interval_ids,
        context,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    ):
        loader()
        computed += 1
    return computed


def lazy_streamed_interval_loaders(
    interval_ids,
    context: StreamingComputeContext,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> tuple:
    """Zero-arg loaders, one per interval: each call computes (or memo-fetches)
    that interval's stage-1 payload and returns it — or ``None`` when the
    interval is mask-empty. Consumers that stream payloads one at a time (the
    two-pass lattice builder) use these to keep peak memory at ONE payload
    instead of the whole shard's list. Each loader scopes the lattice-forward
    default itself, so callers need no surrounding context manager."""
    if context.nufft_eps is not None:
        nufft_eps = float(context.nufft_eps)
    if context.nufft_prefer_cpu is not None:
        nufft_prefer_cpu = bool(context.nufft_prefer_cpu)
    if context.nufft_gpu_only is not None:
        nufft_gpu_only = bool(context.nufft_gpu_only)

    def _make(interval_id: int):
        def _load() -> "IntervalTask | None":
            key = (str(context.cache_token), int(interval_id))
            cached = _memo_get(key)
            if cached is not _MEMO_MISS:
                return cached
            store_dir = getattr(context, "payload_store_dir", None)
            if store_dir:
                stored = read_stored_interval_payload(store_dir, int(interval_id))
                if stored is not _STORE_MISS:
                    _memo_store(key, stored)
                    return stored
            interval = context.interval_lookup.get(int(interval_id))
            if interval is None:
                raise KeyError(
                    "Streaming residual work unit references interval "
                    f"{int(interval_id)} that is missing from the scattering "
                    "plan's interval lookup — the residual plan and the "
                    "scattering identity disagree."
                )
            with streaming_lattice_default(True):
                task = compute_scattering_interval_payload(
                    interval,
                    B_=context.B_,
                    mask_params=context.mask_params,
                    MaskStrategy=context.MaskStrategy,
                    supercell=context.supercell,
                    original_coords=context.original_coords,
                    cells_origin=context.cells_origin,
                    elements_arr=context.elements_arr,
                    charge=context.charge,
                    use_coeff=context.use_coeff,
                    coeff_val=context.coeff_val,
                    unique_elements=list(context.unique_elements),
                    ff_factory=context.ff_factory,
                    nufft_eps=nufft_eps,
                    nufft_prefer_cpu=nufft_prefer_cpu,
                    nufft_gpu_only=nufft_gpu_only,
                )
            if store_dir:
                write_stored_interval_payload(store_dir, int(interval_id), task)
            _memo_store(key, task)
            return task

        return _load

    return tuple(_make(int(interval_id)) for interval_id in interval_ids)


__all__ = [
    "StreamingComputeContext",
    "lazy_streamed_interval_loaders",
    "prewarm_stage1_payload_store",
    "read_stored_interval_payload",
    "stage1_store_has",
    "write_stored_interval_payload",
    "clear_streaming_payload_memo",
    "stage2_streaming_enabled",
    "streaming_slot_map",
]
