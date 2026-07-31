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

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

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


def streaming_slot_for_batch(interval_ids, n_slots: int) -> int:
    """Deterministic subchunk slot for an interval batch.

    Content-addressed (not index-based) so the same batch lands on the same
    slot across chunks, resumes, and worker restarts — which is what lets a
    slot's owner worker reuse one computed payload for every chunk it folds
    the batch into, and lets resume validation reason about slot families."""
    token = ",".join(str(int(interval_id)) for interval_id in interval_ids)
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    return int(digest[:8], 16) % max(1, int(n_slots))


# Per-process memo of computed interval payloads: consecutive work units on
# the same worker (the same batch folded into different chunks) reuse one
# stage-1 computation instead of recomputing per chunk. Entries are
# (IntervalTask | None); None records a mask-empty interval so its emptiness
# is not re-derived either.
_STREAM_MEMO: "OrderedDict[tuple[str, int], tuple[IntervalTask | None, int]]" = OrderedDict()
_STREAM_MEMO_BYTES = 0
_STREAM_MEMO_LOCK = threading.Lock()


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


def clear_streaming_payload_memo() -> None:
    global _STREAM_MEMO_BYTES
    with _STREAM_MEMO_LOCK:
        _STREAM_MEMO.clear()
        _STREAM_MEMO_BYTES = 0


def _memo_get(key: tuple[str, int]):
    with _STREAM_MEMO_LOCK:
        if key in _STREAM_MEMO:
            _STREAM_MEMO.move_to_end(key)
            return _STREAM_MEMO[key]
    return None


def _memo_store(key: tuple[str, int], task: IntervalTask | None) -> None:
    global _STREAM_MEMO_BYTES
    max_bytes = _stream_memo_max_bytes()
    if max_bytes <= 0:
        return
    nbytes = _interval_task_nbytes(task)
    if nbytes > max_bytes:
        return
    with _STREAM_MEMO_LOCK:
        old = _STREAM_MEMO.pop(key, None)
        if old is not None:
            _STREAM_MEMO_BYTES -= int(old[1])
        _STREAM_MEMO[key] = (task, nbytes)
        _STREAM_MEMO_BYTES += nbytes
        while _STREAM_MEMO_BYTES > max_bytes and len(_STREAM_MEMO) > 1:
            _k, (_task, old_bytes) = _STREAM_MEMO.popitem(last=False)
            _STREAM_MEMO_BYTES -= int(old_bytes)


# ---------------------------------------------------------------------------
# Durable stage-1 payload store
# ---------------------------------------------------------------------------
# One uncompressed .npz per interval under the context's payload_store_dir.
# Array members are read back as in-place memmaps (page cache, not RSS), so a
# store hit costs no stage-1 compute AND almost no anonymous memory. Mask-empty
# intervals are recorded as a marker file so their emptiness is durable too.

_STORE_MISS = object()


def _store_payload_path(store_dir: str, interval_id: int) -> "Path":
    from pathlib import Path

    return Path(store_dir) / f"interval_{int(interval_id):06d}.npz"


def stage1_store_has(store_dir: str, interval_id: int) -> bool:
    return _store_payload_path(store_dir, interval_id).exists()


def read_stored_interval_payload(store_dir: str, interval_id: int):
    """Return the stored payload, ``None`` for a recorded mask-empty interval,
    or the ``_STORE_MISS`` sentinel when nothing durable exists yet."""
    import json

    from core.storage.npz_mmap import mmap_npz_member

    path = _store_payload_path(store_dir, interval_id)
    if not path.exists():
        return _STORE_MISS
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(np.asarray(data["meta"]).item()))
        if meta.get("empty"):
            return None
        arrays = {}
        for member in ("q_grid", "q_amp", "q_amp_av"):
            mapped = mmap_npz_member(path, member)
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
    except Exception:
        logger.warning(
            "Unreadable stage-1 store entry %s; recomputing.", path, exc_info=True
        )
        return _STORE_MISS


def write_stored_interval_payload(
    store_dir: str, interval_id: int, task: "IntervalTask | None"
) -> None:
    """Atomically persist one interval payload (idempotent, race-safe)."""
    import json
    from uuid import uuid4

    path = _store_payload_path(store_dir, interval_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        if task is None:
            np.savez(tmp, meta=np.asarray(json.dumps({"empty": True})))
        else:
            meta = {
                "empty": False,
                "irecip_id": int(task.irecip_id),
                "element": str(task.element),
                "q_grid_digest": task.q_grid_digest,
                "half_space_role": str(task.half_space_role),
                "reciprocal_multiplicity": int(task.reciprocal_multiplicity),
            }
            np.savez(
                tmp,
                meta=np.asarray(json.dumps(meta)),
                q_grid=np.ascontiguousarray(task.q_grid, dtype=np.float64),
                q_amp=np.ascontiguousarray(
                    np.asarray(task.q_amp).reshape(-1), dtype=np.complex128
                ),
                q_amp_av=np.ascontiguousarray(
                    np.asarray(task.q_amp_av).reshape(-1), dtype=np.complex128
                ),
            )
        # np.savez appends .npz when the target lacks it; our tmp ends in .tmp
        produced = tmp if tmp.exists() else tmp.with_name(tmp.name + ".npz")
        os.replace(produced, path)
    except Exception:
        logger.warning(
            "Failed to persist stage-1 store entry %s; run continues without it.",
            path,
            exc_info=True,
        )
    finally:
        for candidate in (tmp, tmp.with_name(tmp.name + ".npz")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass


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


def _stage1_parallelism(n_intervals: int) -> int:
    """In-task thread count for streamed stage-1 interval computes.

    Defaults to a quarter of the worker's thread allotment (min 1, max 8) so
    concurrent work units on the same worker do not oversubscribe it;
    ``MOSAIC_STREAMING_STAGE1_PARALLEL`` overrides (1 disables)."""
    raw = os.getenv("MOSAIC_STREAMING_STAGE1_PARALLEL")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1, min(int(raw), max(1, int(n_intervals))))
        except (TypeError, ValueError):
            pass
    try:
        threads = int(os.getenv("DASK_THREADS_PER_WORKER", "16"))
    except ValueError:
        threads = 16
    return max(1, min(threads // 4, 8, max(1, int(n_intervals))))


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
            if cached is not None:
                return cached[0]
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


def compute_streamed_interval_tasks(
    interval_ids,
    context: StreamingComputeContext,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> tuple[IntervalTask, ...]:
    """Compute (or reuse) the scattering payloads for one interval batch.

    Mask-empty intervals produce no payload and are omitted from the result;
    the caller must still record them as incorporated — a zero contribution
    is exact, and finalize's coverage check counts every planned interval.
    The context's scattering-stage NUFFT settings take precedence over the
    caller's (residual-stage) settings."""
    if context.nufft_eps is not None:
        nufft_eps = float(context.nufft_eps)
    if context.nufft_prefer_cpu is not None:
        nufft_prefer_cpu = bool(context.nufft_prefer_cpu)
    if context.nufft_gpu_only is not None:
        nufft_gpu_only = bool(context.nufft_gpu_only)
    def _compute_one(interval_id: int) -> "IntervalTask | None":
        key = (str(context.cache_token), int(interval_id))
        cached = _memo_get(key)
        if cached is not None:
            return cached[0]
        interval = context.interval_lookup.get(int(interval_id))
        if interval is None:
            raise KeyError(
                "Streaming residual work unit references interval "
                f"{int(interval_id)} that is missing from the scattering "
                "plan's interval lookup — the residual plan and the "
                "scattering identity disagree."
            )
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
        _memo_store(key, task)
        return task

    # Streaming stage-1 runs in-task with no interval IO, so it is
    # compute-bound: default the lattice type-1 forward path ON for the
    # duration of this batch's computation (set -> try/finally -> reset via
    # the ContextVar scope). An explicit MOSAIC_SCATTERING_LATTICE_FFT env
    # value still wins inside.
    #
    # The per-interval computes are independent and mostly release the GIL
    # (numpy mask/q-grid work, GPU type-1 with the lease-guarded plan cache),
    # so a small in-task pool overlaps them. One thread per interval was the
    # dominant wall-clock term of an hkl40 shard (~0.8 s x 293 intervals
    # sequential). ContextVars do not flow into pool threads on their own;
    # each submit carries a fresh copy_context() so the lattice-ON default
    # holds inside workers.
    parallel = _stage1_parallelism(len(interval_ids))
    with streaming_lattice_default(True):
        if parallel <= 1 or len(interval_ids) <= 1:
            results = [_compute_one(int(i)) for i in interval_ids]
        else:
            import contextvars
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [
                    pool.submit(
                        contextvars.copy_context().run, _compute_one, int(i)
                    )
                    for i in interval_ids
                ]
                results = [future.result() for future in futures]
    return tuple(task for task in results if task is not None)


__all__ = [
    "StreamingComputeContext",
    "lazy_streamed_interval_loaders",
    "prewarm_stage1_payload_store",
    "read_stored_interval_payload",
    "stage1_store_has",
    "write_stored_interval_payload",
    "clear_streaming_payload_memo",
    "compute_streamed_interval_tasks",
    "stage2_streaming_enabled",
    "streaming_slot_for_batch",
]
