"""The one on-disk format for a stage-1 interval payload.

Both durable modes persist the SAME object — an :class:`IntervalTask` —
and used to do it two different ways: per-interval HDF5 artifacts for the
default (precompute) mode, and an npz store with a JSON meta member for
streaming mode. Two writers, two readers, two commit protocols and two
skip-checks for one payload meant every integrity fix had to be made
twice, and a mode switch recomputed physics the other mode already had on
disk. This module owns the format; the artifact store and the streaming
store are callers.

Reads may be memory-mapped: the datasets are written contiguous and
uncompressed, so a reader can map them at their HDF5 offset and pay
evictable page cache instead of anonymous RSS (what the streaming store's
npz mmap did, kept).

Every payload carries the identity of the physics that produced it (see
:func:`build_interval_payload_identity`). One mode's directory layout is
digest-scoped and the other's is not, so a shared FILE is the only place
an identity both modes can check will survive; a reader that cannot prove
a payload belongs to its own run refuses it rather than serving stale
amplitudes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from core.scattering.half_space import normalize_half_space_role
from core.scattering.kernels import IntervalTask
from core.storage.digests import digest_dict, require_sha256_hex
from core.storage.hdf5_atomic import atomic_hdf5_write

logger = logging.getLogger(__name__)

INTERVAL_PAYLOAD_FORMAT = "mosaic.scattering.interval"
# 3: payloads carry `payload_identity`. Version 2 files stay readable, but
# only where the DIRECTORY already proves identity (the digest-scoped
# streaming store) — never from the shared precompute artifact directory.
INTERVAL_PAYLOAD_SCHEMA_VERSION = 3

# Returned when nothing durable exists yet — distinct from ``None``, which
# records an interval the mask emptied (a real, reusable answer).
PAYLOAD_MISS = object()

_ARRAY_MEMBERS = ("q_grid", "q_amp", "q_amp_av")


class IntervalPayloadIdentityMismatch(ValueError):
    """A payload exists but was not produced by the caller's run.

    Callers that treat an unusable payload as absence (both durable stores
    recompute on a miss) catch this; callers consuming a payload as
    required transport let it propagate — silently substituting another
    run's amplitudes is the failure this type exists to prevent."""


def build_interval_payload_identity(
    *,
    scientific_digest: str,
    source_structure_digest: str,
    eps: float,
    dtype: str,
    pre_sum_mode: str,
) -> str:
    """The identity a stage-1 payload must match to be reusable.

    A payload is ``A(q) = sum_k w_k exp(+i q.r_k)`` over the interval's
    q-grid. What determines it:

    * ``scientific_digest`` — the intervals, mask, cell, supercell, charge
      and coefficients that fix the q-grid and the weights;
    * ``source_structure_digest`` — the coordinates, cell origins and
      elements the transform sums over;
    * ``eps`` / ``dtype`` — the numerical contract of the transform itself.

    ``pre_sum_mode`` does not reach stage-1 today (it is an accumulation
    knob), but it is folded in anyway: a false miss costs one recompute
    while a false hit is wrong physics, so the conservative direction is
    free to take.

    What is deliberately EXCLUDED is the stage-2 reducer strategy — the
    only reason ``run_digest`` cannot serve as this identity. ``run_digest``
    is built with ``reducer_strategy``, which is ``"stage2-streaming"`` in
    streaming mode and ``"attempt-commit"`` in precompute mode, so the two
    durable modes address DIFFERENT run trees for identical physics
    (measured: 1760a0ae… vs cb86ad49… on the same CaTiO3 case). Keying
    payload reuse on it would make cross-mode reuse impossible by
    construction. The reducer consumes stage-1 output; it does not change
    it.

    ``run_digest`` is also missing something this identity must have: it is
    invariant to the atomic COORDINATES, because nothing in the pipeline
    populates the structure keys its scientific payload hashes (see
    docs/architecture_fix_plan_2026-08-01.md). ``source_structure_digest``
    supplies what it lacks, so displaced coordinates cannot reuse the
    undisplaced run's amplitudes."""
    return digest_dict(
        {
            "schema_version": INTERVAL_PAYLOAD_SCHEMA_VERSION,
            "scientific_digest": require_sha256_hex(
                str(scientific_digest), field_name="scientific_digest"
            ),
            "source_structure_digest": require_sha256_hex(
                str(source_structure_digest),
                field_name="source_structure_digest",
            ),
            "eps": float(eps),
            "dtype": str(dtype),
            "pre_sum_mode": str(pre_sum_mode),
        },
        domain="mosaic.scattering.interval_payload.v1",
    )


def _q_grid_digest(q_grid) -> str:
    from core.scattering.artifacts import _q_grid_digest as _digest

    return _digest(q_grid)


def interval_payload_datasets(task: IntervalTask) -> dict[str, np.ndarray]:
    """The dataset map for one payload, in the historical artifact layout."""
    return {
        "irecip_id": np.array([int(task.irecip_id)], dtype=np.int64),
        "element": np.asarray(
            str(task.element), dtype=h5py.string_dtype("utf-8")
        ),
        "q_grid": np.ascontiguousarray(task.q_grid, dtype=np.float64),
        "q_amp": np.ascontiguousarray(
            np.asarray(task.q_amp).reshape(-1), dtype=np.complex128
        ),
        "q_amp_av": np.ascontiguousarray(
            np.asarray(task.q_amp_av).reshape(-1), dtype=np.complex128
        ),
        "q_grid_digest": np.asarray(
            task.q_grid_digest
            if task.q_grid_digest
            else _q_grid_digest(task.q_grid),
            dtype=h5py.string_dtype("ascii"),
        ),
        "half_space_role": np.asarray(
            str(task.half_space_role), dtype=h5py.string_dtype("ascii")
        ),
        "reciprocal_multiplicity": np.array(
            [int(task.reciprocal_multiplicity)], dtype=np.int64
        ),
    }


def write_interval_payload(
    path: Path | str,
    task: IntervalTask | None,
    *,
    interval_id: int | None = None,
    payload_identity: str | None = None,
) -> None:
    """Persist one payload atomically (temp file -> fsync -> validate ->
    rename). ``task=None`` records a mask-emptied interval, so its
    emptiness is durable too and never re-derived.

    ``payload_identity`` stamps the producing run into the file; a payload
    written without one can only ever be reused from a directory that
    proves identity by itself."""
    path = Path(path)
    attrs: dict[str, object] = {
        "schema_version": INTERVAL_PAYLOAD_SCHEMA_VERSION,
        "format": INTERVAL_PAYLOAD_FORMAT,
    }
    if payload_identity is not None:
        attrs["payload_identity"] = str(payload_identity)
    if task is None:
        if interval_id is None:
            raise ValueError("Recording an empty interval requires interval_id.")
        atomic_hdf5_write(
            path,
            {"irecip_id": np.array([int(interval_id)], dtype=np.int64)},
            attrs={**attrs, "interval_id": int(interval_id), "empty": True},
        )
        return
    atomic_hdf5_write(
        path,
        interval_payload_datasets(task),
        attrs={**attrs, "interval_id": int(task.irecip_id), "empty": False},
    )


def mmap_is_safe_for_gpu_transfer(path: Path | str) -> bool:
    """Whether arrays mapped from ``path`` may be handed to CUDA.

    A mapping of a NETWORK-filesystem file must not be: its pages fault in
    over the network and the driver's host-to-device DMA from them fails.
    Measured in the 3-node sim — every GPU fold died with
    cudaErrorDevicesUnavailable while the byte-identical run against a
    node-local store passed — so a shared-FS store is read materialized."""
    from core.runtime.worker_hooks import path_is_network_fs

    return not path_is_network_fs(path)


def mmap_h5_dataset(path: Path | str, member: str) -> np.ndarray | None:
    """Memory-map one contiguous, uncompressed HDF5 dataset.

    Returns ``None`` when the dataset is chunked, compressed or otherwise
    unmappable — callers fall back to a materializing read."""
    try:
        with h5py.File(path, "r") as handle:
            if member not in handle:
                return None
            dataset = handle[member]
            if dataset.chunks is not None or dataset.compression is not None:
                return None
            offset = dataset.id.get_offset()
            if offset is None:  # not yet allocated (empty dataset)
                return None
            shape, dtype = dataset.shape, dataset.dtype
        if dtype.hasobject:
            return None
        return np.memmap(
            path, dtype=dtype, mode="r", offset=int(offset), shape=shape
        )
    except Exception:
        return None


def _require_payload_identity(
    path: Path,
    attrs,
    *,
    expect_identity: str | None,
    accept_unstamped: bool,
) -> None:
    """Refuse a payload the caller cannot prove belongs to its own run."""
    if expect_identity is None:
        return
    stamped = attrs.get("payload_identity")
    if stamped is None:
        if accept_unstamped:
            return
        raise IntervalPayloadIdentityMismatch(
            f"Interval payload {path} carries no payload_identity; it "
            "predates identity stamping and its directory does not scope "
            "identity, so it cannot be proven to match this run."
        )
    if isinstance(stamped, bytes):
        stamped = stamped.decode("ascii")
    if str(stamped) != str(expect_identity):
        raise IntervalPayloadIdentityMismatch(
            f"Interval payload {path} was produced by a different run "
            f"({stamped} != {expect_identity})."
        )


def check_interval_payload_identity(
    path: Path | str,
    expect_identity: str,
    *,
    accept_unstamped: bool = False,
) -> None:
    """Validate a payload's identity WITHOUT reading its arrays.

    The gate that decides whether stage-1 must rerun asks this once per
    interval, so it must cost an attribute read and not a multi-GB
    materialization. Raises exactly what :func:`read_interval_payload`
    raises: :class:`IntervalPayloadIdentityMismatch` for a payload from
    another run, ``FileNotFoundError`` for an absent one."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with h5py.File(path, "r") as data:
        _require_payload_identity(
            path,
            data.attrs,
            expect_identity=expect_identity,
            accept_unstamped=accept_unstamped,
        )


def read_interval_payload(
    path: Path | str,
    *,
    mmap: bool = False,
    expect_identity: str | None = None,
    accept_unstamped: bool = False,
):
    """Load one payload.

    Returns an :class:`IntervalTask`, ``None`` for a recorded mask-empty
    interval, or :data:`PAYLOAD_MISS` when the file is absent. Raises for a
    present-but-invalid payload — callers that treat corruption as absence
    (the streaming store self-heals by recomputing) catch it themselves.

    ``expect_identity`` rejects a payload from another run with
    :class:`IntervalPayloadIdentityMismatch`. ``accept_unstamped`` admits
    pre-stamping files and belongs ONLY to readers whose directory is
    already digest-scoped."""
    path = Path(path)
    if not path.exists():
        return PAYLOAD_MISS
    if mmap and not mmap_is_safe_for_gpu_transfer(path):
        # These arrays go straight to the GPU; a network-FS mapping would
        # make the driver DMA from pages that fault in over the wire.
        mmap = False
    with h5py.File(path, "r") as data:
        _require_payload_identity(
            path,
            data.attrs,
            expect_identity=expect_identity,
            accept_unstamped=accept_unstamped,
        )
        if bool(data.attrs.get("empty", False)):
            return None
        if "half_space_role" not in data or "reciprocal_multiplicity" not in data:
            raise ValueError(
                "Interval payloads must include half_space_role and "
                "reciprocal_multiplicity metadata."
            )
        element = data["element"][()]
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        q_grid_digest = None
        if "q_grid_digest" in data:
            q_grid_digest = data["q_grid_digest"][()]
            if isinstance(q_grid_digest, bytes):
                q_grid_digest = q_grid_digest.decode("ascii")
            else:
                q_grid_digest = str(q_grid_digest)
        half_space_role = normalize_half_space_role(data["half_space_role"][()])
        reciprocal_multiplicity = int(
            np.asarray(data["reciprocal_multiplicity"]).reshape(-1)[0]
        )
        irecip_id = int(np.asarray(data["irecip_id"]).reshape(-1)[0])
        arrays = {}
        if not mmap:
            for member in _ARRAY_MEMBERS:
                arrays[member] = np.asarray(data[member])
    if mmap:
        for member in _ARRAY_MEMBERS:
            mapped = mmap_h5_dataset(path, member)
            if mapped is None:
                with h5py.File(path, "r") as data:
                    mapped = np.asarray(data[member])
            arrays[member] = mapped
    return IntervalTask(
        irecip_id,
        str(element),
        arrays["q_grid"],
        arrays["q_amp"],
        arrays["q_amp_av"],
        q_grid_digest=q_grid_digest,
        half_space_role=half_space_role,
        reciprocal_multiplicity=reciprocal_multiplicity,
    )


__all__ = [
    "INTERVAL_PAYLOAD_FORMAT",
    "INTERVAL_PAYLOAD_SCHEMA_VERSION",
    "PAYLOAD_MISS",
    "IntervalPayloadIdentityMismatch",
    "build_interval_payload_identity",
    "check_interval_payload_identity",
    "interval_payload_datasets",
    "mmap_h5_dataset",
    "mmap_is_safe_for_gpu_transfer",
    "read_interval_payload",
    "write_interval_payload",
]
