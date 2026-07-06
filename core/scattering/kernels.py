from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from core.scattering.grid import (
    IntervalTask,
    _point_list_to_recarray,
    _process_chunk as _build_rifft_grid_for_chunk,
    _to_interval_dict,
    generate_q_space_grid,
    generate_q_space_grid_sync,
    reciprocal_space_points_counter,
)
from core.adapters.cunufft_wrapper import (
    execute_cunufft,
    execute_type1_on_lattice,
    plan_lattice,
)

logger = logging.getLogger(__name__)


# Streaming-mode default for the lattice type-1 forward path. A ContextVar
# (not a module global, not an os.environ mutation) so the streaming compute
# loop can scope it with set/reset: concurrent worker threads and test
# ordering never observe leaked state.
_STREAMING_LATTICE_DEFAULT: ContextVar[bool] = ContextVar(
    "_STREAMING_LATTICE_DEFAULT", default=False
)


@contextmanager
def streaming_lattice_default(enabled: bool = True):
    """Scope the streaming default of the lattice forward path (set -> reset)."""
    token = _STREAMING_LATTICE_DEFAULT.set(bool(enabled))
    try:
        yield
    finally:
        _STREAMING_LATTICE_DEFAULT.reset(token)


def _scattering_lattice_enabled() -> bool:
    """Forward transforms via type-1 onto the interval's lattice box.

    An EXPLICIT ``MOSAIC_SCATTERING_LATTICE_FFT`` value always wins: ``1``
    opts in, ``0`` (or anything else) opts out — including inside streaming.
    When the env is unset, the default is OFF except while executing a
    streaming payload computation (see :func:`streaming_lattice_default`),
    where it is ON: streaming stage-1 runs in-task with no interval IO, so
    the stage is compute-bound and the 3-18x faster type-1 transform (GM
    spreading, plan+setpts shared across intervals) pays off. Durable mode
    measured IO-bound end-to-end at hkl32 scale — 455 s (lattice) vs 387 s
    (type-3 with the shared-q_av dedup) — hence default OFF outside
    streaming."""
    explicit = os.getenv("MOSAIC_SCATTERING_LATTICE_FFT")
    if explicit is not None:
        return explicit == "1"
    return bool(_STREAMING_LATTICE_DEFAULT.get())


# Post-LSQ snap deviation ceiling for the forward path: truly on-lattice q
# (exact float64 products) refines to <=1e-9 of a step; near-lattice data
# (e.g. a cell sheared by ~1e-7 relative) sits around 1e-6 and MUST fall back
# to type-3 -- snapping it would corrupt every Stage-1 amplitude at ~1e-5
# relative with no warning.
_FORWARD_LATTICE_MAX_SNAP_DEV = 1e-7


def build_interval_lattice_meta(q_grid: np.ndarray) -> dict | None:
    """Lattice plan for one interval's stored q-points.

    Returns ``None`` when the lattice path is disabled or the q-points are not
    lattice-eligible (including NEAR-lattice data whose snap deviation exceeds
    the eps-parity ceiling) -- callers then keep the type-3 path. Computed
    once per interval and shared by every forward transform of that interval."""
    if not _scattering_lattice_enabled():
        return None
    try:
        return plan_lattice(
            np.asarray(q_grid, dtype=np.float64),
            n_trans=1,
            max_snap_dev=_FORWARD_LATTICE_MAX_SNAP_DEV,
        )
    except Exception as exc:
        logger.debug("Interval lattice plan rejected: %s", exc)
        return None


def forward_interval_amplitudes(
    coords: np.ndarray,
    weights: np.ndarray,
    q_grid: np.ndarray,
    *,
    lattice_meta: dict | None = None,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
) -> np.ndarray:
    """One forward transform ``A(q) = sum_k w_k exp(+i q.r_k)`` for an
    interval: lattice type-1 when eligible, type-3 otherwise. Identical to
    NUFFT eps either way."""
    if lattice_meta is not None:
        out = execute_type1_on_lattice(
            lattice_meta,
            coords,
            weights,
            eps=nufft_eps,
            prefer_cpu=nufft_prefer_cpu,
            gpu_only=nufft_gpu_only,
        )
        if out is not None:
            return out
    return execute_cunufft(
        coords,
        weights,
        q_grid,
        eps=nufft_eps,
        prefer_cpu=nufft_prefer_cpu,
        gpu_only=nufft_gpu_only,
    )


def to_interval_dict(iv: Dict[str, Any]) -> Dict[str, float]:
    return _to_interval_dict(iv)


def build_rifft_grid_for_chunk(chunk_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return _build_rifft_grid_for_chunk(chunk_data)


def point_list_to_recarray(point_data_list: list[dict]) -> np.recarray:
    return _point_list_to_recarray(point_data_list)


def compute_interval_element_contribution(
    interval: dict,
    q_grid: np.ndarray,
    element: str,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
    lattice_meta: dict | None = None,
    q_av: np.ndarray | None = None,
) -> Tuple | None:
    ff = ff_factory.calculate(q_grid, element, charge=charge)
    mask = elements_arr == element
    if not np.any(mask):
        return None
    q_amp = ff * forward_interval_amplitudes(
        original_coords[mask],
        np.ones(mask.sum()),
        q_grid,
        lattice_meta=lattice_meta,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    if q_av is None:
        # identical for every element of the interval; callers hoist it
        q_av = forward_interval_amplitudes(
            cells_origin,
            np.ones(original_coords.shape[0]),
            q_grid,
            lattice_meta=lattice_meta,
            nufft_eps=nufft_eps,
            nufft_prefer_cpu=nufft_prefer_cpu,
            nufft_gpu_only=nufft_gpu_only,
        )
    q_delta = forward_interval_amplitudes(
        original_coords[mask] - cells_origin[mask],
        np.ones(mask.sum()),
        q_grid,
        lattice_meta=lattice_meta,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    q_av_final = ff * q_av * q_delta / original_coords.shape[0]
    return (interval["id"], element, q_grid, q_amp, q_av_final)


def compute_interval_coeff_contribution(
    interval: dict,
    q_grid: np.ndarray,
    coeff: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    *,
    nufft_eps: float = 1e-12,
    nufft_prefer_cpu: bool = False,
    nufft_gpu_only: bool = False,
    lattice_meta: dict | None = None,
) -> Tuple:
    n_points = original_coords.shape[0]
    coeff_arr = coeff * (np.ones(n_points) + 1j * np.zeros(n_points))
    q_amplitudes = forward_interval_amplitudes(
        original_coords,
        coeff_arr,
        q_grid,
        lattice_meta=lattice_meta,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    q_amplitudes_av = forward_interval_amplitudes(
        cells_origin,
        coeff_arr * 0.0 + 1.0,
        q_grid,
        lattice_meta=lattice_meta,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    q_amplitudes_delta = forward_interval_amplitudes(
        original_coords - cells_origin,
        coeff_arr,
        q_grid,
        lattice_meta=lattice_meta,
        nufft_eps=nufft_eps,
        nufft_prefer_cpu=nufft_prefer_cpu,
        nufft_gpu_only=nufft_gpu_only,
    )
    q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / n_points
    return (interval["id"], "All", q_grid, q_amplitudes, q_amplitudes_av_final)


def aggregate_interval_contributions(
    contributions: list[tuple],
    *,
    use_coeff: bool,
) -> IntervalTask:
    if use_coeff:
        interval_id, element, q_grid, q_amp, q_amp_av = contributions[0]
        return IntervalTask(interval_id, element, q_grid, q_amp, q_amp_av)

    ordered = sorted(contributions, key=lambda contribution: str(contribution[1]))
    interval_id = ordered[0][0]
    q_grid = ordered[0][2]
    q_amp = np.sum(
        np.stack([np.asarray(contribution[3], dtype=np.complex128) for contribution in ordered]),
        axis=0,
        dtype=np.complex128,
    )
    q_amp_av = np.sum(
        np.stack([np.asarray(contribution[4], dtype=np.complex128) for contribution in ordered]),
        axis=0,
        dtype=np.complex128,
    )
    return IntervalTask(interval_id, "All", q_grid, q_amp, q_amp_av)


__all__ = [
    "IntervalTask",
    "aggregate_interval_contributions",
    "build_interval_lattice_meta",
    "build_rifft_grid_for_chunk",
    "compute_interval_coeff_contribution",
    "compute_interval_element_contribution",
    "forward_interval_amplitudes",
    "generate_q_space_grid",
    "generate_q_space_grid_sync",
    "point_list_to_recarray",
    "reciprocal_space_points_counter",
    "to_interval_dict",
]
