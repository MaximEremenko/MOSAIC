"""Striped lattice scatter must reproduce the serial scatter bitwise."""
import numpy as np

from core.residual_field.tasks import (
    _apply_scatter_stripe,
    _prepare_interval_scatter,
    _SCATTER_INVALID,
)


class _Task:
    def __init__(self, q, amp, av, role="pos"):
        self.q_grid = q
        self.q_amp = amp
        self.q_amp_av = av
        self.half_space_role = role


def _state(dims, dq, qmin):
    return {
        "dims": tuple(dims),
        "dq": np.asarray(dq, dtype=float),
        "qmin": np.asarray(qmin, dtype=float),
    }


def _mk_interval(rng, dims, n, duplicate_frac=0.0):
    idx = np.stack(
        [rng.integers(0, d, size=n) for d in dims], axis=1
    ).astype(np.int64)
    if duplicate_frac > 0:
        dup = max(1, int(n * duplicate_frac))
        idx[:dup] = idx[n - dup : n][::-1]
    q = idx.astype(np.float64) * 0.25
    amp = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    av = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    return _Task(q, amp, av)


def _serial_reference(tasks, dims):
    grids = np.zeros((2, int(np.prod(dims))), dtype=np.complex128)
    for task in tasks:
        idx = (task.q_grid / 0.25).round().astype(np.int64)
        flat = np.ravel_multi_index(tuple(idx.T), dims)
        amp = task.q_amp
        av = task.q_amp_av
        order = np.sort(flat)
        has_dup = order.size > 1 and bool(np.any(order[1:] == order[:-1]))
        if has_dup:
            np.add.at(grids[0], flat, amp - av)
            np.add.at(grids[1], flat, av)
        else:
            grids[0][flat] += amp - av
            grids[1][flat] += av
    return grids


def _striped(tasks, dims, workers):
    state = _state(dims, [0.25] * len(dims), [0.0] * len(dims))
    grids = np.zeros((2, int(np.prod(dims))), dtype=np.complex128)
    n_cells = int(np.prod(dims))
    bounds = np.linspace(0, n_cells + 1, workers + 1).astype(np.int64)
    for task in tasks:
        prepared = _prepare_interval_scatter(task, state, snap_tol=0.05)
        assert prepared is not None and prepared is not _SCATTER_INVALID
        for w in range(workers):
            _apply_scatter_stripe(
                grids, prepared, int(bounds[w]), int(bounds[w + 1])
            )
    return grids


def test_striped_equals_serial_no_duplicates():
    rng = np.random.default_rng(7)
    dims = (13, 11, 9)
    tasks = [_mk_interval(rng, dims, 400) for _ in range(5)]
    ref = _serial_reference(tasks, dims)
    for workers in (2, 3, 8):
        got = _striped(tasks, dims, workers)
        np.testing.assert_array_equal(got, ref)


def test_striped_equals_serial_with_duplicates():
    rng = np.random.default_rng(11)
    dims = (7, 5)
    tasks = [_mk_interval(rng, dims, 300, duplicate_frac=0.3) for _ in range(4)]
    ref = _serial_reference(tasks, dims)
    for workers in (2, 5):
        got = _striped(tasks, dims, workers)
        np.testing.assert_array_equal(got, ref)


def test_prepare_flags_out_of_bounds_as_invalid():
    state = _state((4, 4), [0.25, 0.25], [0.0, 0.0])
    q = np.array([[0.0, 0.0], [2.0, 0.0]])  # index 8 out of bounds
    task = _Task(q, np.ones(2, complex), np.zeros(2, complex))
    assert _prepare_interval_scatter(task, state, snap_tol=0.05) is _SCATTER_INVALID


def test_prepare_flags_off_lattice_as_invalid():
    state = _state((4, 4), [0.25, 0.25], [0.0, 0.0])
    q = np.array([[0.1, 0.0]])  # 0.4 of a step off the lattice
    task = _Task(q, np.ones(1, complex), np.zeros(1, complex))
    assert _prepare_interval_scatter(task, state, snap_tol=0.05) is _SCATTER_INVALID
