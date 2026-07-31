"""Async snapshot writer: folds must not block on durable writes, and
durability accounting must only cover CAPTURED interval sets."""
import threading

import numpy as np
import pytest

from core.residual_field import backend as backend_mod
from core.residual_field.backend import (
    _LocalSnapshotWriter,
    build_residual_field_reducer_backend,
)
from core.residual_field.contracts import ResidualFieldWorkUnit


def _work_unit(interval_id, chunk_id=1, digest="d0", output_dir="."):
    return ResidualFieldWorkUnit.interval_chunk(
        interval_id=interval_id,
        chunk_id=chunk_id,
        parameter_digest=digest,
        output_dir=output_dir,
    )


def _contribution(seed):
    rng = np.random.default_rng(seed)
    n = 16
    return {
        "grid_shape_nd": np.asarray([n], dtype=np.int64),
        "total_reciprocal_points": 4,
        "contribution_reciprocal_points": 1,
        "amplitudes_delta": (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
            np.complex128
        ),
        "amplitudes_average": (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(
            np.complex128
        ),
        "point_ids": np.arange(16, dtype=np.int64),
    }


def _accept(backend, tmp_path, interval_id, seed, total_expected=100):
    backend.accept_local_contribution(
        _work_unit(interval_id, output_dir=str(tmp_path)),
        output_dir=str(tmp_path),
        scratch_root=str(tmp_path / "scratch"),
        db_path=str(tmp_path / "db.sqlite"),
        total_expected_partials=total_expected,
        **_contribution(seed),
    )


def test_writer_single_flight_and_error_surfacing():
    writer = _LocalSnapshotWriter()
    release = threading.Event()
    entered = threading.Event()

    def slow_commit():
        entered.set()
        release.wait(5)

    assert writer.submit(("k",), slow_commit)
    entered.wait(5)
    assert writer.in_flight(("k",))
    assert not writer.submit(("k",), slow_commit)  # one in flight per key
    release.set()
    assert writer.drain(("k",)) is True
    assert writer.drain(("k",)) is False

    def failing_commit():
        raise RuntimeError("boom")

    assert writer.submit(("k",), failing_commit)
    with pytest.raises(RuntimeError, match="boom"):
        writer.drain(("k",))


class _NullStatusUpdater:
    def __init__(self, *args, **kwargs):
        pass

    def mark_saved_many(self, *args, **kwargs):
        return None


def test_fold_proceeds_while_write_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "1")
    monkeypatch.setenv("MOSAIC_RESIDUAL_CHECKPOINT_CADENCE_BATCHES", "1")
    monkeypatch.setattr(
        backend_mod, "_ResidualFieldChunkStatusUpdater", _NullStatusUpdater
    )
    backend = build_residual_field_reducer_backend("local_restartable")
    release = threading.Event()
    entered = threading.Event()
    real_write = backend_mod.write_local_accumulator_snapshot

    def blocking_write(*args, **kwargs):
        entered.set()
        assert release.wait(10), "writer never released"
        return real_write(*args, **kwargs)

    monkeypatch.setattr(
        backend_mod, "write_local_accumulator_snapshot", blocking_write
    )
    _accept(backend, tmp_path, interval_id=1, seed=1)  # triggers capture+submit
    entered.wait(5)
    # Second fold for the SAME target must complete while the write blocks.
    done = threading.Event()

    def second_fold():
        _accept(backend, tmp_path, interval_id=2, seed=2)
        done.set()

    thread = threading.Thread(target=second_fold, daemon=True)
    thread.start()
    assert done.wait(5), "fold blocked behind in-flight snapshot write"
    release.set()
    key = (1, "d0", None)
    backend._snapshot_writer().drain(key)
    accumulator = backend._local_accumulators[key]
    # Only the captured set {1} is durable; interval 2 folded mid-write
    # must stay non-durable until the next snapshot commits.
    assert accumulator.durable_interval_ids == {1}
    assert accumulator.current_interval_ids == {1, 2}
    # Explicit flush drains and commits the rest synchronously.
    assert backend.flush_local_reducer_target(
        chunk_id=1,
        parameter_digest="d0",
        partition_id=None,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "db.sqlite"),
    )
    assert accumulator.durable_interval_ids == {1, 2}


def test_sync_kill_switch_restores_inline_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "0")
    monkeypatch.setenv("MOSAIC_RESIDUAL_CHECKPOINT_CADENCE_BATCHES", "1")
    monkeypatch.setattr(
        backend_mod, "_ResidualFieldChunkStatusUpdater", _NullStatusUpdater
    )
    backend = build_residual_field_reducer_backend("local_restartable")
    _accept(backend, tmp_path, interval_id=1, seed=1)
    key = (1, "d0", None)
    accumulator = backend._local_accumulators[key]
    assert accumulator.durable_interval_ids == {1}  # committed inline
    assert not backend._snapshot_writer().in_flight(key)


def test_writer_error_raises_on_next_fold(tmp_path, monkeypatch):
    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "1")
    monkeypatch.setenv("MOSAIC_RESIDUAL_CHECKPOINT_CADENCE_BATCHES", "1")
    backend = build_residual_field_reducer_backend("local_restartable")

    def failing_write(*args, **kwargs):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(
        backend_mod, "write_local_accumulator_snapshot", failing_write
    )
    _accept(backend, tmp_path, interval_id=1, seed=1)
    key = (1, "d0", None)
    backend._snapshot_writer().drain(key) if False else None
    # wait for the failed commit to land its error
    event_deadline = threading.Event()
    for _ in range(100):
        if not backend._snapshot_writer().in_flight(key):
            break
        event_deadline.wait(0.05)
    with pytest.raises(RuntimeError, match="disk gone"):
        _accept(backend, tmp_path, interval_id=2, seed=2)


def test_pickle_roundtrip_drops_writer_state(tmp_path, monkeypatch):
    import pickle

    monkeypatch.setenv("MOSAIC_RESIDUAL_ASYNC_SNAPSHOT_WRITES", "1")
    backend = build_residual_field_reducer_backend("local_restartable")
    assert backend._snapshot_writer() is backend._snapshot_writer()
    clone = pickle.loads(pickle.dumps(backend))
    assert getattr(clone, "_snapshot_writer_obj", None) is None
    assert clone._local_accumulators == {}
