"""Owner-epoch fencing on (chunk, slot) snapshot commits.

A worker the scheduler declared dead can still be alive (dask-mpi ranks
have no nanny); after the driver remaps its target, both the zombie and
the replacement could write the same snapshot seq — the manifest union
recorded whichever rename landed last and finalize aborted hours later
with a vanished slot. The fence: each ownership tenure sequences its
snapshots from epoch * STRIDE and the union refuses to move a partition's
seq backwards, so the zombie's next commit raises immediately instead."""
import numpy as np
import pytest

from core.residual_field.backend import build_residual_field_reducer_backend
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.local_accumulator import LiveLocalAccumulator


def _work_unit(tmp_path, interval_id):
    return ResidualFieldWorkUnit.interval_chunk_batch(
        interval_ids=(interval_id,),
        chunk_id=0,
        parameter_digest="f" * 12,
        output_dir=str(tmp_path),
    ).with_subchunk(subchunk_id=0, point_count=4)


def _accept(backend, tmp_path, interval_id, *, owner_epoch):
    backend.accept_local_contribution(
        _work_unit(tmp_path, interval_id),
        grid_shape_nd=np.array([4], dtype=np.int64),
        total_reciprocal_points=4,
        contribution_reciprocal_points=4,
        amplitudes_delta=np.full(4, 1 + 1j, dtype=np.complex128),
        amplitudes_average=np.full(4, 2 + 2j, dtype=np.complex128),
        point_ids=np.arange(4, dtype=np.int64),
        output_dir=str(tmp_path),
        scratch_root=str(tmp_path / "scratch"),
        db_path=str(tmp_path / "state.db"),
        total_expected_partials=64,  # cadence never fires; flush explicitly
        owner_epoch=owner_epoch,
    )


def _flush(backend, tmp_path):
    return backend.flush_local_reducer_target(
        chunk_id=0,
        parameter_digest="f" * 12,
        partition_id=0,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        cleanup_policy="off",
    )


def test_replacement_owner_seq_jumps_ahead_and_zombie_commit_is_fenced(tmp_path):
    zombie = build_residual_field_reducer_backend("local_restartable")
    replacement = build_residual_field_reducer_backend("local_restartable")

    _accept(zombie, tmp_path, interval_id=1, owner_epoch=0)
    assert _flush(zombie, tmp_path)  # zombie's tenure: seq 1

    # Driver remapped the target; the replacement folds under epoch 1 and
    # its first snapshot seq jumps past anything the zombie can reach.
    _accept(replacement, tmp_path, interval_id=2, owner_epoch=1)
    assert _flush(replacement, tmp_path)

    # The zombie is still alive and folds one more batch, then tries to
    # commit: the union must refuse to move the partition's seq backwards.
    _accept(zombie, tmp_path, interval_id=3, owner_epoch=0)
    with pytest.raises(RuntimeError, match="superseded"):
        _flush(zombie, tmp_path)


def test_epoch_floor_is_monotonic_and_epoch_zero_keeps_seq_numbering(tmp_path):
    work_unit = _work_unit(tmp_path, 1)
    accumulator = LiveLocalAccumulator.from_arrays(
        work_unit,
        point_ids=np.arange(4, dtype=np.int64),
        grid_shape_nd=np.array([4], dtype=np.int64),
        total_reciprocal_points=4,
        amplitudes_delta=np.zeros(4, dtype=np.complex128),
        amplitudes_average=np.zeros(4, dtype=np.complex128),
        scratch_root=str(tmp_path / "scratch"),
        max_ram_bytes=1 << 30,
    )
    assert accumulator.next_snapshot_seq() == 1
    accumulator.raise_owner_epoch_floor(0)
    assert accumulator.next_snapshot_seq() == 1  # epoch 0 is byte-compatible
    accumulator.raise_owner_epoch_floor(2)
    assert (
        accumulator.next_snapshot_seq()
        == 2 * LiveLocalAccumulator.OWNER_EPOCH_SEQ_STRIDE + 1
    )
    accumulator.raise_owner_epoch_floor(1)  # lower epoch never lowers the floor
    assert (
        accumulator.next_snapshot_seq()
        == 2 * LiveLocalAccumulator.OWNER_EPOCH_SEQ_STRIDE + 1
    )
