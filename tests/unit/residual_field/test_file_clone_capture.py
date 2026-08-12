"""File-mode async snapshot capture via reflink clone.

hkl40-scale accumulators are always file-mode, so the async writer used to
be gated OFF for exactly the workload it was built for. The clone capture
closes that: on reflink-capable filesystems (XFS/btrfs/NFS4.2) the capture
is an O(1) copy-on-write of the memmap backing files; elsewhere it raises
SnapshotCloneUnsupported and the backend falls back to the synchronous
flush (previous behavior, unchanged results either way)."""
import numpy as np
import pytest

from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.local_accumulator import (
    LiveLocalAccumulator,
    SnapshotCloneUnsupported,
    _reflink_clone_file,
)


def _file_mode_accumulator(tmp_path, n=64):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1,
        chunk_id=0,
        parameter_digest="c" * 12,
        output_dir=str(tmp_path),
    )
    return LiveLocalAccumulator.from_arrays(
        work_unit,
        point_ids=np.arange(n, dtype=np.int64),
        grid_shape_nd=np.array([n], dtype=np.int64),
        total_reciprocal_points=n,
        amplitudes_delta=np.full(n, 1 + 1j, dtype=np.complex128),
        amplitudes_average=np.full(n, 2 + 2j, dtype=np.complex128),
        scratch_root=str(tmp_path / "scratch"),
        max_ram_bytes=0,  # force file mode
    )


def _reflink_supported(tmp_path) -> bool:
    probe_src = tmp_path / "probe_src"
    probe_dst = tmp_path / "probe_dst"
    probe_src.write_bytes(b"x" * 4096)
    try:
        _reflink_clone_file(probe_src, probe_dst)
    except SnapshotCloneUnsupported:
        return False
    return True


def test_ram_mode_accumulator_rejects_clone_capture(tmp_path):
    work_unit = ResidualFieldWorkUnit.interval_chunk(
        interval_id=1, chunk_id=0, parameter_digest="c" * 12, output_dir=str(tmp_path)
    )
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
    assert accumulator.storage_mode == "ram"
    with pytest.raises(SnapshotCloneUnsupported):
        accumulator.capture_snapshot_payload_file_clone()


def test_unsupported_fs_raises_and_leaves_no_clone_litter(tmp_path):
    accumulator = _file_mode_accumulator(tmp_path)
    assert accumulator.storage_mode == "file"
    if _reflink_supported(tmp_path):
        pytest.skip("filesystem supports reflink; failure path not reachable")
    with pytest.raises(SnapshotCloneUnsupported):
        accumulator.capture_snapshot_payload_file_clone()
    litter = list(accumulator.live_dir.glob("*.clone.npy"))
    assert litter == []


def test_clone_capture_freezes_arrays_against_later_folds(tmp_path):
    accumulator = _file_mode_accumulator(tmp_path)
    if not _reflink_supported(tmp_path):
        pytest.skip("filesystem lacks reflink (ext4/tmpfs)")
    payload = accumulator.capture_snapshot_payload_file_clone()
    frozen = np.array(payload["amplitudes_delta"], copy=True)
    accumulator.amplitudes_delta[:] += 100.0  # a "fold" after capture
    accumulator.amplitudes_delta.flush()
    assert np.array_equal(np.asarray(payload["amplitudes_delta"]), frozen)
    for clone in payload["_clone_paths"]:
        assert clone.exists()
        clone.unlink()
