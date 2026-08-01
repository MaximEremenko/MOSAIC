"""One on-disk format for stage-1 interval payloads (M14).

The default (precompute) mode and streaming mode persist the SAME object.
They used to do it two ways — HDF5 artifacts vs an npz store with a JSON
meta member — so every integrity property had to be implemented twice.
These tests pin the consolidation: either mode's writer produces a file
the other mode's reader accepts, mask-emptiness is durable, reads can be
memory-mapped, and pre-consolidation npz stores stay readable.
"""
import numpy as np
import pytest

from core.scattering.interval_payload import (
    PAYLOAD_MISS,
    mmap_h5_dataset,
    read_interval_payload,
    write_interval_payload,
)
from core.scattering.kernels import IntervalTask
from core.scattering.streaming import (
    _STORE_MISS,
    _store_payload_path,
    read_stored_interval_payload,
    stage1_store_has,
    write_stored_interval_payload,
)
from core.scattering.tasks import _read_interval_task_payload


def _task(interval_id=7, n=16):
    rng = np.random.default_rng(interval_id)
    return IntervalTask(
        irecip_id=interval_id,
        element="O",
        q_grid=np.ascontiguousarray(rng.random((n, 3))),
        q_amp=np.ascontiguousarray(rng.random(n) + 1j * rng.random(n)),
        q_amp_av=np.ascontiguousarray(rng.random(n) + 1j * rng.random(n)),
        q_grid_digest="d" * 16,
        half_space_role="positive_half",
        reciprocal_multiplicity=2,
    )


def _assert_same_payload(actual, expected):
    assert int(actual.irecip_id) == int(expected.irecip_id)
    assert str(actual.element) == str(expected.element)
    assert str(actual.half_space_role) == str(expected.half_space_role)
    assert int(actual.reciprocal_multiplicity) == int(expected.reciprocal_multiplicity)
    # Bit-exact: the format carries FP64/complex128 through unchanged.
    for member in ("q_grid", "q_amp", "q_amp_av"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, member)), np.asarray(getattr(expected, member))
        )


def test_roundtrip_is_bit_exact(tmp_path):
    task = _task()
    path = tmp_path / "interval.h5"
    write_interval_payload(path, task)
    _assert_same_payload(read_interval_payload(path), task)


def test_streaming_store_entry_reads_as_a_precompute_artifact(tmp_path):
    """The consolidation invariant, one direction: what streaming wrote is a
    valid precompute-mode interval artifact."""
    task = _task(interval_id=11)
    write_stored_interval_payload(str(tmp_path), 11, task)
    _assert_same_payload(
        _read_interval_task_payload(_store_payload_path(str(tmp_path), 11)), task
    )


def test_precompute_artifact_reads_as_a_streaming_store_entry(tmp_path):
    """The other direction: an artifact dropped into the store directory
    under the store's name is served by the store reader — the same physics
    is no longer computed twice when a user switches modes."""
    task = _task(interval_id=12)
    write_interval_payload(_store_payload_path(str(tmp_path), 12), task)
    assert stage1_store_has(str(tmp_path), 12)
    _assert_same_payload(read_stored_interval_payload(str(tmp_path), 12), task)


def test_mask_empty_interval_is_durable_and_distinct_from_a_miss(tmp_path):
    write_stored_interval_payload(str(tmp_path), 3, None)
    assert stage1_store_has(str(tmp_path), 3)
    # None = "the mask emptied this interval", a real reusable answer;
    # _STORE_MISS = "nothing durable yet". Collapsing them would either
    # recompute empties forever or fabricate empty physics.
    assert read_stored_interval_payload(str(tmp_path), 3) is None
    assert read_stored_interval_payload(str(tmp_path), 999) is _STORE_MISS
    assert read_interval_payload(tmp_path / "nope.h5") is PAYLOAD_MISS


def test_store_reads_are_memory_mapped(tmp_path):
    """A store hit must cost page cache, not anonymous RSS — materializing
    multi-GB members is what OOM-killed the hkl40 driver before."""
    task = _task(interval_id=5, n=4096)
    write_stored_interval_payload(str(tmp_path), 5, task)
    payload = read_stored_interval_payload(str(tmp_path), 5)
    assert isinstance(payload.q_amp, np.memmap)
    assert isinstance(payload.q_grid, np.memmap)
    _assert_same_payload(payload, task)


def test_mmap_declines_chunked_or_compressed_datasets(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "compressed.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "q_amp", data=np.arange(256, dtype=np.float64), compression="gzip"
        )
    assert mmap_h5_dataset(path, "q_amp") is None
    assert mmap_h5_dataset(path, "absent") is None


def test_write_is_idempotent_and_never_rewrites(tmp_path):
    task = _task(interval_id=8)
    write_stored_interval_payload(str(tmp_path), 8, task)
    path = _store_payload_path(str(tmp_path), 8)
    stamp = path.stat().st_mtime_ns
    write_stored_interval_payload(str(tmp_path), 8, _task(interval_id=8))
    assert path.stat().st_mtime_ns == stamp


def test_legacy_npz_store_entries_stay_readable(tmp_path):
    """An existing store keeps its value across the format change: entries
    written before consolidation are still served (a cold store would mean
    recomputing every interval)."""
    import json

    task = _task(interval_id=21)
    legacy = tmp_path / "interval_000021.npz"
    np.savez(
        legacy,
        meta=np.asarray(
            json.dumps(
                {
                    "empty": False,
                    "irecip_id": 21,
                    "element": "O",
                    "q_grid_digest": "d" * 16,
                    "half_space_role": "positive_half",
                    "reciprocal_multiplicity": 2,
                }
            )
        ),
        q_grid=task.q_grid,
        q_amp=task.q_amp,
        q_amp_av=task.q_amp_av,
    )
    assert stage1_store_has(str(tmp_path), 21)
    _assert_same_payload(read_stored_interval_payload(str(tmp_path), 21), task)


def test_torn_entry_reads_as_a_miss_instead_of_failing_the_run(tmp_path):
    task = _task(interval_id=9)
    write_stored_interval_payload(str(tmp_path), 9, task)
    path = _store_payload_path(str(tmp_path), 9)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])
    assert read_stored_interval_payload(str(tmp_path), 9) is _STORE_MISS


def test_network_fs_store_reads_are_not_memory_mapped(tmp_path, monkeypatch):
    """Payload arrays go straight to the GPU. A mapping of a network-FS file
    makes the driver DMA from pages that fault in over the wire — measured in
    the 3-node sim as cudaErrorDevicesUnavailable on every GPU fold, while the
    byte-identical run against a node-local store passed. Shared-FS stores are
    therefore read materialized."""
    task = _task(interval_id=31, n=512)
    write_stored_interval_payload(str(tmp_path), 31, task)

    monkeypatch.setattr(
        "core.runtime.worker_hooks.path_filesystem_type", lambda path: "nfs"
    )
    payload = read_stored_interval_payload(str(tmp_path), 31)
    assert not isinstance(payload.q_amp, np.memmap)
    assert not isinstance(payload.q_grid, np.memmap)
    _assert_same_payload(payload, task)   # identical physics either way


def test_local_fs_store_reads_stay_memory_mapped(tmp_path, monkeypatch):
    task = _task(interval_id=32, n=512)
    write_stored_interval_payload(str(tmp_path), 32, task)
    monkeypatch.setattr(
        "core.runtime.worker_hooks.path_filesystem_type", lambda path: "ext4"
    )
    assert isinstance(read_stored_interval_payload(str(tmp_path), 32).q_amp, np.memmap)
