"""Streaming (fused stage-1 -> stage-2) scattering mode unit tests.

End-to-end behavior (no interval store, subchunk merge, kill-resume) is
covered by the CaTiO3-small parity runs; these tests pin the pure pieces:
flag parsing, deterministic slot routing, the payload memo, and the
work-unit subchunk contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.residual_field.contracts import (
    ResidualFieldWorkUnit,
    validate_residual_field_work_unit,
)
from core.scattering import kernels as kernels_mod
from core.scattering import streaming as streaming_mod
from core.scattering.streaming import (
    StreamingComputeContext,
    clear_streaming_payload_memo,
    lazy_streamed_interval_loaders,
    stage2_streaming_enabled,
    streaming_slot_map,
)


def _load_all(interval_ids, context):
    return [loader() for loader in lazy_streamed_interval_loaders(interval_ids, context)]


@pytest.fixture(autouse=True)
def _fresh_memo():
    clear_streaming_payload_memo()
    yield
    clear_streaming_payload_memo()


class TestEnabledFlag:
    def test_mode_string_wins(self):
        assert stage2_streaming_enabled(
            {"runtime_info": {"scattering_stage2_mode": "streaming"}}
        )
        assert not stage2_streaming_enabled(
            {"runtime_info": {"scattering_stage2_mode": "replacement"}}
        )

    def test_boolean_key(self):
        assert stage2_streaming_enabled(
            {"runtime_info": {"scattering_stage2_streaming": "1"}}
        )
        assert not stage2_streaming_enabled({"runtime_info": {}})

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_SCATTERING_STAGE2_STREAMING", "1")
        assert stage2_streaming_enabled({"runtime_info": {}})
        monkeypatch.setenv("MOSAIC_SCATTERING_STAGE2_STREAMING", "0")
        assert not stage2_streaming_enabled({"runtime_info": {}})


class TestSlotRouting:
    def test_deterministic_and_in_range(self):
        batches = [(1,), (1, 2, 3), tuple(range(100, 140))]
        for n_slots in (1, 2, 7):
            mapping = streaming_slot_map(batches, n_slots)
            assert mapping == streaming_slot_map(reversed(batches), n_slots)
            for batch in batches:
                assert 0 <= mapping[batch] < n_slots

    def test_duplicate_batches_across_chunks_share_a_slot(self):
        # every chunk submits the same batch universe; the map must be keyed
        # by batch content so all chunks' copies of a batch share one slot.
        mapping = streaming_slot_map([(5, 6), (7, 8), (5, 6), (7, 8)], 4)
        assert set(mapping) == {(5, 6), (7, 8)}

    def test_balances_batches_exactly(self):
        """The whole point of retiring the sha256 assignment: rank-mod-slots
        balances batch counts per slot exactly (the hash was binomially
        skewed and could leave slots empty — the rod case concentrated its
        real scattering volume in ONE slot)."""
        batches = [tuple(range(i * 10, i * 10 + 5)) for i in range(32)]
        mapping = streaming_slot_map(batches, 4)
        counts = [0, 0, 0, 0]
        for slot in mapping.values():
            counts[slot] += 1
        assert counts == [8, 8, 8, 8]


def _context(lookup=None, token="tok"):
    return StreamingComputeContext(
        cache_token=token,
        interval_lookup=lookup or {},
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=object,
        supercell=np.array([2, 2, 2]),
        original_coords=np.zeros((4, 3)),
        cells_origin=np.zeros((4, 3)),
        elements_arr=np.array(["O"] * 4),
        charge=0.0,
        use_coeff=False,
        coeff_val=None,
        unique_elements=("O",),
        ff_factory=object,
    )


class TestLazyStreamedIntervalLoaders:
    def test_computes_memoizes_and_records_empty(self, monkeypatch):
        calls = []

        def fake_compute(interval, **kwargs):
            calls.append(interval["id"])
            if interval["id"] == 2:
                return None  # mask-empty
            task = type("T", (), {})()
            task.q_grid = np.zeros((3, 3))
            task.q_amp = np.zeros(3, dtype=np.complex128)
            task.q_amp_av = np.zeros(3, dtype=np.complex128)
            return task

        monkeypatch.setattr(
            streaming_mod, "compute_scattering_interval_payload", fake_compute
        )
        ctx = _context(lookup={1: {"id": 1}, 2: {"id": 2}, 3: {"id": 3}})
        out = _load_all((1, 2, 3), ctx)
        assert [task is None for task in out] == [False, True, False]
        assert calls == [1, 2, 3]
        out2 = _load_all((1, 2, 3), ctx)
        assert [task is None for task in out2] == [False, True, False]
        assert calls == [1, 2, 3]                 # memo: no recompute (incl. None)

    def test_unknown_interval_raises(self, monkeypatch):
        monkeypatch.setattr(
            streaming_mod,
            "compute_scattering_interval_payload",
            lambda interval, **kwargs: None,
        )
        ctx = _context(lookup={1: {"id": 1}})
        with pytest.raises(KeyError, match="missing from the scattering"):
            _load_all((1, 99), ctx)

    def test_memo_scoped_by_cache_token(self, monkeypatch):
        calls = []

        def fake_compute(interval, **kwargs):
            calls.append(interval["id"])
            return None

        monkeypatch.setattr(
            streaming_mod, "compute_scattering_interval_payload", fake_compute
        )
        _load_all((1,), _context(lookup={1: {"id": 1}}, token="a"))
        _load_all((1,), _context(lookup={1: {"id": 1}}, token="b"))
        assert calls == [1, 1]                    # different runs never share

    def test_memo_byte_cap_evicts(self, monkeypatch):
        def fake_compute(interval, **kwargs):
            task = type("T", (), {})()
            task.q_grid = np.zeros((1000, 3))
            task.q_amp = np.zeros(1000, dtype=np.complex128)
            task.q_amp_av = np.zeros(1000, dtype=np.complex128)
            return task

        monkeypatch.setattr(
            streaming_mod, "compute_scattering_interval_payload", fake_compute
        )
        monkeypatch.setenv("MOSAIC_STREAMING_PAYLOAD_MEMO_MAX_BYTES", "60000")
        lookup = {i: {"id": i} for i in range(1, 6)}
        _load_all(tuple(lookup), _context(lookup=lookup))
        assert streaming_mod._STREAM_MEMO.bytes() <= 60000
        assert len(streaming_mod._STREAM_MEMO) < 5


class TestSubchunkWorkUnitContract:
    def _batch_unit(self):
        return ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=(4, 5, 6),
            chunk_id=1,
            parameter_digest="d" * 12,
            output_dir="/tmp/x",
        )

    def test_with_subchunk_sets_axis_and_full_range(self):
        unit = self._batch_unit().with_subchunk(subchunk_id=3, point_count=50)
        assert unit.partition_axis == "intervals"
        assert (unit.partition_id, unit.point_start, unit.point_stop) == (3, 0, 50)
        assert "subchunk-3" in unit.artifact_key
        validate_residual_field_work_unit(unit)

    def test_subchunk_and_partition_keys_differ(self):
        base = self._batch_unit()
        sub = base.with_subchunk(subchunk_id=0, point_count=50)
        part = base.with_partition(partition_id=0, point_start=0, point_stop=50)
        assert sub.artifact_key != part.artifact_key
        assert sub.retry.idempotency_key != part.retry.idempotency_key

    def test_intervals_axis_requires_slot(self):
        from dataclasses import replace

        unit = replace(self._batch_unit(), partition_axis="intervals")
        with pytest.raises(ValueError, match="require a"):
            validate_residual_field_work_unit(unit)

    def test_bad_axis_rejected(self):
        from dataclasses import replace

        unit = replace(self._batch_unit(), partition_axis="bananas")
        with pytest.raises(ValueError, match="partition_axis"):
            validate_residual_field_work_unit(unit)


class TestStreamingLatticeDefault:
    """Type-1 lattice forward default: OFF, except inside streaming compute."""

    def test_env_unset_off_outside_on_inside_streaming_compute(self, monkeypatch):
        monkeypatch.delenv("MOSAIC_SCATTERING_LATTICE_FFT", raising=False)
        assert kernels_mod._scattering_lattice_enabled() is False

        seen = []

        def fake_compute(interval, **kwargs):
            seen.append(kernels_mod._scattering_lattice_enabled())
            return None

        monkeypatch.setattr(
            streaming_mod, "compute_scattering_interval_payload", fake_compute
        )
        _load_all((1,), _context(lookup={1: {"id": 1}}))
        assert seen == [True]
        # reset after the streaming compute returns
        assert kernels_mod._scattering_lattice_enabled() is False

    def test_set_reset_helper_scopes_the_default(self, monkeypatch):
        monkeypatch.delenv("MOSAIC_SCATTERING_LATTICE_FFT", raising=False)
        assert not kernels_mod._scattering_lattice_enabled()
        with kernels_mod.streaming_lattice_default(True):
            assert kernels_mod._scattering_lattice_enabled()
        assert not kernels_mod._scattering_lattice_enabled()

    def test_reset_even_when_loader_raises(self, monkeypatch):
        monkeypatch.delenv("MOSAIC_SCATTERING_LATTICE_FFT", raising=False)
        monkeypatch.setattr(
            streaming_mod,
            "compute_scattering_interval_payload",
            lambda interval, **kwargs: None,
        )
        with pytest.raises(KeyError):
            _load_all((99,), _context(lookup={1: {"id": 1}}))
        assert kernels_mod._scattering_lattice_enabled() is False

    def test_explicit_env_zero_wins_even_inside_streaming(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_SCATTERING_LATTICE_FFT", "0")

        seen = []

        def fake_compute(interval, **kwargs):
            seen.append(kernels_mod._scattering_lattice_enabled())
            return None

        monkeypatch.setattr(
            streaming_mod, "compute_scattering_interval_payload", fake_compute
        )
        _load_all((1,), _context(lookup={1: {"id": 1}}))
        assert seen == [False]
        with kernels_mod.streaming_lattice_default(True):
            assert not kernels_mod._scattering_lattice_enabled()

    def test_explicit_env_one_wins_outside_streaming(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_SCATTERING_LATTICE_FFT", "1")
        assert kernels_mod._scattering_lattice_enabled()


class TestStreamingSlotOwnerMap:
    def test_round_robin_over_sorted_targets(self):
        from core.residual_field.execution import _streaming_slot_owner_map

        workers = ["tcp://w0", "tcp://w1", "tcp://w2"]
        keys = [(0, 0), (0, 1), (0, 5), (1, 0), (1, 1), (1, 5), (7, 2)]
        # slot-major rank order: (slot,chunk) = (0,0),(0,1),(1,0),(1,1),
        # (2,7),(5,0),(5,1) -> ranks 0..6 -> round-robin over 3 workers
        assert _streaming_slot_owner_map(keys, workers) == {
            (0, 0): "tcp://w0",
            (1, 0): "tcp://w1",
            (0, 1): "tcp://w2",
            (1, 1): "tcp://w0",
            (7, 2): "tcp://w1",
            (0, 5): "tcp://w2",
            (1, 5): "tcp://w0",
        }

    def test_targets_balance_across_workers(self):
        # A sparse mask concentrates real work in ONE slot's batches
        # (observed: hkl40 'rod' ran half its residual on a single GPU
        # under slot-keyed placement). Per-(chunk, slot) round-robin must
        # spread any single slot's chunk accumulators across workers.
        from collections import Counter

        from core.residual_field.execution import _streaming_slot_owner_map

        workers = ["a", "b", "c", "d"]
        keys = [(chunk, slot) for chunk in range(4) for slot in (4, 5, 6, 7)]
        owners = _streaming_slot_owner_map(keys, workers)
        counts = Counter(owners.values())
        assert max(counts.values()) - min(counts.values()) <= 1
        heavy_slot_owners = {owners[(chunk, 6)] for chunk in range(4)}
        assert len(heavy_slot_owners) == 4  # one heavy slot -> all workers

    def test_deterministic_given_same_inputs(self):
        from core.residual_field.execution import _streaming_slot_owner_map

        workers = ["a", "b", "c"]
        keys = [(1, 5), (0, 4), (0, 5), (1, 4)]
        assert _streaming_slot_owner_map(keys, workers) == _streaming_slot_owner_map(
            list(reversed(keys)), workers
        )


class TestBatchMajorOrdering:
    def _unit(self, interval_ids, chunk_id, slot):
        return ResidualFieldWorkUnit.interval_chunk_batch(
            interval_ids=interval_ids,
            chunk_id=chunk_id,
            parameter_digest="d" * 12,
            output_dir="/tmp/x",
        ).with_subchunk(subchunk_id=slot, point_count=10)

    def test_same_batch_adjacent_within_owner_and_slots_interleaved(self):
        """Two invariants of the submission order:

        1. WITHIN a subchunk slot (one owner worker), a batch's units for the
           different chunks stay adjacent and batch-major — that is what lets
           the owner's payload memo / lattice cache serve every fold of a
           batch from one stage-1 computation.
        2. ACROSS slots, submissions round-robin so the first S units cover S
           distinct slots — pure global batch-major kept all but
           ceil(window / num_chunks) owners idle (measured: 2 of 4 GPUs busy).
        """
        from core.residual_field.execution import (
            _sort_streaming_work_units_batch_major,
        )

        x0 = self._unit((1, 2), 0, 0)
        x1 = self._unit((1, 2), 1, 0)
        y0 = self._unit((3, 4), 0, 1)
        y1 = self._unit((3, 4), 1, 1)
        # plan order is chunk-major: chunk 0's batches, then chunk 1's
        ordered = _sort_streaming_work_units_batch_major([x0, y0, x1, y1])

        first_two_slots = {int(unit.partition_id) for unit in ordered[:2]}
        assert first_two_slots == {0, 1}, "early submissions must cover all slots"

        for slot in (0, 1):
            per_owner = [
                (tuple(unit.interval_ids), int(unit.chunk_id))
                for unit in ordered
                if int(unit.partition_id) == slot
            ]
            assert per_owner == sorted(per_owner), (
                "each owner's queue must stay batch-major"
            )

    def test_single_slot_order_is_plain_batch_major(self):
        from core.residual_field.execution import (
            _sort_streaming_work_units_batch_major,
        )

        x0 = self._unit((1, 2), 0, 0)
        x1 = self._unit((1, 2), 1, 0)
        y0 = self._unit((3, 4), 0, 0)
        y1 = self._unit((3, 4), 1, 0)
        ordered = _sort_streaming_work_units_batch_major([x0, y0, x1, y1])
        assert [
            (tuple(unit.interval_ids), int(unit.chunk_id)) for unit in ordered
        ] == [((1, 2), 0), ((1, 2), 1), ((3, 4), 0), ((3, 4), 1)]

    def test_owner_keyed_interleave_covers_workers_when_count_divides_chunks(self):
        """hkl40's degenerate shape: W workers, C chunks, W divides C.

        The (slot,chunk) owner map collapses to worker = chunk mod W there,
        and every slot queue's head is the lowest chunk — slot-keyed
        interleave then submits the first S units to ONE worker. Keying the
        round-robin by resolved owner must cover all W workers within the
        first W submissions regardless of prefetch."""
        from core.residual_field.execution import (
            _sort_streaming_work_units_batch_major,
            _streaming_slot_owner_map,
        )

        workers = [f"tcp://w{i}" for i in range(4)]
        chunks = range(4)
        slots = range(8)
        units = [
            self._unit((10 * slot + 1, 10 * slot + 2), chunk, slot)
            for slot in slots
            for chunk in chunks
        ]
        target_keys = [(int(u.chunk_id), u.partition_id) for u in units]
        owners = _streaming_slot_owner_map(target_keys, workers)

        ordered = _sort_streaming_work_units_batch_major(
            units, target_owners=owners
        )
        first_owners = {
            owners[(int(unit.chunk_id), unit.partition_id)]
            for unit in ordered[: len(workers)]
        }
        assert first_owners == set(workers), (
            "first W submissions must land on W distinct workers"
        )
        # locality invariant survives: each owner's own queue stays batch-major
        for worker in workers:
            per_owner = [
                (tuple(unit.interval_ids), int(unit.chunk_id))
                for unit in ordered
                if owners[(int(unit.chunk_id), unit.partition_id)] == worker
            ]
            assert per_owner == sorted(per_owner)
