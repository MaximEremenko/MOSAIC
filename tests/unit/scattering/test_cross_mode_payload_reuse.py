"""Cross-mode stage-1 payload reuse, and the identity that makes it safe.

Both durable modes persist the same payload in the same format, but in
directories with different guarantees: the streaming store's leaf is the
payload identity, while ``precomputed_intervals/`` is keyed by interval id
alone and is rewritten in place by every run sharing an output directory.
Reuse across that boundary is therefore admissible only against the
payload's own identity stamp — these tests pin that both modes reuse what
they may and refuse what they cannot prove.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from core.scattering.artifacts import (
    interval_artifact_reusable,
    persist_precomputed_interval_artifact,
)
from core.scattering.contracts import ScatteringWorkUnit, interval_artifact_path
from core.scattering.grid import IntervalTask
from core.scattering.interval_payload import (
    IntervalPayloadIdentityMismatch,
    build_interval_payload_identity,
    read_interval_payload,
    write_interval_payload,
)
from core.scattering.streaming import (
    _STORE_MISS,
    StreamingComputeContext,
    clear_streaming_payload_memo,
    lazy_streamed_interval_loaders,
    read_precomputed_interval_artifact,
    read_stored_interval_payload,
    resolve_stage1_payload_store_dir,
    write_stored_interval_payload,
)

SCIENCE_A = "a" * 64
SCIENCE_B = "b" * 64
STRUCT_A = "1" * 64
STRUCT_B = "2" * 64
NUMERICS = dict(eps=1e-12, dtype="complex128", pre_sum_mode="off")

IDENTITY_A = build_interval_payload_identity(
    scientific_digest=SCIENCE_A, source_structure_digest=STRUCT_A, **NUMERICS
)
IDENTITY_OTHER_STRUCTURE = build_interval_payload_identity(
    scientific_digest=SCIENCE_A, source_structure_digest=STRUCT_B, **NUMERICS
)


@pytest.fixture(autouse=True)
def _fresh_memo():
    clear_streaming_payload_memo()
    yield
    clear_streaming_payload_memo()


def _task(interval_id: int = 7, seed: int = 3, n: int = 64) -> IntervalTask:
    rng = np.random.default_rng(seed)
    return IntervalTask(
        irecip_id=interval_id,
        element="O",
        q_grid=rng.uniform(-5, 5, size=(n, 3)),
        q_amp=(rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex128
        ),
        q_amp_av=(rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex128
        ),
        q_grid_digest="abc123",
        half_space_role="positive_half",
        reciprocal_multiplicity=2,
    )


def _assert_same_payload(loaded, task) -> None:
    assert loaded.irecip_id == task.irecip_id
    assert loaded.element == task.element
    assert loaded.half_space_role == task.half_space_role
    assert loaded.reciprocal_multiplicity == task.reciprocal_multiplicity
    np.testing.assert_array_equal(loaded.q_grid, task.q_grid)
    np.testing.assert_array_equal(loaded.q_amp, task.q_amp)
    np.testing.assert_array_equal(loaded.q_amp_av, task.q_amp_av)


def _work_unit(output_dir, interval_id: int = 7) -> ScatteringWorkUnit:
    return ScatteringWorkUnit.precompute_interval(
        interval_id=interval_id,
        dimension=3,
        output_dir=str(output_dir),
    )


def _context(*, store_dir=None, artifact_dir=None, identity=None, lookup=None):
    return StreamingComputeContext(
        cache_token="tok",
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
        payload_store_dir=store_dir,
        precomputed_artifact_dir=artifact_dir,
        payload_identity=identity,
    )


class TestPayloadIdentity:
    def test_binds_the_science_the_structure_and_the_numerics(self):
        """Each input a stage-1 amplitude actually depends on must move the
        identity — the coordinates above all, since run_digest ignores them."""
        assert IDENTITY_A != IDENTITY_OTHER_STRUCTURE
        assert IDENTITY_A != build_interval_payload_identity(
            scientific_digest=SCIENCE_B, source_structure_digest=STRUCT_A, **NUMERICS
        )
        assert IDENTITY_A != build_interval_payload_identity(
            scientific_digest=SCIENCE_A,
            source_structure_digest=STRUCT_A,
            **{**NUMERICS, "eps": 1e-10},
        )
        assert IDENTITY_A == build_interval_payload_identity(
            scientific_digest=SCIENCE_A, source_structure_digest=STRUCT_A, **NUMERICS
        )

    def test_is_independent_of_the_stage2_reducer_strategy(self):
        """The identity must NOT come from run_digest: that carries the
        reducer strategy, which differs between the two durable modes, so
        keying reuse on it makes cross-mode reuse impossible by
        construction."""
        from core.scattering.planning import build_run_identity

        base = dict(
            supercell=np.array([2, 2, 2]),
            vectors=np.eye(3) * 5.0,
            reciprocal_space_intervals=[
                {"id": 0, "h_range": (1.0, 2.0), "k_range": (1.0, 2.0),
                 "l_range": (1.0, 2.0)}
            ],
            charge=0.0,
            original_coords=np.zeros((4, 3)),
            cells_origin=np.zeros((4, 3)),
            elements=np.array(["Si"] * 4),
        )
        common = dict(backend="cpu", eps=1e-12, dtype="complex128",
                      pre_sum_mode="off")
        streaming = build_run_identity(
            base, reducer_strategy="stage2-streaming", **common
        )
        precompute = build_run_identity(
            base, reducer_strategy="attempt-commit", **common
        )
        assert streaming.run_digest != precompute.run_digest
        assert streaming.scientific_digest == precompute.scientific_digest
        # ...and the payload identity, built from the shared parts, agrees.
        identity = dict(
            source_structure_digest=STRUCT_A,
            eps=1e-12, dtype="complex128", pre_sum_mode="off",
        )
        assert build_interval_payload_identity(
            scientific_digest=streaming.scientific_digest, **identity
        ) == build_interval_payload_identity(
            scientific_digest=precompute.scientific_digest, **identity
        )

    def test_rejects_a_malformed_structure_digest(self):
        with pytest.raises(ValueError):
            build_interval_payload_identity(
                scientific_digest=SCIENCE_A,
                source_structure_digest="not-a-digest",
                **NUMERICS,
            )

    def test_stamp_is_written_and_honoured(self, tmp_path):
        path = tmp_path / "interval_7.hdf5"
        task = _task()
        write_interval_payload(path, task, payload_identity=IDENTITY_A)
        with h5py.File(path, "r") as handle:
            assert handle.attrs["payload_identity"] == IDENTITY_A
        _assert_same_payload(
            read_interval_payload(path, expect_identity=IDENTITY_A), task
        )

    def test_empty_records_carry_the_stamp_too(self, tmp_path):
        path = tmp_path / "interval_9.hdf5"
        write_interval_payload(path, None, interval_id=9, payload_identity=IDENTITY_A)
        assert read_interval_payload(path, expect_identity=IDENTITY_A) is None
        with pytest.raises(IntervalPayloadIdentityMismatch):
            read_interval_payload(path, expect_identity=IDENTITY_OTHER_STRUCTURE)

    def test_mismatch_and_missing_stamp_both_raise(self, tmp_path):
        stamped = tmp_path / "stamped.hdf5"
        unstamped = tmp_path / "unstamped.hdf5"
        write_interval_payload(stamped, _task(), payload_identity=IDENTITY_A)
        write_interval_payload(unstamped, _task())
        with pytest.raises(IntervalPayloadIdentityMismatch):
            read_interval_payload(stamped, expect_identity=IDENTITY_OTHER_STRUCTURE)
        with pytest.raises(IntervalPayloadIdentityMismatch):
            read_interval_payload(unstamped, expect_identity=IDENTITY_A)
        # An unstamped payload is still readable where identity is not asked.
        _assert_same_payload(read_interval_payload(unstamped), _task())
        # ...and admissible where the DIRECTORY proves identity instead.
        _assert_same_payload(
            read_interval_payload(
                unstamped, expect_identity=IDENTITY_A, accept_unstamped=True
            ),
            _task(),
        )


class TestStoreReads:
    def test_digest_scoped_store_admits_pre_stamping_entries(self, tmp_path):
        store = str(tmp_path / "store")
        task = _task(interval_id=11)
        write_stored_interval_payload(store, 11, task)  # no identity: legacy
        _assert_same_payload(
            read_stored_interval_payload(store, 11, expect_identity=IDENTITY_A), task
        )

    def test_store_still_refuses_a_contradicting_stamp(self, tmp_path):
        store = str(tmp_path / "store")
        write_stored_interval_payload(
            store, 12, _task(interval_id=12), payload_identity=IDENTITY_OTHER_STRUCTURE
        )
        assert (
            read_stored_interval_payload(store, 12, expect_identity=IDENTITY_A)
            is _STORE_MISS
        )

    def test_shared_artifact_directory_demands_a_matching_stamp(self, tmp_path):
        artifact_dir = tmp_path / "precomputed_intervals"
        artifact_dir.mkdir()
        task = _task(interval_id=13)

        write_interval_payload(artifact_dir / "interval_13.hdf5", task)
        assert (
            read_precomputed_interval_artifact(
                str(artifact_dir), 13, expect_identity=IDENTITY_A
            )
            is _STORE_MISS
        ), "an unstamped file in a shared directory proves nothing"

        write_interval_payload(
            artifact_dir / "interval_13.hdf5",
            task,
            payload_identity=IDENTITY_OTHER_STRUCTURE,
        )
        assert (
            read_precomputed_interval_artifact(
                str(artifact_dir), 13, expect_identity=IDENTITY_A
            )
            is _STORE_MISS
        )

        write_interval_payload(
            artifact_dir / "interval_13.hdf5", task, payload_identity=IDENTITY_A
        )
        _assert_same_payload(
            read_precomputed_interval_artifact(
                str(artifact_dir), 13, expect_identity=IDENTITY_A
            ),
            task,
        )

    def test_missing_artifact_is_a_miss_not_an_error(self, tmp_path):
        assert (
            read_precomputed_interval_artifact(
                str(tmp_path), 404, expect_identity=IDENTITY_A
            )
            is _STORE_MISS
        )


class TestStreamingReusesPrecomputeArtifacts:
    def _spy(self, monkeypatch):
        calls: list[int] = []

        def fake_compute(interval, **kwargs):
            calls.append(int(interval["id"]))
            return _task(interval_id=int(interval["id"]), seed=99)

        monkeypatch.setattr(
            "core.scattering.streaming.compute_scattering_interval_payload",
            fake_compute,
        )
        return calls

    def test_validated_artifact_replaces_the_transform(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch)
        artifact_dir = tmp_path / "precomputed_intervals"
        artifact_dir.mkdir()
        task = _task(interval_id=21)
        write_interval_payload(
            artifact_dir / "interval_21.hdf5", task, payload_identity=IDENTITY_A
        )
        context = _context(
            artifact_dir=str(artifact_dir),
            identity=IDENTITY_A,
            lookup={21: {"id": 21}},
        )
        (loader,) = lazy_streamed_interval_loaders([21], context)
        _assert_same_payload(loader(), task)
        assert calls == [], "stage-1 must not run when a valid payload exists"

    def test_foreign_artifact_is_recomputed_not_consumed(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch)
        artifact_dir = tmp_path / "precomputed_intervals"
        artifact_dir.mkdir()
        stale = _task(interval_id=22, seed=1)
        write_interval_payload(
            artifact_dir / "interval_22.hdf5",
            stale,
            payload_identity=IDENTITY_OTHER_STRUCTURE,
        )
        context = _context(
            artifact_dir=str(artifact_dir),
            identity=IDENTITY_A,
            lookup={22: {"id": 22}},
        )
        (loader,) = lazy_streamed_interval_loaders([22], context)
        payload = loader()
        assert calls == [22]
        assert not np.array_equal(payload.q_amp, stale.q_amp)

    def test_unstamped_artifact_is_recomputed(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch)
        artifact_dir = tmp_path / "precomputed_intervals"
        artifact_dir.mkdir()
        write_interval_payload(
            artifact_dir / "interval_23.hdf5", _task(interval_id=23, seed=1)
        )
        context = _context(
            artifact_dir=str(artifact_dir),
            identity=IDENTITY_A,
            lookup={23: {"id": 23}},
        )
        (loader,) = lazy_streamed_interval_loaders([23], context)
        loader()
        assert calls == [23]

    def test_store_is_preferred_over_the_shared_directory(self, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch)
        store = str(tmp_path / "store")
        artifact_dir = tmp_path / "precomputed_intervals"
        artifact_dir.mkdir()
        in_store = _task(interval_id=24, seed=5)
        write_stored_interval_payload(store, 24, in_store, payload_identity=IDENTITY_A)
        write_interval_payload(
            artifact_dir / "interval_24.hdf5",
            _task(interval_id=24, seed=6),
            payload_identity=IDENTITY_A,
        )
        context = _context(
            store_dir=store,
            artifact_dir=str(artifact_dir),
            identity=IDENTITY_A,
            lookup={24: {"id": 24}},
        )
        (loader,) = lazy_streamed_interval_loaders([24], context)
        _assert_same_payload(loader(), in_store)
        assert calls == []

    def test_fresh_computes_are_stamped_into_the_store(self, tmp_path, monkeypatch):
        self._spy(monkeypatch)
        store = str(tmp_path / "store")
        context = _context(
            store_dir=store, identity=IDENTITY_A, lookup={25: {"id": 25}}
        )
        (loader,) = lazy_streamed_interval_loaders([25], context)
        loader()
        with h5py.File(tmp_path / "store" / "interval_000025.h5", "r") as handle:
            assert handle.attrs["payload_identity"] == IDENTITY_A


class TestPrecomputeAdoptsStoreEntries:
    def _import(self, tmp_path, *, store, identity):
        from core.scattering.execution import _import_stage1_store_payloads

        class _NoCacheDB:
            cache_enabled = False
            db_path = None

        return _import_stage1_store_payloads(
            [_work_unit(tmp_path, interval_id=31)],
            store_dir=store,
            payload_identity=identity,
            db=_NoCacheDB(),
        )

    def test_matching_store_entry_becomes_this_run_s_artifact(self, tmp_path):
        store = str(tmp_path / "stage1_payload_store" / IDENTITY_A)
        task = _task(interval_id=31)
        write_stored_interval_payload(store, 31, task, payload_identity=IDENTITY_A)

        remaining, adopted = self._import(tmp_path, store=store, identity=IDENTITY_A)

        assert remaining == [], "an adopted interval must not be recomputed"
        artifact = interval_artifact_path(tmp_path, 31)
        assert adopted == [artifact]
        _assert_same_payload(
            read_interval_payload(artifact, expect_identity=IDENTITY_A), task
        )

    def test_foreign_store_entry_is_left_to_recompute(self, tmp_path):
        store = str(tmp_path / "stage1_payload_store" / IDENTITY_A)
        write_stored_interval_payload(
            store, 31, _task(interval_id=31), payload_identity=IDENTITY_OTHER_STRUCTURE
        )
        remaining, adopted = self._import(tmp_path, store=store, identity=IDENTITY_A)
        assert adopted == []
        assert [unit.interval_id for unit in remaining] == [31]
        assert not interval_artifact_path(tmp_path, 31).exists()

    def test_empty_record_adopts_as_emptiness_without_an_artifact(self, tmp_path):
        store = str(tmp_path / "stage1_payload_store" / IDENTITY_A)
        write_stored_interval_payload(store, 31, None, payload_identity=IDENTITY_A)
        remaining, adopted = self._import(tmp_path, store=store, identity=IDENTITY_A)
        assert remaining == [] and adopted == []
        assert not interval_artifact_path(tmp_path, 31).exists()

    def test_disabled_store_changes_nothing(self, tmp_path):
        remaining, adopted = self._import(tmp_path, store=None, identity=IDENTITY_A)
        assert adopted == []
        assert [unit.interval_id for unit in remaining] == [31]


class TestSamePathStaleness:
    """The artifact directory is shared, so 'the file is there' has never
    been evidence that THIS run wrote it."""

    def test_foreign_artifact_at_the_same_path_is_not_reusable(self, tmp_path):
        work_unit = _work_unit(tmp_path)
        persist_precomputed_interval_artifact(
            work_unit,
            _task(),
            db_path=None,
            payload_identity=IDENTITY_OTHER_STRUCTURE,
        )
        assert interval_artifact_path(tmp_path, 7).exists()
        assert not interval_artifact_reusable(
            work_unit, payload_identity=IDENTITY_A
        )
        assert interval_artifact_reusable(
            work_unit, payload_identity=IDENTITY_OTHER_STRUCTURE
        )

    def test_unstamped_artifact_is_not_reusable_under_an_identity(self, tmp_path):
        work_unit = _work_unit(tmp_path)
        persist_precomputed_interval_artifact(work_unit, _task(), db_path=None)
        assert not interval_artifact_reusable(work_unit, payload_identity=IDENTITY_A)
        # Callers that never derived an identity keep the historical
        # existence-only behaviour.
        assert interval_artifact_reusable(work_unit, payload_identity=None)

    def test_absent_artifact_is_never_reusable(self, tmp_path):
        work_unit = _work_unit(tmp_path)
        assert not interval_artifact_reusable(work_unit, payload_identity=None)
        assert not interval_artifact_reusable(work_unit, payload_identity=IDENTITY_A)

    def test_truncated_artifact_is_treated_as_absent(self, tmp_path):
        work_unit = _work_unit(tmp_path)
        persist_precomputed_interval_artifact(
            work_unit, _task(), db_path=None, payload_identity=IDENTITY_A
        )
        path = interval_artifact_path(tmp_path, 7)
        path.write_bytes(path.read_bytes()[: 1 << 10])
        assert not interval_artifact_reusable(work_unit, payload_identity=IDENTITY_A)


class TestStoreLocationIsOneOwner:
    """Both modes must resolve the same directory or neither reuses the
    other's payloads."""

    def test_default_layout_is_identity_scoped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOSAIC_STREAMING_PAYLOAD_STORE", raising=False)
        assert resolve_stage1_payload_store_dir(str(tmp_path), IDENTITY_A) == str(
            tmp_path / "stage1_payload_store" / IDENTITY_A
        )

    def test_relocation_keeps_the_identity_leaf(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MOSAIC_STREAMING_PAYLOAD_STORE", "/scratch/mosaic")
        assert (
            resolve_stage1_payload_store_dir(str(tmp_path), IDENTITY_A)
            == f"/scratch/mosaic/{IDENTITY_A}"
        )

    def test_a_structure_change_moves_the_store(self, tmp_path, monkeypatch):
        """The old run_digest leaf did not: it is coordinate-blind, so two
        structures shared one store directory."""
        monkeypatch.delenv("MOSAIC_STREAMING_PAYLOAD_STORE", raising=False)
        assert resolve_stage1_payload_store_dir(
            str(tmp_path), IDENTITY_A
        ) != resolve_stage1_payload_store_dir(
            str(tmp_path), IDENTITY_OTHER_STRUCTURE
        )

    def test_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MOSAIC_STREAMING_PAYLOAD_STORE", "0")
        assert resolve_stage1_payload_store_dir(str(tmp_path), IDENTITY_A) is None
