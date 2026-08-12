from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from core.scattering.accumulation import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    build_scattering_partial_result,
    build_scattering_partial_result_from_payloads,
    materialize_scattering_payload,
)
from core.scattering import execution as scattering_execution
from core.scattering.artifacts import (
    ScatteringArtifactStore,
    persist_precomputed_interval_artifact,
)
from core.scattering.contracts import (
    ScatteringArtifactManifest,
    ScatteringWorkUnit,
)
from core.scattering.execution import run_interval_precompute
from core.scattering.kernels import IntervalTask
from core.scattering.planning import (
    ScatteringWorkIdentity,
    build_scattering_interval_chunk_work_units,
    build_scattering_interval_lookup,
    build_scattering_precompute_work_units,
)
from core.scattering.tasks import (
    IntervalPayloadRef,
    clear_scattering_interval_payload_cache,
    compute_scattering_interval_payload,
    load_interval_task_payload,
)
from core.contracts import CompletionStatus
from core.storage.database_manager import DatabaseManager
from core.storage.fingerprint import file_sha256


def _write_current_interval_payload(
    path,
    *,
    q_grid,
    q_amp,
    q_amp_av,
    half_space_role="full",
    reciprocal_multiplicity=1,
):
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("irecip_id", data=np.array([1], dtype=np.int64))
        h5file.create_dataset("element", data=np.bytes_("All"))
        h5file.create_dataset("q_grid", data=np.asarray(q_grid))
        h5file.create_dataset("q_amp", data=np.asarray(q_amp))
        h5file.create_dataset("q_amp_av", data=np.asarray(q_amp_av))
        h5file.create_dataset("half_space_role", data=np.bytes_(half_space_role))
        h5file.create_dataset(
            "reciprocal_multiplicity",
            data=np.array([int(reciprocal_multiplicity)], dtype=np.int64),
        )


def test_planning_builds_deterministic_scattering_work_units(tmp_path):
    precompute_units = build_scattering_precompute_work_units(
        [{"id": 2}, {"id": 1}],
        dimension=2,
        output_dir=str(tmp_path),
    )
    assert [unit.interval_id for unit in precompute_units] == [1, 2]
    assert all(unit.chunk_id is None for unit in precompute_units)

    chunk_units = build_scattering_interval_chunk_work_units(
        [(2, 3), (1, 4), (1, 3)],
        dimension=2,
        output_dir=str(tmp_path),
    )
    assert [(unit.interval_id, unit.chunk_id) for unit in chunk_units] == [
        (1, 3),
        (1, 4),
        (2, 3),
    ]


def test_accumulation_round_trips_payloads_without_changing_layout():
    payload = np.array([[10 + 0j, 0 + 0j], [11 + 0j, 0 + 0j]], dtype=np.complex128)
    avg_payload = np.array([[10 + 0j, 0 + 0j], [11 + 0j, 0 + 0j]], dtype=np.complex128)
    partial = build_scattering_partial_result_from_payloads(
        chunk_id=3,
        contributing_interval_ids=(1,),
        amplitudes_payload=payload,
        amplitudes_average_payload=avg_payload,
        grid_shape_nd=np.array([[2, 2]]),
        reciprocal_point_count=5,
    )
    new_partial = build_scattering_partial_result(
        chunk_id=3,
        interval_id=2,
        point_ids=partial.point_ids,
        amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
        amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
        grid_shape_nd=np.array([[2, 2]]),
        reciprocal_point_count=7,
    )

    persisted = materialize_scattering_payload(payload, new_partial.point_ids, new_partial.amplitudes_delta)

    assert persisted.shape == (2, 2)
    np.testing.assert_allclose(np.real(persisted[:, 0]), np.array([10, 11]))
    np.testing.assert_allclose(persisted[:, 1], np.array([1 + 0j, 2 + 0j]))


def test_artifacts_persist_interval_artifact_marks_precomputed(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    try:
        interval_id = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])[0]
        work_unit = ScatteringWorkUnit.precompute_interval(
            interval_id=interval_id,
            dimension=1,
            output_dir=str(tmp_path),
        )
        interval_task = IntervalTask(
            interval_id,
            "All",
            np.array([[0.0]]),
            np.array([1 + 0j]),
            np.array([0 + 0j]),
            half_space_role=HALF_SPACE_ROLE_POSITIVE_HALF,
            reciprocal_multiplicity=2,
        )

        manifest = persist_precomputed_interval_artifact(
            work_unit,
            interval_task,
            db_path=db.db_path,
        )

        assert db.is_interval_precomputed(interval_id) is True
        assert manifest.completion_status is CompletionStatus.COMMITTED
        assert manifest.artifacts[0].path is not None
        loaded = load_interval_task_payload(manifest.artifacts[0].path)
        assert loaded.half_space_role == HALF_SPACE_ROLE_POSITIVE_HALF
        assert loaded.reciprocal_multiplicity == 2
    finally:
        db.close()


def test_compute_scattering_interval_payload_attaches_hkl_half_space_role(monkeypatch):
    monkeypatch.setattr(
        "core.scattering.tasks.generate_q_space_grid_sync",
        lambda *args, **kwargs: np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    monkeypatch.setattr(
        "core.scattering.tasks.compute_interval_coeff_contribution",
        lambda interval, q_grid, *args, **kwargs: (
            interval["id"],
            "All",
            q_grid,
            np.array([1.0 + 0.0j], dtype=np.complex128),
            np.array([0.0 + 0.0j], dtype=np.complex128),
        ),
    )

    interval_task = compute_scattering_interval_payload(
        {
            "id": 1,
            "h_range": (0.0, 0.0),
            "k_range": (0.0, 0.0),
            "l_range": (0.25, 0.5),
        },
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([4.0, 4.0, 4.0]),
        original_coords=np.zeros((1, 3), dtype=np.float64),
        cells_origin=np.zeros((1, 3), dtype=np.float64),
        elements_arr=np.array(["Li"], dtype=object),
        charge=0.0,
        use_coeff=True,
        coeff_val=np.array([1.0], dtype=np.float64),
        unique_elements=["Li"],
        ff_factory=None,
    )

    assert interval_task is not None
    assert interval_task.half_space_role == HALF_SPACE_ROLE_POSITIVE_HALF
    assert interval_task.reciprocal_multiplicity == 2


def test_execution_serial_precompute_uses_work_units(monkeypatch, tmp_path):
    work_unit = ScatteringWorkUnit.precompute_interval(
        interval_id=1,
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "core.scattering.execution.run_scattering_interval_task",
        lambda unit, interval, **kwargs: ScatteringArtifactManifest.from_work_unit(
            unit,
            artifacts=(unit.interval_artifact,),
            completion_status=CompletionStatus.COMMITTED,
            consumer_stage="residual_field",
        ),
    )

    paths = run_interval_precompute(
        [work_unit],
        interval_lookup=build_scattering_interval_lookup([{"id": 1, "h_range": (0.0, 1.0)}]),
        B_=np.eye(1),
        parameters={},
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0]]),
        cells_origin=np.array([[0.0]]),
        elements_arr=np.array(["El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=None,
    )

    assert paths == [tmp_path / "precomputed_intervals" / "interval_1.hdf5"]


def test_execution_local_fast_precompute_caches_payload_without_writing_interval_artifact(
    monkeypatch,
    tmp_path,
):
    work_unit = ScatteringWorkUnit.precompute_interval(
        interval_id=1,
        dimension=1,
        output_dir=str(tmp_path),
    )
    payload_cache = {}
    interval_task = IntervalTask(
        1,
        "All",
        np.array([[0.0]]),
        np.array([1 + 0j]),
        np.array([0 + 0j]),
    )

    monkeypatch.setattr(
        "core.scattering.execution.compute_scattering_interval_payload",
        lambda *args, **kwargs: interval_task,
    )
    monkeypatch.setattr(
        "core.scattering.execution.persist_precomputed_interval_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local fast path should not persist interval artifacts by default")
        ),
    )

    paths = run_interval_precompute(
        [work_unit],
        interval_lookup=build_scattering_interval_lookup([{"id": 1, "h_range": (0.0, 1.0)}]),
        B_=np.eye(1),
        parameters={"runtime_info": {}, "transient_interval_payloads": payload_cache},
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0]]),
        cells_origin=np.array([[0.0]]),
        elements_arr=np.array(["El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=None,
        transient_interval_payloads=payload_cache,
    )

    assert paths == []
    assert 1 in payload_cache
    assert payload_cache[1].irecip_id == 1
    assert not (tmp_path / "precomputed_intervals" / "interval_1.hdf5").exists()


def test_execution_async_local_fast_precompute_caches_scattered_payload_once(
    monkeypatch,
    tmp_path,
):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _ScatterRef:
        def __init__(self, value):
            self.value = value

    class _FakeClient:
        def __init__(self):
            self.scatter_calls = []
            self.submit_calls = []
            self.loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, value, **kwargs):
            self.scatter_calls.append((value, kwargs))
            if isinstance(value, dict):
                return {key: _ScatterRef(item) for key, item in value.items()}
            return _ScatterRef(value)

        def submit(self, func, *args, **kwargs):
            self.submit_calls.append((func, args, kwargs))
            return _FakeFuture(
                IntervalTask(
                    1,
                    "All",
                    np.array([[0.0]]),
                    np.array([1 + 0j]),
                    np.array([0 + 0j]),
                )
            )

    client = _FakeClient()
    payload_cache = {}
    work_unit = ScatteringWorkUnit.precompute_interval(
        interval_id=1,
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.scattering.execution.yield_futures_with_results",
        lambda futures, client: ((future, True) for future in futures),
    )
    monkeypatch.setattr(
        "core.scattering.execution.persist_precomputed_interval_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("async local fast path should not persist interval artifacts by default")
        ),
    )

    paths = run_interval_precompute(
        [work_unit],
        interval_lookup=build_scattering_interval_lookup([{"id": 1, "h_range": (0.0, 1.0)}]),
        B_=np.eye(1),
        parameters={"runtime_info": {}, "transient_interval_payloads": payload_cache},
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0]]),
        cells_origin=np.array([[0.0]]),
        elements_arr=np.array(["El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=client,
        transient_interval_payloads=payload_cache,
    )

    monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert paths == []
    assert len(client.submit_calls) == 1
    assert len(client.scatter_calls) == 6
    assert isinstance(payload_cache[1], _ScatterRef)
    assert isinstance(payload_cache[1].value, IntervalTask)
    scattered_mapping, scatter_kwargs = client.scatter_calls[-1]
    assert isinstance(scattered_mapping, dict)
    assert list(scattered_mapping.keys()) == [1]
    assert isinstance(scattered_mapping[1], IntervalTask)
    assert scatter_kwargs["broadcast"] is False
    assert scatter_kwargs["hash"] is False


def test_execution_async_local_fast_precompute_falls_back_to_required_transport_when_interval_count_is_too_large(
    monkeypatch,
    tmp_path,
):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _ScatterRef:
        def __init__(self, value):
            self.value = value

    class _FakeClient:
        def __init__(self):
            self.scatter_calls = []
            self.submit_calls = []
            self.loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, value, **kwargs):
            ref = _ScatterRef(value)
            self.scatter_calls.append((value, kwargs, ref))
            return ref

        def submit(self, func, *args, **kwargs):
            self.submit_calls.append((func, args, kwargs))
            work_unit = args[0]
            manifest = ScatteringArtifactManifest.from_work_unit(
                work_unit,
                artifacts=(work_unit.interval_artifact,),
                completion_status=CompletionStatus.COMMITTED,
                consumer_stage="residual_field",
            )
            return _FakeFuture(manifest)

    client = _FakeClient()
    payload_cache = {}
    work_units = build_scattering_precompute_work_units(
        [{"id": 1, "h_range": (0.0, 1.0)}, {"id": 2, "h_range": (1.0, 2.0)}],
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.scattering.execution.yield_futures_with_results",
        lambda futures, client: ((future, True) for future in futures),
    )
    monkeypatch.setattr(
        "core.scattering.execution.compute_scattering_interval_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe local direct-handoff should fall back before payload caching")
        ),
    )

    paths = run_interval_precompute(
        work_units,
        interval_lookup=build_scattering_interval_lookup(
            [{"id": 1, "h_range": (0.0, 1.0)}, {"id": 2, "h_range": (1.0, 2.0)}]
        ),
        B_=np.eye(1),
        parameters={
            "runtime_info": {"local_direct_handoff_max_intervals": 1},
            "transient_interval_payloads": payload_cache,
        },
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0], [1.0]]),
        cells_origin=np.array([[0.0], [0.0]]),
        elements_arr=np.array(["El", "El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=client,
        transient_interval_payloads=payload_cache,
    )

    monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert [path.name for path in paths] == ["interval_1.hdf5", "interval_2.hdf5"]
    assert payload_cache == {}
    assert len(client.scatter_calls) == 5
    assert all(call[1].get("broadcast") is True for call in client.scatter_calls)
    assert all(call[1].get("hash") is False for call in client.scatter_calls)
    assert all(call[0].__name__ == "run_scattering_interval_task" for call in client.submit_calls)


def test_execution_async_local_fast_precompute_falls_back_to_required_transport_when_payload_bytes_are_too_large(
    monkeypatch,
    tmp_path,
):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _ScatterRef:
        def __init__(self, value):
            self.value = value

    class _FakeClient:
        def __init__(self):
            self.scatter_calls = []
            self.submit_calls = []
            self.loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, value, **kwargs):
            ref = _ScatterRef(value)
            self.scatter_calls.append((value, kwargs, ref))
            return ref

        def submit(self, func, *args, **kwargs):
            self.submit_calls.append((func, args, kwargs))
            work_unit = args[0]
            manifest = ScatteringArtifactManifest.from_work_unit(
                work_unit,
                artifacts=(work_unit.interval_artifact,),
                completion_status=CompletionStatus.COMMITTED,
                consumer_stage="residual_field",
            )
            return _FakeFuture(manifest)

    client = _FakeClient()
    payload_cache = {}
    work_unit = ScatteringWorkUnit.precompute_interval(
        interval_id=1,
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setenv("DASK_BACKEND", "local")
    monkeypatch.setattr(
        "core.scattering.execution.yield_futures_with_results",
        lambda futures, client: ((future, True) for future in futures),
    )
    monkeypatch.setattr(
        "core.scattering.execution.reciprocal_space_points_counter",
        lambda *args, **kwargs: 1_000_000,
    )
    monkeypatch.setattr(
        "core.scattering.execution.compute_scattering_interval_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe local direct-handoff should fall back before payload caching")
        ),
    )

    paths = run_interval_precompute(
        [work_unit],
        interval_lookup=build_scattering_interval_lookup([{"id": 1, "h_range": (0.0, 1.0)}]),
        B_=np.eye(1),
        parameters={
            "runtime_info": {"local_direct_handoff_max_bytes": 1},
            "transient_interval_payloads": payload_cache,
        },
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0]]),
        cells_origin=np.array([[0.0]]),
        elements_arr=np.array(["El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=client,
        transient_interval_payloads=payload_cache,
    )

    monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert [path.name for path in paths] == ["interval_1.hdf5"]
    assert payload_cache == {}
    assert len(client.scatter_calls) == 5
    assert all(call[1].get("broadcast") is True for call in client.scatter_calls)
    assert all(call[0].__name__ == "run_scattering_interval_task" for call in client.submit_calls)


def test_execution_durable_precompute_scatter_shared_inputs_once_and_keeps_required_transport(
    monkeypatch,
    tmp_path,
):
    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

        def done(self):
            return True

    class _ScatterRef:
        def __init__(self, value):
            self.value = value

    class _FakeClient:
        def __init__(self):
            self.scatter_calls = []
            self.submit_calls = []
            self.loop = SimpleNamespace(asyncio_loop=object())

        def scatter(self, value, **kwargs):
            ref = _ScatterRef(value)
            self.scatter_calls.append((value, kwargs, ref))
            return ref

        def submit(self, func, *args, **kwargs):
            self.submit_calls.append((func, args, kwargs))
            work_unit = args[0]
            manifest = ScatteringArtifactManifest.from_work_unit(
                work_unit,
                artifacts=(work_unit.interval_artifact,),
                completion_status=CompletionStatus.COMMITTED,
                consumer_stage="residual_field",
            )
            return _FakeFuture(manifest)

    client = _FakeClient()
    work_units = build_scattering_precompute_work_units(
        [{"id": 1, "h_range": (0.0, 1.0)}, {"id": 2, "h_range": (1.0, 2.0)}],
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "core.scattering.execution.yield_futures_with_results",
        lambda futures, client: ((future, True) for future in futures),
    )

    paths = run_interval_precompute(
        work_units,
        interval_lookup=build_scattering_interval_lookup(
            [{"id": 1, "h_range": (0.0, 1.0)}, {"id": 2, "h_range": (1.0, 2.0)}]
        ),
        B_=np.eye(1),
        parameters={"runtime_info": {"save_scattering_interval_artifacts": False}},
        unique_elements=[],
        mask_params={},
        MaskStrategy=None,
        supercell=np.array([1]),
        output_dir=str(tmp_path),
        original_coords=np.array([[0.0], [1.0]]),
        cells_origin=np.array([[0.0], [0.0]]),
        elements_arr=np.array(["El", "El"], dtype=object),
        charge=0.0,
        ff_factory=SimpleNamespace(),
        db=SimpleNamespace(cache_enabled=True, db_path=str(tmp_path / "state.db")),
        client=client,
    )

    assert [path.name for path in paths] == ["interval_1.hdf5", "interval_2.hdf5"]
    assert len(client.scatter_calls) == 5
    assert len(client.submit_calls) == 2
    assert all(call[1].get("broadcast") is True for call in client.scatter_calls)
    assert all(call[1].get("hash") is False for call in client.scatter_calls)
    assert all(call[0].__name__ == "run_scattering_interval_task" for call in client.submit_calls)
    first_submit_kwargs = client.submit_calls[0][2]
    assert isinstance(first_submit_kwargs["B_"], _ScatterRef)
    assert isinstance(first_submit_kwargs["original_coords"], _ScatterRef)


def test_run_scattering_stage_default_leaves_pairs_for_residual_stage(
    monkeypatch,
    tmp_path,
):
    identity = ScatteringWorkIdentity(
        run_digest="run123",
        scientific_digest="1" * 64,
        execution_digest="2" * 64,
        qspace_plan_digest="3" * 64,
        backend_policy_digest="4" * 64,
        source_structure_digest="5" * 64,
    )
    calls = []

    monkeypatch.setattr(
        scattering_execution,
        "_current_scattering_identity",
        lambda **kwargs: identity,
    )
    monkeypatch.setattr(
        scattering_execution,
        "profile_output_filesystem",
        lambda *args, **kwargs: SimpleNamespace(capability_digest="cap"),
    )
    monkeypatch.setattr(
        scattering_execution,
        "build_scattering_execution_plan",
        lambda **kwargs: SimpleNamespace(
            interval_work_units=(
                ScatteringWorkUnit.precompute_interval(
                    interval_id=1,
                    dimension=1,
                    output_dir=str(tmp_path),
                ),
            ),
            total_reciprocal_points=11,
        ),
    )
    monkeypatch.setattr(
        scattering_execution,
        "rebuild_sqlite_cache_from_manifests",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        scattering_execution,
        "run_interval_precompute",
        lambda *args, **kwargs: calls.append("precompute") or [],
    )

    db_manager = SimpleNamespace(get_interval_chunks=lambda: [(1, 3)])
    result = scattering_execution.run_scattering_stage(
        parameters={
            "reciprocal_space_intervals": [{"id": 1, "h_range": (0.0, 1.0)}],
            "point_data_list": [
                {
                    "central_point_id": 10,
                    "coordinates": [0.0],
                    "dist_from_atom_center": [0.0],
                    "step_in_frac": [0.0],
                    "chunk_id": 3,
                    "grid_amplitude_initialized": 1,
                }
            ],
            "original_coords": np.array([[0.0]]),
            "cells_origin": np.array([[0.0]]),
            "elements": np.array(["El"], dtype=object),
            "vectors": np.array([[1.0]]),
            "supercell": np.array([1.0]),
            "runtime_info": {},
            "residual_parameter_digest": "abc123",
        },
        FormFactorFactoryProducer=SimpleNamespace(),
        MaskStrategy=None,
        MaskStrategyParameters={},
        db_manager=db_manager,
        output_dir=str(tmp_path),
        point_data_processor=None,
        client=None,
    )

    assert calls == ["precompute"]
    assert result["scattering_run_digest"] == "run123"
    assert result["source_scattering_commit_digest"]


def test_total_reciprocal_points_artifact_recovers_from_corrupted_file(tmp_path):
    store = ScatteringArtifactStore(str(tmp_path))
    fn = tmp_path / store.saver.generate_filename(
        0, "_amplitudes_ntotal_reciprocal_space_points"
    )
    fn.write_bytes(b"not-an-hdf5-file")

    store.ensure_total_reciprocal_points(0, 11)

    data = store.saver.load_data(fn.name)
    assert int(np.asarray(data["ntotal_reciprocal_space_points"]).ravel()[0]) == 11
    assert int(np.asarray(data["ntotal_reciprocal_points"]).ravel()[0]) == 11


def test_interval_payload_ref_uses_digest_backed_cache(monkeypatch, tmp_path):
    interval_path = tmp_path / "interval_1.hdf5"
    _write_current_interval_payload(
        interval_path,
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    clear_scattering_interval_payload_cache()
    ref = IntervalPayloadRef(
        path=str(interval_path),
        file_sha256=file_sha256(interval_path),
        interval_id=1,
    )
    open_count = {"count": 0}
    real_file = h5py.File

    def counted_file(*args, **kwargs):
        open_count["count"] += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("core.scattering.tasks.h5py.File", counted_file)

    first = load_interval_task_payload(ref)
    second = load_interval_task_payload(ref)

    assert first is second
    assert open_count["count"] == 1


def test_interval_payload_ref_rejects_digest_mismatch(tmp_path):
    interval_path = tmp_path / "interval_1.hdf5"
    _write_current_interval_payload(
        interval_path,
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )

    ref = IntervalPayloadRef(
        path=str(interval_path),
        file_sha256="0" * 64,
        interval_id=1,
    )

    with pytest.raises(ValueError, match="file_sha256 mismatch"):
        load_interval_task_payload(ref)
