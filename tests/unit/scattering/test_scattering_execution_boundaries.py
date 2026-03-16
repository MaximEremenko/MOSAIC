from types import SimpleNamespace

import numpy as np

from core.scattering.accumulation import (
    build_scattering_partial_result,
    build_scattering_partial_result_from_payloads,
    materialize_scattering_payload,
)
from core.scattering.artifacts import (
    ScatteringArtifactStore,
    persist_precomputed_interval_artifact,
    persist_scattering_interval_chunk_result,
)
from core.scattering.contracts import (
    ScatteringArtifactManifest,
    ScatteringWorkUnit,
)
from core.scattering.execution import run_interval_precompute
from core.scattering.kernels import IntervalTask
from core.scattering.planning import (
    build_scattering_interval_chunk_work_units,
    build_scattering_interval_lookup,
    build_scattering_precompute_work_units,
)
from core.scattering.tasks import run_scattering_interval_chunk_task
from core.contracts import CompletionStatus
from core.storage.database_manager import DatabaseManager


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
        )

        manifest = persist_precomputed_interval_artifact(
            work_unit,
            interval_task,
            db_path=db.db_path,
        )

        assert db.is_interval_precomputed(interval_id) is True
        assert manifest.completion_status is CompletionStatus.COMMITTED
        assert manifest.artifacts[0].path is not None
    finally:
        db.close()


def test_artifacts_persist_chunk_result_updates_saved_state_and_artifacts(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    store = ScatteringArtifactStore(str(tmp_path))
    try:
        db.insert_point_data_batch(
            [
                {
                    "central_point_id": 10,
                    "coordinates": [0.1],
                    "dist_from_atom_center": [0.2],
                    "step_in_frac": [0.05],
                    "chunk_id": 3,
                    "grid_amplitude_initialized": 1,
                }
            ]
        )
        interval_id = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])[0]
        db.insert_interval_chunk_status_batch([(interval_id, 3, 0)])

        baseline = np.array([[10 + 0j, 0 + 0j], [11 + 0j, 0 + 0j]], dtype=np.complex128)
        store.save_chunk_payloads(
            3,
            amplitudes_payload=baseline,
            amplitudes_average_payload=baseline.copy(),
            reciprocal_point_count=0,
        )
        work_unit = ScatteringWorkUnit.interval_chunk(
            interval_id=interval_id,
            chunk_id=3,
            dimension=1,
            output_dir=str(tmp_path),
        )

        manifest = persist_scattering_interval_chunk_result(
            work_unit,
            grid_shape_nd=np.array([[2]]),
            total_reciprocal_points=11,
            contribution_reciprocal_points=5,
            amplitudes_delta=np.array([1 + 0j, 2 + 0j]),
            amplitudes_average=np.array([0.5 + 0j, 0.75 + 0j]),
            output_dir=str(tmp_path),
            db_path=db.db_path,
            quiet_logs=True,
        )

        current, current_av, nrec, _ = store.load_chunk_payloads(3)
        applied = store.load_applied_interval_ids(3)

        assert (interval_id, 3) not in set(db.get_unsaved_interval_chunks())
        assert manifest.completion_status is CompletionStatus.COMMITTED
        assert interval_id in applied
        np.testing.assert_allclose(current[:, 1], np.array([1 + 0j, 2 + 0j]))
        np.testing.assert_allclose(current_av[:, 1], np.array([0.5 + 0j, 0.75 + 0j]))
        assert nrec == 5
    finally:
        db.close()


def test_execution_serial_precompute_uses_work_units(monkeypatch, tmp_path):
    work_unit = ScatteringWorkUnit.precompute_interval(
        interval_id=1,
        dimension=1,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
        client=None,
    )

    assert paths == [tmp_path / "precomputed_intervals" / "interval_1.npz"]


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
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
        client=None,
        transient_interval_payloads=payload_cache,
    )

    assert paths == []
    assert 1 in payload_cache
    assert payload_cache[1].irecip_id == 1
    assert not (tmp_path / "precomputed_intervals" / "interval_1.npz").exists()


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
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
    )
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
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
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
    )
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
        client=client,
        transient_interval_payloads=payload_cache,
    )

    monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert [path.name for path in paths] == ["interval_1.npz", "interval_2.npz"]
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
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
    )
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
        client=client,
        transient_interval_payloads=payload_cache,
    )

    monkeypatch.delenv("DASK_BACKEND", raising=False)

    assert [path.name for path in paths] == ["interval_1.npz"]
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
        "core.scattering.execution.is_interval_artifact_committed",
        lambda *args, **kwargs: False,
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
        db=SimpleNamespace(db_path=str(tmp_path / "state.db")),
        client=client,
    )

    assert [path.name for path in paths] == ["interval_1.npz", "interval_2.npz"]
    assert len(client.scatter_calls) == 5
    assert len(client.submit_calls) == 2
    assert all(call[1].get("broadcast") is True for call in client.scatter_calls)
    assert all(call[1].get("hash") is False for call in client.scatter_calls)
    assert all(call[0].__name__ == "run_scattering_interval_task" for call in client.submit_calls)
    first_submit_kwargs = client.submit_calls[0][2]
    assert isinstance(first_submit_kwargs["B_"], _ScatterRef)
    assert isinstance(first_submit_kwargs["original_coords"], _ScatterRef)


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


def test_scattering_interval_chunk_task_uses_batched_inverse(monkeypatch, tmp_path):
    interval_path = tmp_path / "interval_1.npz"
    np.savez(
        interval_path,
        irecip_id=np.array([1], dtype=np.int64),
        element=np.array(["All"]),
        q_grid=np.array([[0.0]], dtype=np.float64),
        q_amp=np.array([2.0 + 0.0j]),
        q_amp_av=np.array([1.0 + 0.0j]),
    )
    atoms = np.array(
        [([0.0], [0.1], [0.05])],
        dtype=[
            ("coordinates", object),
            ("dist_from_atom_center", object),
            ("step_in_frac", object),
        ],
    )
    captured = {}

    monkeypatch.setattr(
        "core.scattering.tasks.build_rifft_grid_for_chunk",
        lambda chunk_data: (np.array([[0.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)),
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        "core.scattering.tasks.execute_inverse_cunufft_batch_materialize_once",
        lambda **kwargs: calls.__setitem__("count", calls["count"] + 1) or np.array([[3.0 + 0.0j], [4.0 + 0.0j]]),
    )
    monkeypatch.setattr(
        "core.scattering.tasks.persist_scattering_interval_chunk_result",
        lambda work_unit, **kwargs: captured.update(kwargs) or "manifest",
    )

    result = run_scattering_interval_chunk_task(
        ScatteringWorkUnit.interval_chunk(
            interval_id=1,
            chunk_id=3,
            dimension=1,
            output_dir=str(tmp_path),
        ),
        interval_path,
        atoms,
        total_reciprocal_points=11,
        output_dir=str(tmp_path),
        db_path=str(tmp_path / "state.db"),
        quiet_logs=True,
    )

    assert result == "manifest"
    assert calls["count"] == 1
    np.testing.assert_allclose(captured["amplitudes_delta"], np.array([3.0 + 0.0j]))
    np.testing.assert_allclose(captured["amplitudes_average"], np.array([4.0 + 0.0j]))
