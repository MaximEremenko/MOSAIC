from types import SimpleNamespace

import numpy as np

from core.scattering.context import build_amplitude_execution_context
from core.scattering.service import ScatteringExecutionService
from core.scattering.coefficients import CoefficientCenteringService
from core.config import ParameterLoadingService
from core.scattering.form_factors.registry import FormFactorRegistry
from core.qspace.masking.service import MaskStrategyService
from core.patch_centers.service import PointSelectionService
from core.decoding.context import build_postprocessing_context
from core.decoding.service import DecodingService
from core.qspace.service import ReciprocalSpacePreparationService
from core.residual_field.service import ResidualFieldExecutionService
from core.qspace.intervals.interval_reconstruction import (
    IntervalReconstructionService,
)
from core.structure.service import StructureLoadingService
from core.workflow import WorkflowService
from core.models import (
    PointData,
    PointSelectionRequest,
    ReciprocalSpaceArtifacts,
    StructureData,
    WorkflowParameters,
)
from core.storage.database_manager import (
    DatabaseManager,
    create_db_manager_for_thread,
)


def _build_workflow_parameters() -> WorkflowParameters:
    return WorkflowParameters(
        schema_version=1,
        struct_info={"dimension": 1},
        peak_info={"mask_equation": "h > 0"},
        rspace_info={"method": "central", "mode": "displacement", "use_coeff": True},
        runtime_info={},
    )


def _build_structure() -> StructureData:
    return StructureData(
        vectors=np.array([[1.0]]),
        metric={"a": 1.0},
        supercell=np.array([4]),
        original_coords=np.array([[0.1]]),
        average_coords=np.array([[0.0]]),
        elements=np.array(["El1"], dtype=object),
        refnumbers=np.array([1]),
        cells_origin=np.array([[0.0]]),
        coeff=np.array([2.0]),
    )


def _build_artifacts(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=1)
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 0,
                "coordinates": [0.1],
                "dist_from_atom_center": [0.2],
                "step_in_frac": [0.05],
                "chunk_id": 3,
                "grid_amplitude_initialized": 0,
            }
        ]
    )
    interval_ids = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])
    db.insert_interval_chunk_status_batch([(interval_ids[0], 3, 0)])
    return ReciprocalSpaceArtifacts(
        output_dir=str(tmp_path),
        saver=SimpleNamespace(),
        point_data_processor=SimpleNamespace(),
        db_manager=db,
        compact_intervals=[{"h_range": (0.0, 1.0)}],
        padded_intervals=[{"h_range": (0.0, 1.0), "h_start": 0.0, "h_end": 1.0, "k_range": (0, 0), "k_start": 0, "k_end": 0, "l_range": (0, 0), "l_start": 0, "l_end": 0}],
    )


def _build_point_data() -> PointData:
    return PointData(
        coordinates=np.array([[0.1]]),
        dist_from_atom_center=np.array([[0.2]]),
        step_in_frac=np.array([[0.05]]),
        central_point_ids=np.array([0]),
        chunk_ids=np.array([3]),
        grid_amplitude_initialized=np.array([False]),
    )


def test_coefficient_centering_service_centers_by_refnumber():
    coeff = np.array([1.0, 2.0, 10.0, 14.0])
    refnumbers = np.array([1, 1, 2, 2])
    centered = CoefficientCenteringService().center(coeff, refnumbers, "refnumbers")
    np.testing.assert_allclose(centered, np.array([-0.5, 0.5, -2.0, 2.0]))


def test_mask_strategy_service_builds_equation_strategy():
    strategy = MaskStrategyService().build(1, {"mask_equation": "h > 0"})
    mask = strategy.generate_mask(np.array([[1.0], [-1.0]]))
    assert mask.tolist() == [True, False]


def test_mask_strategy_service_defaults_to_no_mask_without_definition():
    strategy = MaskStrategyService().build(1, {})
    mask = strategy.generate_mask(np.array([[1.0], [-1.0]]))
    assert mask.tolist() == [True, True]


def test_interval_reconstruction_service_loads_pending_work(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    try:
        work = IntervalReconstructionService().load_pending_work(artifacts, 1)
        assert len(work.point_rows) == 1
        assert work.intervals == [
            {
                "h_range": (0.0, 1.0),
                "h_start": 0.0,
                "h_end": 1.0,
                "k_range": (0, 0),
                "k_start": 0,
                "k_end": 0,
                "l_range": (0, 0),
                "l_start": 0,
                "l_end": 0,
                "id": 1,
            }
        ]
    finally:
        artifacts.db_manager.close()


def test_amplitude_execution_service_builds_typed_context(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    try:
        service = ScatteringExecutionService()
        context = build_amplitude_execution_context(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
            parameter_loading_service=service.parameter_loading_service,
            coefficient_centering_service=service.coefficient_centering_service,
            mask_strategy_service=service.mask_strategy_service,
            interval_reconstruction_service=service.interval_reconstruction_service,
        )
        assert context.dimension == 1
        assert context.form_factor_selection.family == "neutron"
        assert context.form_factor_selection.calculator == "default"
        np.testing.assert_allclose(context.centered_coefficients, np.array([2.0]))
        assert len(context.unsaved_interval_chunks) == 1
        assert len(context.intervals) == 1
    finally:
        artifacts.db_manager.close()


def test_amplitude_execution_service_uses_injected_compute_callable(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    captured = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)

    try:
        service = ScatteringExecutionService(compute_amplitudes=fake_compute)
        result = service.execute(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
            client=None,
        )
        assert result["postprocessing_mode"] == "displacement"
        assert captured["db_manager"] is artifacts.db_manager
        assert captured["point_data_processor"] is artifacts.point_data_processor
    finally:
        artifacts.db_manager.close()


def test_postprocessing_service_builds_typed_context(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    try:
        context = build_postprocessing_context(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
        )
        assert context.postprocessing_mode == "displacement"
        assert context.artifacts.output_dir == str(tmp_path)
    finally:
        artifacts.db_manager.close()


def test_postprocessing_service_uses_injected_processor_factory(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    calls = []

    class FakeProcessor:
        def process_chunk(self, chunk_id, rifft_saver, output_dir):
            calls.append((chunk_id, rifft_saver, output_dir))

    factory_calls = []

    def processor_factory(db_manager, point_data_processor, parameters):
        factory_calls.append((db_manager, point_data_processor, parameters))
        return FakeProcessor()

    try:
        service = DecodingService(processor_factory=processor_factory)
        service.execute(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
            client=None,
        )
        assert factory_calls[0][0] is artifacts.db_manager
        assert factory_calls[0][1] is artifacts.point_data_processor
        assert calls == [(3, artifacts.saver, str(tmp_path))]
    finally:
        artifacts.db_manager.close()


def test_point_selection_service_uses_injected_processor_factory(tmp_path):
    captured = {}
    result = _build_point_data()

    class FakeProcessor:
        def process_parameters(self):
            captured["processed"] = True

        def get_point_data(self):
            return result

    def processor_factory(method, parameters, average_structure=None):
        captured["method"] = method
        captured["parameters"] = parameters
        captured["average_structure"] = average_structure
        return FakeProcessor()

    service = PointSelectionService(processor_factory=processor_factory)
    request = PointSelectionRequest(
        method="central",
        parameters=_build_workflow_parameters(),
        structure=_build_structure(),
        hdf5_file_path=str(tmp_path / "point_data.hdf5"),
    )

    selected = service.select(request)

    assert selected is result
    assert captured["processed"] is True
    assert captured["method"] == "central"
    assert captured["parameters"]["hdf5_file_path"].endswith("point_data.hdf5")
    assert "average_coords" in captured["average_structure"]


def test_structure_loading_service_uses_injected_registry(monkeypatch, tmp_path):
    captured = {}

    class FakeProcessor:
        def process(self):
            captured["processed"] = True

        def get_vectors(self):
            return np.array([[1.0]])

        def get_metric(self):
            return {"a": 1.0}

        def get_supercell(self):
            return np.array([4])

        def get_coordinates(self):
            return np.array([[0.1]])

        def get_average_coordinates(self):
            return np.array([[0.0]])

        def get_elements(self):
            return SimpleNamespace(apply=lambda fn: np.array([fn("El1")]))

        def get_refnumbers(self):
            return np.array([1])

        def get_cells_origin(self):
            return SimpleNamespace(to_numpy=lambda: np.array([[0.0]]))

    class FakeFactory:
        def create_processor(self, cfg_path, mode):
            captured["cfg_path"] = cfg_path
            captured["mode"] = mode
            return FakeProcessor()

    class FakeRegistry:
        def get_factory(self, file_type):
            captured["file_type"] = file_type
            return FakeFactory()

    monkeypatch.setattr(
        "core.structure.service.resolve_structure_coefficients",
        lambda **kwargs: np.array([2.0]),
    )
    workflow_parameters = WorkflowParameters(
        schema_version=1,
        struct_info={"dimension": 1, "filename": "sample_1d.f1d"},
        peak_info={},
        rspace_info={},
        runtime_info={},
    )
    service = StructureLoadingService(registry=FakeRegistry())

    structure = service.load(workflow_parameters, str(tmp_path))

    assert captured["file_type"] == "f1d"
    assert captured["cfg_path"].endswith("sample_1d.f1d")
    assert captured["mode"] == "calculate"
    assert captured["processed"] is True
    np.testing.assert_allclose(structure.coeff, np.array([2.0]))


def test_reciprocal_space_service_uses_injected_artifact_builder(tmp_path):
    calls = []

    class FakePointDataProcessor:
        def process_point_data(self, point_data):
            calls.append(("process_point_data", point_data))

    class FakeDbManager:
        def insert_point_data_batch(self, rows):
            calls.append(("insert_point_data_batch", rows))

        def insert_reciprocal_space_interval_batch(self, intervals):
            calls.append(("insert_interval_batch", intervals))
            return [7]

        def insert_interval_chunk_status_batch(self, status_rows):
            calls.append(("insert_status_batch", status_rows))

    class FakeReciprocalManager:
        reciprocal_space_intervals = [{"h_range": (0.0, 1.0)}]

        def process_reciprocal_space_intervals(self):
            calls.append(("process_intervals", None))

    class FakeArtifactBuilder:
        def create(self, *, workflow_parameters, output_dir, supercell):
            calls.append(("build_artifacts", output_dir, tuple(supercell.tolist())))
            return SimpleNamespace(
                parameters=workflow_parameters.to_payload(),
                dimension=1,
                saver=SimpleNamespace(name="saver"),
                point_data_processor=FakePointDataProcessor(),
                db_manager=FakeDbManager(),
                reciprocal_manager=FakeReciprocalManager(),
            )

    service = ReciprocalSpacePreparationService(artifact_builder=FakeArtifactBuilder())
    point_data = _build_point_data()

    artifacts = service.prepare(
        workflow_parameters=_build_workflow_parameters(),
        point_data=point_data,
        supercell=np.array([4]),
        output_dir=str(tmp_path),
    )

    assert artifacts.saver.name == "saver"
    assert [name for name, *_ in calls] == [
        "build_artifacts",
        "process_point_data",
        "insert_point_data_batch",
        "process_intervals",
        "insert_interval_batch",
        "insert_status_batch",
    ]


def test_reciprocal_space_artifacts_close_delegates_to_db_manager():
    state = {"closed": False}

    class FakeDbManager:
        def close(self):
            state["closed"] = True

    artifacts = ReciprocalSpaceArtifacts(
        output_dir="/tmp/example",
        saver=SimpleNamespace(),
        point_data_processor=SimpleNamespace(),
        db_manager=FakeDbManager(),
        compact_intervals=[],
        padded_intervals=[],
    )

    artifacts.close()

    assert state["closed"] is True


def test_workflow_service_uses_injected_services_and_closes_artifacts(tmp_path):
    calls = []

    class FakeStructureService:
        def load(self, workflow_parameters, working_path):
            calls.append(("load_structure", working_path))
            return SimpleNamespace(supercell=np.array([4]))

    class FakePointSelectionService:
        def select(self, request):
            calls.append(("select_points", request.method))
            return SimpleNamespace(chunk_ids=np.array([0]))

    class FakeArtifacts:
        def __init__(self):
            self.closed = False
            self.output_dir = str(tmp_path / "processed_point_data")

        def close(self):
            self.closed = True

    artifacts = FakeArtifacts()

    class FakeReciprocalService:
        def prepare(self, workflow_parameters, point_data, supercell, output_dir):
            calls.append(("prepare_reciprocal", output_dir))
            return artifacts

    class FakeAmplitudeService:
        def execute(self, **kwargs):
            calls.append(("amplitude", kwargs["artifacts"]))
            return {"stage": "scattering"}

    class FakeResidualFieldService:
        def execute(self, **kwargs):
            calls.append(
                (
                    "residual_field",
                    kwargs["artifacts"],
                    kwargs["scattering_parameters"],
                )
            )
            return {}

    class FakeDecodingService:
        def execute(self, **kwargs):
            calls.append(("postprocessing", kwargs["artifacts"]))
            return None

    workflow_service = WorkflowService(
        structure_loading_service=FakeStructureService(),
        point_selection_service=FakePointSelectionService(),
        reciprocal_space_service=FakeReciprocalService(),
        amplitude_service=FakeAmplitudeService(),
        residual_field_service=FakeResidualFieldService(),
        postprocessing_service=FakeDecodingService(),
    )
    workflow_parameters = _build_workflow_parameters()
    workflow_parameters.struct_info["working_directory"] = str(tmp_path / "workdir")
    workflow_service.run(
        run_settings=SimpleNamespace(working_path=tmp_path),
        workflow_parameters=workflow_parameters,
        client=None,
    )

    assert artifacts.closed is True
    assert [name for name, *_ in calls] == [
        "load_structure",
        "select_points",
        "prepare_reciprocal",
        "amplitude",
        "residual_field",
        "postprocessing",
    ]


def test_workflow_service_clears_processed_output_on_fresh_start(tmp_path):
    stale_dir = tmp_path / "workdir" / "processed_point_data"
    stale_dir.mkdir(parents=True)
    stale_file = stale_dir / "stale.hdf5"
    stale_file.write_text("stale", encoding="utf-8")

    class FakeStructureService:
        def load(self, workflow_parameters, working_path):
            return SimpleNamespace(supercell=np.array([4]))

    class FakePointSelectionService:
        def select(self, request):
            return SimpleNamespace(chunk_ids=np.array([0]))

    class FakeArtifacts:
        output_dir = str(stale_dir)

        def close(self):
            pass

    class FakeReciprocalService:
        def prepare(self, workflow_parameters, point_data, supercell, output_dir):
            return FakeArtifacts()

    workflow_service = WorkflowService(
        structure_loading_service=FakeStructureService(),
        point_selection_service=FakePointSelectionService(),
        reciprocal_space_service=FakeReciprocalService(),
        amplitude_service=SimpleNamespace(execute=lambda **kwargs: {}),
        residual_field_service=SimpleNamespace(execute=lambda **kwargs: {}),
        postprocessing_service=SimpleNamespace(execute=lambda **kwargs: None),
    )
    workflow_parameters = _build_workflow_parameters()
    workflow_parameters.struct_info["working_directory"] = str(tmp_path / "workdir")
    workflow_parameters.rspace_info["fresh_start"] = True

    workflow_service.run(
        run_settings=SimpleNamespace(working_path=tmp_path),
        workflow_parameters=workflow_parameters,
        client=None,
    )

    assert stale_dir.exists()
    assert stale_file.exists() is False


def test_parameter_loading_service_returns_form_factor_selection():
    selection = ParameterLoadingService().resolve_form_factor_settings(
        WorkflowParameters(
            schema_version=1,
            struct_info={},
            peak_info={},
            rspace_info={},
            runtime_info={"form_factor": {"family": "xray", "calculator": "lobato"}},
        )
    )
    assert selection.family == "xray"
    assert selection.calculator == "lobato"


def test_form_factor_registry_rejects_unknown_family():
    registry = FormFactorRegistry()
    try:
        registry.create_calculator(SimpleNamespace(family="unknown", calculator="default"))
    except ValueError as exc:
        assert "Unsupported experiment type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown form factor family")


def test_create_db_manager_for_thread_opens_separate_connection(tmp_path):
    db_path = tmp_path / "threaded.db"
    manager1 = DatabaseManager(str(db_path), dimension=1)
    manager2 = create_db_manager_for_thread(str(db_path), dimension=1)
    try:
        assert manager1.connection is not manager2.connection
    finally:
        manager1.close()
        manager2.close()
