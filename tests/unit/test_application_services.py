from types import SimpleNamespace

import numpy as np

from core.application.amplitude import AmplitudeExecutionService
from core.application.coefficients import CoefficientCenteringService
from core.application.form_factor_registry import FormFactorRegistry
from core.application.intervals import IntervalReconstructionService
from core.application.masking import MaskStrategyService
from core.application.parameters import ParameterLoadingService
from core.application.postprocessing import PostprocessingService
from core.domain.models import ReciprocalSpaceArtifacts, StructureData, WorkflowParameters
from core.managers.database_manager import DatabaseManager, create_db_manager_for_thread


def _build_workflow_parameters() -> WorkflowParameters:
    return WorkflowParameters(
        schema_version=1,
        struct_info={"dimension": 1},
        peak_info={"mask_equation": "h > 0"},
        rspace_info={"mode": "displacement", "use_coeff": True},
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


def test_coefficient_centering_service_centers_by_refnumber():
    coeff = np.array([1.0, 2.0, 10.0, 14.0])
    refnumbers = np.array([1, 1, 2, 2])
    centered = CoefficientCenteringService().center(coeff, refnumbers, "refnumbers")
    np.testing.assert_allclose(centered, np.array([-0.5, 0.5, -2.0, 2.0]))


def test_mask_strategy_service_builds_equation_strategy():
    strategy = MaskStrategyService().build(1, {"mask_equation": "h > 0"})
    mask = strategy.generate_mask(np.array([[1.0], [-1.0]]))
    assert mask.tolist() == [True, False]


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
        context = AmplitudeExecutionService()._build_context(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
        )
        assert context.dimension == 1
        assert context.form_factor_selection.family == "neutron"
        assert context.form_factor_selection.calculator == "default"
        np.testing.assert_allclose(context.centered_coefficients, np.array([2.0]))
        assert len(context.unsaved_interval_chunks) == 1
        assert len(context.intervals) == 1
    finally:
        artifacts.db_manager.close()


def test_postprocessing_service_builds_typed_context(tmp_path):
    artifacts = _build_artifacts(tmp_path)
    try:
        context = PostprocessingService()._build_context(
            workflow_parameters=_build_workflow_parameters(),
            structure=_build_structure(),
            artifacts=artifacts,
        )
        assert context.postprocessing_mode == "displacement"
        assert context.artifacts.output_dir == str(tmp_path)
    finally:
        artifacts.db_manager.close()


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
