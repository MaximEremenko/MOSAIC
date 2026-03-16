import json
import shutil
from pathlib import Path

import pytest

from core.config import ParameterLoadingService
from core.workflow import build_default_workflow_service


ROOT = Path(__file__).resolve().parents[2]


def _prepare_fixture(tmp_path: Path, name: str, input_payload: dict, files: dict[str, Path]):
    fixture_dir = tmp_path / name
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for target_name, source_path in files.items():
        shutil.copy2(source_path, fixture_dir / target_name)
    input_path = fixture_dir / "input_parameters.json"
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")
    run_path = fixture_dir / "run_parameters.json"
    run_path.write_text(
        json.dumps({"input_parameters_path": str(input_path)}),
        encoding="utf-8",
    )
    return run_path


def _run_until_preparation(run_path: Path, monkeypatch):
    parameter_loading_service = ParameterLoadingService()
    run_settings, workflow_parameters = parameter_loading_service.load(str(run_path))
    workflow_service = build_default_workflow_service()
    monkeypatch.setattr(workflow_service.scattering_stage, "execute", lambda **kwargs: {})
    monkeypatch.setattr(workflow_service.residual_field_stage, "execute", lambda **kwargs: {})
    monkeypatch.setattr(workflow_service.decoding_stage, "execute", lambda **kwargs: None)
    workflow_service.run(run_settings, workflow_parameters, client=None)
    output_dir = Path(workflow_parameters.struct_info.working_directory) / "processed_point_data"
    return workflow_parameters, output_dir


def test_workflow_services_prepare_1d_fixture(tmp_path, monkeypatch):
    input_payload = {
        "schema_version": 2,
        "paths": {
            "config_root": ".",
            "structure_file": "slice_displacement_1d.f1d",
            "output_directory": "./output_displacement",
        },
        "structure": {"dimension": 1},
        "reciprocal_space": {
            "intervals": [{"limit": [1.0], "subvolume_step": [1.0]}],
            "mask": {"equation": "((Mod(h,1.0) - 0.5)**2 <= (0.25**2))"},
        },
        "processing": {
            "mode": "displacement",
            "method": "from_average",
            "num_chunks": 1,
            "fresh_start": True,
            "run_postprocessing": False,
            "points": [
                {
                    "selector": {"element": "El1", "reference_number": 1},
                    "window": {
                        "dist_from_atom_center": [0.1],
                        "step_in_angstrom": [0.05],
                    },
                }
            ],
        },
        "runtime": {
            "dask": {
                "backend": "synchronous",
                "max_workers": 1,
                "threads_per_worker": 1,
                "processes": False,
                "worker_wait_timeout": "30s",
            }
        },
    }
    run_path = _prepare_fixture(
        tmp_path,
        "fixture_1d",
        input_payload,
        {
            "slice_displacement_1d.f1d": ROOT
            / "examples"
            / "config_1D"
            / "displacement"
            / "slice_displacement_1d.f1d",
        },
    )
    _, output_dir = _run_until_preparation(run_path, monkeypatch)
    assert output_dir.exists()
    assert (output_dir / "point_reciprocal_space_associations.db").exists()
    assert (output_dir / "point_data.hdf5").exists()


@pytest.mark.parametrize(
    ("fixture_name", "input_file", "structure_file"),
    [
        (
            "fixture_2d",
            ROOT / "examples" / "config_2D" / "displacement" / "input_parameters.json",
            ROOT / "examples" / "config_2D" / "displacement" / "sample_2d.f2d",
        ),
        (
            "fixture_3d",
            ROOT / "examples" / "config_3D" / "chemical_ordering" / "input_parameters.json",
            ROOT / "examples" / "config_3D" / "chemical_ordering" / "LiFeO2.rmc6f",
        ),
    ],
)
def test_workflow_services_prepare_existing_fixtures(
    tmp_path,
    monkeypatch,
    fixture_name,
    input_file,
    structure_file,
):
    input_payload = json.loads(input_file.read_text(encoding="utf-8"))
    structure_name = structure_file.name
    if "paths" in input_payload:
        input_payload["paths"]["structure_file"] = structure_name
    else:
        input_payload["structInfo"]["filename"] = structure_name

    run_path = _prepare_fixture(
        tmp_path,
        fixture_name,
        input_payload,
        {structure_name: structure_file},
    )
    workflow_parameters, output_dir = _run_until_preparation(run_path, monkeypatch)
    assert output_dir.exists()
    assert (output_dir / "point_reciprocal_space_associations.db").exists()
    assert (output_dir / "point_data.hdf5").exists()
    assert workflow_parameters.struct_info.working_directory in str(output_dir.parent)
