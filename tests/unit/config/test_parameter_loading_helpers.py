import json
import os
from pathlib import Path

import core.config.service as configuration_service
from core.config import ParameterLoadingService
from core.config.run_files import (
    load_run_settings,
    resolve_input_parameters_path,
)
from core.config.schema import normalize_input_schema


def test_load_run_settings_prefers_examples_when_no_local_file(tmp_path):
    repo_root = tmp_path
    examples_dir = repo_root / "examples"
    examples_dir.mkdir()
    payload = {"input_parameters_path": "./config_2D/input_parameters.json"}
    (examples_dir / "run_parameters.json").write_text(json.dumps(payload), encoding="utf-8")
    run_path, loaded = load_run_settings("run_parameters.json", repo_root=repo_root)
    assert run_path == examples_dir / "run_parameters.json"
    assert loaded == payload


def test_resolve_input_parameters_path_prefers_examples_over_tests(tmp_path):
    run_path = tmp_path / "run_parameters.json"
    run_path.write_text("{}", encoding="utf-8")
    examples_dir = tmp_path / "examples" / "fixture"
    tests_dir = tmp_path / "tests" / "fixture"
    examples_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (examples_dir / "input_parameters.json").write_text("{}", encoding="utf-8")
    (tests_dir / "input_parameters.json").write_text("{}", encoding="utf-8")

    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        input_path, root_path = resolve_input_parameters_path(run_path, {})
    finally:
        os.chdir(old_cwd)

    assert input_path == examples_dir / "input_parameters.json"
    assert root_path == examples_dir


def test_normalize_input_schema_maps_unified_payload():
    payload = {
        "schema_version": 2,
        "paths": {
            "output_directory": "./out",
            "structure_file": "sample.f2d",
        },
        "structure": {"dimension": 2},
        "reciprocal_space": {"intervals": [{"limit": [1.0, 1.0], "subvolume_step": [1.0, 1.0]}]},
        "processing": {"method": "from_average", "num_chunks": 2, "points": []},
        "runtime": {"form_factor": {"family": "neutron", "calculator": "default"}},
    }
    normalized = normalize_input_schema(payload)
    assert normalized["structInfo"]["dimension"] == 2
    assert normalized["structInfo"]["filename"] == "sample.f2d"
    assert normalized["rspace_info"]["method"] == "from_average"
    assert normalized["runtime_info"]["form_factor"]["family"] == "neutron"


def test_parameter_loading_service_finds_examples_run_file_from_core_workdir(
    tmp_path, monkeypatch
):
    repo_root = tmp_path
    (repo_root / "core").mkdir()
    fixture_dir = repo_root / "examples" / "config_2D"
    fixture_dir.mkdir(parents=True)

    input_payload = {
        "structInfo": {
            "dimension": 1,
            "working_directory": "sample_run",
            "filename": "sample_1d.f1d",
        },
        "peakInfo": {},
        "rspace_info": {"method": "full_list", "num_chunks": 1, "points": []},
    }
    (fixture_dir / "input_parameters.json").write_text(
        json.dumps(input_payload), encoding="utf-8"
    )
    (repo_root / "examples" / "run_parameters.json").write_text(
        json.dumps(
            {
                "input_parameters_path": "./config_2D/input_parameters.json",
                "working_path": "./config_2D/",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root / "core")
    monkeypatch.setattr(configuration_service, "resolve_repo_root", lambda: repo_root)

    run_settings, workflow_parameters = ParameterLoadingService().load("run_parameters.json")

    assert run_settings.run_parameters_path == repo_root / "examples" / "run_parameters.json"
    assert run_settings.input_parameters_path == fixture_dir / "input_parameters.json"
    assert workflow_parameters.struct_info["filename"] == str(
        fixture_dir / "sample_1d.f1d"
    )


def test_resolve_repo_root_points_to_repository_root():
    expected = Path(__file__).resolve().parents[3]
    assert configuration_service.resolve_repo_root() == expected
