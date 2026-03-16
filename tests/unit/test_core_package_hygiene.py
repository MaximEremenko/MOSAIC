from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_NON_CODE = {
    "core/README.md",
    "core/environment.yml",
}
ALLOWED_TOP_LEVEL_DIRS = {
    "adapters",
    "config",
    "decoding",
    "entrypoints",
    "patch_centers",
    "qspace",
    "residual_field",
    "runtime",
    "scattering",
    "storage",
    "structure",
    "workflow",
}
EXPECTED_UNIT_STAGE_DIRS = {
    "config",
    "structure",
    "patch_centers",
    "qspace",
    "scattering",
    "residual_field",
    "decoding",
    "storage",
    "runtime",
}
ALLOWED_CROSS_CUTTING_UNIT_TESTS = {
    "test_application_services.py",
    "test_core_package_hygiene.py",
    "test_core_package_smoke.py",
    "test_entrypoints.py",
    "test_import_style_guard.py",
    "test_no_print_guard.py",
    "test_stage_contracts.py",
    "test_stage_manifest_semantics.py",
}


def test_core_has_no_tracked_runtime_or_artifact_files():
    disallowed = []
    for path in ROOT.joinpath("core").rglob("*"):
        if path.is_dir():
            rel = str(path.relative_to(ROOT))
            if rel in {"core/dask_logs", "core/entrypoints/dask_logs"}:
                disallowed.append(rel)
            continue
        rel = str(path.relative_to(ROOT))
        if rel in ALLOWED_NON_CODE:
            continue
        if rel.endswith("run_parameters.json"):
            disallowed.append(rel)
        if rel.endswith("logging.conf"):
            disallowed.append(rel)
        if rel.endswith("dependencies.json") or rel.endswith("dependencies.txt"):
            disallowed.append(rel)
        if rel.startswith("core/files/") or rel.startswith("core/tests/"):
            disallowed.append(rel)
    assert disallowed == []


def test_core_has_only_scientific_stage_top_level_packages():
    core_root = ROOT / "core"
    top_level_dirs = {
        path.name
        for path in core_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert top_level_dirs == ALLOWED_TOP_LEVEL_DIRS


def test_unit_tests_are_grouped_by_stage_or_kept_cross_cutting():
    unit_root = ROOT / "tests" / "unit"
    stage_dirs = {
        path.name
        for path in unit_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert EXPECTED_UNIT_STAGE_DIRS.issubset(stage_dirs)

    flat_tests = {
        path.name
        for path in unit_root.iterdir()
        if path.is_file() and path.name.startswith("test_") and path.suffix == ".py"
    }
    assert flat_tests == ALLOWED_CROSS_CUTTING_UNIT_TESTS
