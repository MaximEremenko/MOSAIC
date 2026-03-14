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
