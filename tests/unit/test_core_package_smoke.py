import importlib
from pathlib import Path

import pytest


MODULES = sorted(
    ".".join(path.with_suffix("").parts)
    for path in Path("core").rglob("*.py")
    if "__pycache__" not in path.parts
)
SCIENTIFIC_STAGE_PACKAGES = [
    "core.config",
    "core.structure",
    "core.patch_centers",
    "core.qspace",
    "core.scattering",
    "core.residual_field",
    "core.decoding",
    "core.storage",
    "core.runtime",
    "core.adapters",
    "core.workflow",
]


@pytest.mark.parametrize("module_name", SCIENTIFIC_STAGE_PACKAGES)
def test_import_scientific_stage_packages(module_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MOSAIC_NUFFT_CPU_ONLY", "1")
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", MODULES)
def test_import_all_core_modules(module_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MOSAIC_NUFFT_CPU_ONLY", "1")
    importlib.import_module(module_name)
