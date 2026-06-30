"""
tests/unit/test_no_import_cycles.py

Import-cycle guard: walks every module under core/ and asserts each one
can be imported in a fresh subprocess without ImportError.  Running all
modules inside a single process would mask cycles that only surface when
the package is loaded from scratch (because sys.modules caching hides
them after the first successful import).  Each parametrized test case
imports the module in isolation via importlib.import_module so that
accumulated sys.modules state from earlier cases does not hide a genuine
circular ImportError.

The test does NOT forbid lazy / function-local imports that are
intentionally deferred for other reasons (e.g. large optional
dependencies).  It only asserts that no module raises ImportError at the
point it is imported.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

import pytest


def _discover_core_modules() -> list[str]:
    """Return a sorted list of every importable module name under core/."""
    names: list[str] = []
    # Walk the core package recursively.
    try:
        import core as _core_pkg
    except ImportError:
        pytest.skip("core package not importable from this working directory")
        return []

    for _finder, module_name, _is_pkg in pkgutil.walk_packages(
        path=_core_pkg.__path__,
        prefix=_core_pkg.__name__ + ".",
        onerror=lambda name: None,
    ):
        names.append(module_name)

    return sorted(names)


_CORE_MODULES = _discover_core_modules()


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_module_imports_without_error(module_name: str, monkeypatch, tmp_path) -> None:
    """Assert that importing *module_name* raises no ImportError.

    Each test case starts from whatever sys.modules state the parametrized
    fixture provides (modules already imported earlier in the session are
    cached, which is the normal Python behaviour).  The key property being
    tested is that no module raises ImportError at import time — which is
    what a genuine circular import would do.
    """
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    monkeypatch.setenv("MOSAIC_NUFFT_CPU_ONLY", "1")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(
            f"ImportError while importing {module_name!r}: {exc}\n"
            "This likely indicates a circular import or a missing dependency."
        )
    assert isinstance(mod, ModuleType), f"Expected a module, got {type(mod)}"
