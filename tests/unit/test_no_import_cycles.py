"""
tests/unit/test_no_import_cycles.py

Import smoke test: discovers every module under core/ and, for each one,
asserts that importing it raises no ImportError.  Discovery happens via
pkgutil.walk_packages, and each module name becomes a separate
pytest.parametrize case.  Every case imports its module with
importlib.import_module in the SAME process as the rest of the test
session (there is no subprocess and no fresh-interpreter isolation), so
modules already pulled in by an earlier case stay cached in sys.modules.

What this guards against: the common runtime failure mode of a circular
import, where a module raises ImportError at import time.  Catching that
ImportError is the whole assertion.

What this does NOT do:
  * It does NOT prove the import graph is acyclic.  A cycle that happens
    to resolve without raising ImportError (e.g. because the needed names
    are already bound by the time they are accessed) will pass here.
  * It does NOT see TYPE_CHECKING-guarded imports, which are never
    executed at runtime and so are invisible to importlib.
  * It does NOT forbid lazy / function-local imports that are
    intentionally deferred (e.g. large optional dependencies); only
    top-level import failures are observed.
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

    The import runs in the current process via importlib.import_module,
    reusing whatever sys.modules state earlier parametrized cases left
    behind (normal Python caching — there is no subprocess isolation).
    The only property asserted is that the import does not raise
    ImportError, which is how a runtime circular import typically fails.
    This does not establish acyclicity and does not exercise
    TYPE_CHECKING-only imports.
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
