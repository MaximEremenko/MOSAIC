"""Verify that the decoding import cycle (A3) stays fixed.

Before the fix, ``decoder_service`` used a function-local
``from core.decoding.processor import PointDataPostprocessingProcessor``
to dodge the cycle:

    decoder_service  <-- displacement_service
         |
         v
      processor --> displacement_service --> decoder_service  (CYCLE)

After the fix ``apply_decoder``, ``apply_decoder_family``, ``ensure_decoder``,
and ``_stack_features_into_columns`` were moved to the leaf module
``displacement_inputs``.  ``displacement_service`` now imports these from
``displacement_inputs`` instead of ``decoder_service``, which breaks the edge
``displacement_service --> decoder_service`` and allows ``decoder_service`` to
import ``processor`` at module scope without a cycle.

These tests:
1. Verify that all four modules import cleanly (including in a fresh
   sub-process to catch deferred errors).
2. Verify that ``decoder_service`` does NOT contain any function-local import
   of ``processor``, ``displacement_service``, or ``decoder_service`` itself.
3. Verify that ``displacement_inputs`` remains a true leaf — it must not import
   either service module.
4. Verify that ``PointDataPostprocessingProcessor`` is imported at module
   scope in ``decoder_service`` (not hidden in a function body).
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import core.decoding.decoder_service as decoder_service_mod
import core.decoding.displacement_inputs as displacement_inputs_mod
import core.decoding.displacement_service as displacement_service_mod
import core.decoding.processor as processor_mod


# ---------------------------------------------------------------------------
# Fresh-process import test
# ---------------------------------------------------------------------------

def test_all_modules_import_in_fresh_subprocess():
    """Import all four modules in an isolated Python process.

    A circular import would raise ``ImportError`` or ``AttributeError`` in the
    child process, causing a non-zero exit code.
    """
    script = (
        "import core.decoding.displacement_inputs; "
        "import core.decoding.decoder_service; "
        "import core.decoding.displacement_service; "
        "import core.decoding.processor; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[3]),  # repo root
    )
    assert result.returncode == 0, (
        f"Fresh-subprocess import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Module-level import tests
# ---------------------------------------------------------------------------

def test_decoder_service_imports_cleanly_at_module_level():
    """Reloading decoder_service must not raise."""
    importlib.reload(decoder_service_mod)


def test_displacement_inputs_is_a_leaf_module():
    """displacement_inputs must not import either service module."""
    source = Path(displacement_inputs_mod.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_service" not in source, (
        "displacement_inputs imports decoder_service — leaf property violated."
    )
    assert "core.decoding.displacement_service" not in source, (
        "displacement_inputs imports displacement_service — leaf property violated."
    )
    assert "core.decoding.processor" not in source, (
        "displacement_inputs imports processor — leaf property violated."
    )


def test_decoder_service_does_not_import_displacement_service():
    """The back-edge that created the original cycle must not return."""
    source = Path(decoder_service_mod.__file__).read_text(encoding="utf-8")
    assert "core.decoding.displacement_service" not in source


# ---------------------------------------------------------------------------
# No function-local import band-aids
# ---------------------------------------------------------------------------

def test_no_function_local_import_of_processor_in_decoder_service():
    """``decoder_service`` must import ``processor`` at module scope only.

    Before the cycle fix, ``PointDataPostprocessingProcessor`` was imported
    inside ``_compute_decoder_cache`` to avoid a circular import.  After the
    fix that function-local import must be gone.
    """
    source = Path(decoder_service_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                assert inner.module not in {
                    "core.decoding.processor",
                    "core.decoding.displacement_service",
                    "core.decoding.decoder_service",
                }, (
                    f"Function-local import of {inner.module!r} found in "
                    f"decoder_service.{node.name} — cycle band-aid is back."
                )


def test_no_function_local_import_of_decoder_service_in_displacement_service():
    """``displacement_service`` must not hide a deferred import of
    ``decoder_service`` (or any sibling service) inside a function body."""
    source = Path(displacement_service_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                assert inner.module not in {
                    "core.decoding.decoder_service",
                    "core.decoding.processor",
                }, (
                    f"Function-local import of {inner.module!r} found in "
                    f"displacement_service.{node.name} — cycle band-aid is back."
                )


# ---------------------------------------------------------------------------
# Public re-export compatibility
# ---------------------------------------------------------------------------

def test_processor_importable_at_module_scope_from_decoder_service():
    """``PointDataPostprocessingProcessor`` must be importable from
    ``decoder_service``'s module namespace (i.e. it is imported at the top of
    the file, not only inside a function)."""
    assert hasattr(decoder_service_mod, "PointDataPostprocessingProcessor"), (
        "PointDataPostprocessingProcessor not found in decoder_service namespace; "
        "it may still be hidden inside a function body."
    )
    from core.decoding.processor import PointDataPostprocessingProcessor
    assert decoder_service_mod.PointDataPostprocessingProcessor is PointDataPostprocessingProcessor


def test_shared_apply_helpers_importable_from_displacement_inputs():
    """apply_decoder, apply_decoder_family, ensure_decoder must live in the
    leaf module so that both services can import them without a cycle."""
    from core.decoding.displacement_inputs import (  # noqa: F401
        apply_decoder,
        apply_decoder_family,
        ensure_decoder,
        _stack_features_into_columns,
    )


def test_displacement_service_no_longer_imports_decoder_service():
    """displacement_service must not import from decoder_service at all
    (that was the edge that caused the original cycle)."""
    source = Path(displacement_service_mod.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_service" not in source, (
        "displacement_service still imports from decoder_service — cycle not fully broken."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
