"""Verify that the decoding import cycle stays closed.

Previously, ``decoder_training`` used a function-local
``from core.decoding.decode_chunk import PointDataPostprocessingProcessor``
to dodge the cycle:

    decoder_training  <-- displacement
         |
         v
      decode_chunk --> displacement --> decoder_training  (CYCLE)

After the fix ``apply_decoder``, ``apply_decoder_family``, ``ensure_decoder``,
and ``_stack_features_into_columns`` were moved to the leaf module
``displacement_inputs``.  ``displacement`` now imports these from
``displacement_inputs`` instead of ``decoder_training``, which breaks the edge
``displacement --> decoder_training`` and allows ``decoder_training`` to
import ``decode_chunk`` at module scope without a cycle.

These tests:
1. Verify that all four modules import cleanly (including in a fresh
   sub-process to catch deferred errors).
2. Verify that ``decoder_training`` does NOT contain any function-local import
   of ``decode_chunk``, ``displacement``, or ``decoder_training`` itself.
3. Verify that ``displacement_inputs`` remains a true leaf — it must not import
   either service module.
4. Verify that ``PointDataPostprocessingProcessor`` is imported at module
   scope in ``decoder_training`` (not hidden in a function body).
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Match a reference to the ``displacement`` module but NOT the unrelated
# ``displacement_inputs`` leaf module that both services legitimately import.
_DISPLACEMENT_MODULE_RE = re.compile(r"core\.decoding\.displacement(?!_inputs)")

import core.decoding.decoder_training as decoder_service_mod
import core.decoding.displacement_inputs as displacement_inputs_mod
import core.decoding.displacement as displacement_service_mod
import core.decoding.decode_chunk as processor_mod


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
        "import core.decoding.decoder_training; "
        "import core.decoding.displacement; "
        "import core.decoding.decode_chunk; "
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
    """Reloading decoder_training must not raise."""
    importlib.reload(decoder_service_mod)


def test_displacement_inputs_is_a_leaf_module():
    """displacement_inputs must not import either service module."""
    source = Path(displacement_inputs_mod.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_training" not in source, (
        "displacement_inputs imports decoder_training — leaf property violated."
    )
    assert _DISPLACEMENT_MODULE_RE.search(source) is None, (
        "displacement_inputs imports displacement — leaf property violated."
    )
    assert "core.decoding.decode_chunk" not in source, (
        "displacement_inputs imports decode_chunk — leaf property violated."
    )


def test_decoder_service_does_not_import_displacement_service():
    """The back-edge that created the original cycle must not return."""
    source = Path(decoder_service_mod.__file__).read_text(encoding="utf-8")
    assert _DISPLACEMENT_MODULE_RE.search(source) is None


# ---------------------------------------------------------------------------
# No function-local import band-aids
# ---------------------------------------------------------------------------

def test_no_function_local_import_of_processor_in_decoder_service():
    """``decoder_training`` must import ``decode_chunk`` at module scope only.

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
                    "core.decoding.decode_chunk",
                    "core.decoding.displacement",
                    "core.decoding.decoder_training",
                }, (
                    f"Function-local import of {inner.module!r} found in "
                    f"decoder_training.{node.name} — cycle band-aid is back."
                )


def test_no_function_local_import_of_decoder_service_in_displacement_service():
    """``displacement`` must not hide a deferred import of
    ``decoder_training`` (or any sibling service) inside a function body."""
    source = Path(displacement_service_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                assert inner.module not in {
                    "core.decoding.decoder_training",
                    "core.decoding.decode_chunk",
                }, (
                    f"Function-local import of {inner.module!r} found in "
                    f"displacement.{node.name} — cycle band-aid is back."
                )


# ---------------------------------------------------------------------------
# Public re-export compatibility
# ---------------------------------------------------------------------------

def test_processor_importable_at_module_scope_from_decoder_service():
    """``PointDataPostprocessingProcessor`` must be importable from
    ``decoder_training``'s module namespace (i.e. it is imported at the top of
    the file, not only inside a function)."""
    assert hasattr(decoder_service_mod, "PointDataPostprocessingProcessor"), (
        "PointDataPostprocessingProcessor not found in decoder_training namespace; "
        "it may still be hidden inside a function body."
    )
    from core.decoding.decode_chunk import PointDataPostprocessingProcessor
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
    """displacement must not import from decoder_training at all
    (that was the edge that caused the original cycle)."""
    source = Path(displacement_service_mod.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_training" not in source, (
        "displacement still imports from decoder_training — cycle not fully broken."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
