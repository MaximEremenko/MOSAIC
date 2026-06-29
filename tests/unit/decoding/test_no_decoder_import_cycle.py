"""Guard against the decoder import cycle returning.

``displacement_service`` used to import from ``decoder_service`` while
``decoder_service.build_decoder_training_payload`` reached back into
``displacement_service`` via a *function-local* import to dodge the resulting
circular import. The shared piece (``prepare_displacement_decoder_inputs`` and
the key/spec/feature-building helpers it needs) now lives in
``displacement_inputs`` so neither service imports the other.

These tests fail if anyone reintroduces the cycle, e.g. by moving
``prepare_displacement_decoder_inputs`` back into ``displacement_service`` and
re-adding a function-local fallback import.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import core.decoding.decoder_service as decoder_service
import core.decoding.displacement_inputs as displacement_inputs
import core.decoding.displacement_service as displacement_service


def test_all_decoding_modules_import_cleanly_at_top_level():
    """Importing every decoding module fresh must not trip a circular import."""
    module_names = [
        "core.decoding.displacement_inputs",
        "core.decoding.decoder_service",
        "core.decoding.displacement_service",
        "core.decoding.processor",
    ]
    for name in module_names:
        module = importlib.import_module(name)
        importlib.reload(module)


def test_decoder_service_does_not_import_displacement_service():
    """The back-edge that created the cycle must stay gone.

    ``displacement_service`` may still import from ``decoder_service`` (that edge
    is acyclic: ``displacement_inputs`` <- ``decoder_service`` <-
    ``displacement_service``). What must never return is ``decoder_service``
    importing from ``displacement_service`` -- that was the cycle.
    """
    decoder_src = Path(decoder_service.__file__).read_text(encoding="utf-8")

    assert "core.decoding.displacement_service" not in decoder_src


def test_no_function_local_fallback_import_remains():
    """No function in either service hides a deferred import of the other."""
    for module in (decoder_service, displacement_service):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.ImportFrom) and inner.module:
                    assert inner.module not in {
                        "core.decoding.decoder_service",
                        "core.decoding.displacement_service",
                    }, (
                        f"Function-local import of {inner.module} found in "
                        f"{module.__name__}.{node.name}; the cycle is back."
                    )


def test_shared_module_does_not_import_either_service():
    """``displacement_inputs`` must stay below both services in the graph."""
    source = Path(displacement_inputs.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_service" not in source
    assert "core.decoding.displacement_service" not in source


def test_public_reexports_remain_importable():
    """Existing import paths other modules rely on must keep working."""
    from core.decoding.displacement_inputs import (  # noqa: F401
        prepare_displacement_decoder_inputs,
    )
    from core.decoding.displacement_service import (  # noqa: F401
        compute_and_save_displacements,
        prepare_displacement_decoder_inputs as svc_prepare,
    )
    from core.decoding.decoder_service import (  # noqa: F401
        DisplacementDecoderKey,
        DisplacementDecoderSourceService,
        DisplacementPatchSpec,
        build_decoder_training_payload,
        build_displacement_decoder_key,
        ensure_decoder,
        validate_global_displacement_patch_specs,
    )

    # The service re-export and the canonical definition are the same object.
    assert svc_prepare is prepare_displacement_decoder_inputs


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
