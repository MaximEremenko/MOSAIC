"""Guard against the decoder import cycle returning.

``displacement`` used to import from ``decoder_training`` while
``decoder_training.build_decoder_training_payload`` reached back into
``displacement`` via a *function-local* import to dodge the resulting
circular import. The shared piece (``prepare_displacement_decoder_inputs`` and
the key/spec/feature-building helpers it needs) now lives in
``displacement_inputs`` so neither service imports the other.

These tests fail if anyone reintroduces the cycle, e.g. by moving
``prepare_displacement_decoder_inputs`` back into ``displacement`` and
re-adding a function-local fallback import.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

# Match a reference to the ``displacement`` module but NOT the unrelated
# ``displacement_inputs`` leaf module that both services legitimately import.
_DISPLACEMENT_MODULE_RE = re.compile(r"core\.decoding\.displacement(?!_inputs)")

import core.decoding.decoder_training as decoder_service
import core.decoding.displacement_inputs as displacement_inputs
import core.decoding.displacement as displacement_service


def test_all_decoding_modules_import_cleanly_at_top_level():
    """Importing every decoding module fresh must not trip a circular import."""
    module_names = [
        "core.decoding.displacement_inputs",
        "core.decoding.decoder_training",
        "core.decoding.displacement",
        "core.decoding.decode_chunk",
    ]
    for name in module_names:
        module = importlib.import_module(name)
        importlib.reload(module)


def test_decoder_service_does_not_import_displacement_service():
    """The back-edge that created the cycle must stay gone.

    ``displacement`` may still import from ``decoder_training`` (that edge
    is acyclic: ``displacement_inputs`` <- ``decoder_training`` <-
    ``displacement``). What must never return is ``decoder_training``
    importing from ``displacement`` -- that was the cycle.
    """
    decoder_src = Path(decoder_service.__file__).read_text(encoding="utf-8")

    assert _DISPLACEMENT_MODULE_RE.search(decoder_src) is None


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
                        "core.decoding.decoder_training",
                        "core.decoding.displacement",
                    }, (
                        f"Function-local import of {inner.module} found in "
                        f"{module.__name__}.{node.name}; the cycle is back."
                    )


def test_shared_module_does_not_import_either_service():
    """``displacement_inputs`` must stay below both services in the graph."""
    source = Path(displacement_inputs.__file__).read_text(encoding="utf-8")
    assert "core.decoding.decoder_training" not in source
    assert _DISPLACEMENT_MODULE_RE.search(source) is None


def test_public_reexports_remain_importable():
    """Existing import paths other modules rely on must keep working."""
    from core.decoding.displacement_inputs import (  # noqa: F401
        prepare_displacement_decoder_inputs,
    )
    from core.decoding.displacement import (  # noqa: F401
        compute_and_save_displacements,
        prepare_displacement_decoder_inputs as svc_prepare,
    )
    from core.decoding.decoder_training import (  # noqa: F401
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
