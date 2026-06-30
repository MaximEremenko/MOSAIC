"""W2.2 relocation guard: workflow modules and storage compatibility shims.

``publisher`` and ``cleanup`` were post-stage orchestration misfiled under
``core/storage`` (they import UP into the science-stage ``commit`` modules). They
now live under ``core/workflow`` (workflow is allowed to depend on stages), with
thin re-export shims left at the historical ``core/storage`` paths so existing
import paths keep working. This test asserts both the new home and the shim
re-export the SAME objects.

``run_state_cache`` is deliberately NOT relocated: it is STAGE-CONSUMED
(``core/scattering/execution.py`` and ``core/residual_field/execution.py`` import
it), and ``core/workflow/__init__`` eagerly builds the stages, so moving it under
``core/workflow`` would form a stage -> workflow -> stage import cycle. It stays
the real module under ``core/storage/run_state_cache.py``.
"""
from __future__ import annotations

import core.storage.cleanup as storage_cleanup
import core.storage.publisher as storage_publisher
import core.workflow.cleanup as workflow_cleanup
import core.workflow.publisher as workflow_publisher


def test_publisher_shim_reexports_workflow_objects():
    assert storage_publisher.publish_run is workflow_publisher.publish_run
    assert storage_publisher.public_manifest_path is workflow_publisher.public_manifest_path
    assert (
        storage_publisher.validate_public_manifest_files
        is workflow_publisher.validate_public_manifest_files
    )
    assert storage_publisher.PublicPublishError is workflow_publisher.PublicPublishError


def test_cleanup_shim_reexports_workflow_objects():
    # Patch target used by tests/unit/test_entrypoints.py must resolve to the
    # same callable the workflow implementation exposes.
    assert storage_cleanup.cleanup_run_artifacts is workflow_cleanup.cleanup_run_artifacts
    assert storage_cleanup.CleanupReport is workflow_cleanup.CleanupReport
