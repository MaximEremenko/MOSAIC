"""Relocation guard: workflow modules and storage compatibility shims.

``publisher`` and ``cleanup`` were post-stage orchestration misfiled under
``core/storage`` (they import UP into the science-stage ``commit`` modules). They
now live under ``core/workflow`` (workflow is allowed to depend on stages), with
thin re-export shims left at the historical ``core/storage`` paths so existing
import paths keep working. This test asserts both the new home and the shim
re-export the SAME objects.

``run_state_cache`` joined them: it imports UP into all three stage ``commit``
modules, so leaving it under ``core/storage`` kept that layer from being a leaf.
The earlier objection — that stage-consumed code cannot live under
``core/workflow`` because the package ``__init__`` eagerly builds the stages —
no longer holds: that ``__init__`` is PEP 562 lazy, so a stage importing
``core.workflow.run_state_cache`` pulls in nothing but the submodule itself
(asserted below, since a regression there would resurface as an import cycle).
"""
from __future__ import annotations

import core.storage.cleanup as storage_cleanup
import core.storage.publisher as storage_publisher
import core.storage.run_state_cache as storage_run_state_cache
import core.workflow.cleanup as workflow_cleanup
import core.workflow.publisher as workflow_publisher
import core.workflow.run_state_cache as workflow_run_state_cache


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


def test_run_state_cache_shim_reexports_workflow_objects():
    assert (
        storage_run_state_cache.scan_run_state
        is workflow_run_state_cache.scan_run_state
    )
    assert (
        storage_run_state_cache.rebuild_sqlite_cache_from_manifests
        is workflow_run_state_cache.rebuild_sqlite_cache_from_manifests
    )
    assert (
        storage_run_state_cache.RunStateSnapshot
        is workflow_run_state_cache.RunStateSnapshot
    )


def test_workflow_package_import_stays_lazy():
    """The relocation is only safe while ``core.workflow.__init__`` imports no
    stage at module level — an eager import there would make every stage that
    reads run state cycle back through workflow."""
    import ast
    import pathlib

    source = pathlib.Path("core/workflow/__init__.py").read_text()
    tree = ast.parse(source)
    module_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in module_level:
        names = (
            [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else [alias.name for alias in node.names]
        )
        for name in names:
            assert not name.startswith(("core.", ".")), (
                f"core.workflow.__init__ eagerly imports {name!r}; keep stage "
                "imports inside __getattr__ or run_state_cache must move back"
            )
