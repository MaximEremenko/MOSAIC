"""Compatibility shim.

This module was relocated to :mod:`core.workflow.cleanup` (W2.2): manifest-gated
cleanup is post-stage *orchestration* and depends UP on the scattering/residual
``commit`` modules, which is not allowed for the device-independent
``core.storage`` primitives layer. The implementation now lives under
``core.workflow``; this shim preserves the historical ``core.storage.cleanup``
import path (including the ``core.storage.cleanup.cleanup_run_artifacts`` patch
target used by tests).
"""
from __future__ import annotations

from core.workflow.cleanup import *  # noqa: F401,F403
from core.workflow.cleanup import (
    CleanupReport,
    cleanup_run_artifacts,
)

__all__ = [
    "CleanupReport",
    "cleanup_run_artifacts",
]
