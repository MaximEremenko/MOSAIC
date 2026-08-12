"""Compatibility shim.

This module was relocated to :mod:`core.workflow.publisher`: publishing
a run as a public compatibility projection is post-stage *orchestration* and
depends UP on the scattering/residual/decoding ``commit`` modules, which is not
allowed for the device-independent ``core.storage`` primitives layer. The
implementation now lives under ``core.workflow``; this shim preserves the
``core.storage.publisher`` import path.
"""
from __future__ import annotations

from core.workflow.publisher import *  # noqa: F401,F403
from core.workflow.publisher import (
    PUBLIC_MANIFEST_FILENAME,
    PublicPublishError,
    public_manifest_path,
    publish_run,
    validate_public_manifest_files,
)

__all__ = [
    "PUBLIC_MANIFEST_FILENAME",
    "PublicPublishError",
    "public_manifest_path",
    "publish_run",
    "validate_public_manifest_files",
]
