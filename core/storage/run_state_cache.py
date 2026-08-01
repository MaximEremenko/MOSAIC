"""Compatibility shim.

This module was relocated to :mod:`core.workflow.run_state_cache`: rebuilding
run state from manifests is post-stage *orchestration* and depends UP on the
scattering/residual/decoding ``commit`` modules, which the device-independent
``core.storage`` primitives layer may not do. Relocating it makes
``core.storage`` a true leaf.

(An earlier attempt kept it here on the belief that ``core.workflow.__init__``
eagerly built the stages, which would have made stage -> workflow -> stage a
cycle; that ``__init__`` is PEP 562 lazy, so importing this submodule from a
stage pulls in nothing but the module itself.)

Patch module internals through the real module — the implementation resolves
its own globals there, not through this shim.
"""
from __future__ import annotations

from core.workflow.run_state_cache import *  # noqa: F401,F403
from core.workflow.run_state_cache import (  # noqa: F401
    ChunkState,
    RunStateSnapshot,
    StageState,
    _committed_streaming_residual_credits,
    pending_residual_interval_chunks,
    pending_scattering_interval_chunks,
    rebuild_sqlite_cache_from_manifests,
    scan_run_state,
)
