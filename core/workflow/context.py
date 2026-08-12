"""
core/workflow/context.py

Typed Protocol for the run-artifacts bag that flows through every pipeline
stage.  This is a *static-typing-only* contract: it documents the attributes
that stages consume without changing what is actually passed at runtime
(currently always a ``ReciprocalSpaceArtifacts`` instance built by
``ReciprocalSpacePreparationService.prepare``).

Import notes
------------
This module is intentionally a *leaf*: it imports only from the standard
library (``typing``) so that it can never introduce a circular import.
"""
from __future__ import annotations

from typing import Any, Protocol


class RunArtifacts(Protocol):
    """Structural type for the artifacts object passed to every pipeline stage.

    All attributes listed here are always present on the concrete
    ``ReciprocalSpaceArtifacts`` dataclass (``core.models``).  Annotating
    stage signatures with ``RunArtifacts`` instead of bare ``Any`` makes the
    expected duck-typed interface explicit to static type-checkers without
    adding any runtime cost.

    Do NOT add ``@runtime_checkable`` here: no ``isinstance(artifacts, RunArtifacts)``
    check is performed at runtime, so the overhead is unnecessary.
    """

    # Path to the directory where output files are written.
    output_dir: str

    # Database manager; always present but its concrete type varies
    # (DatabaseManager or ManifestOnlyDatabaseManager).
    db_manager: Any

    # HDF5 saver used to write point-data results.
    saver: Any

    # Processor that accumulates per-point coordinates / amplitudes.
    point_data_processor: Any

    def close(self) -> None:
        """Release any resources held by the artifacts (e.g. close the DB)."""
        ...


__all__ = ["RunArtifacts"]
