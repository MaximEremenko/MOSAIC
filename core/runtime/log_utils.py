"""Logging utilities for human-friendly output."""

from __future__ import annotations

from pathlib import Path


def short_path(p, tail: int = 3) -> str:
    """Return the last *tail* components of a path, or '<none>' if falsy."""
    if not p:
        return "<none>"
    parts = Path(p).parts
    return str(Path(*parts[-tail:])) if len(parts) >= tail else str(p)
