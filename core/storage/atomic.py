from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping


class PathContainmentError(ValueError):
    """Raised when an artifact path escapes the configured output directory."""


def _has_windows_drive(path_text: str) -> bool:
    return len(path_text) >= 2 and path_text[1] == ":" and path_text[0].isalpha()


def assert_path_contained(path: str | Path, *, output_dir: str | Path) -> Path:
    """Resolve a low-level storage path and verify it stays inside output_dir.

    This helper intentionally accepts already-built absolute Path objects for
    internal storage builders. Manifest/public paths must use
    assert_relative_path_contained() instead.
    """
    output_root = Path(output_dir).resolve()
    path_text = os.fspath(path)
    if not path_text:
        raise PathContainmentError("Path must not be empty.")
    if _has_windows_drive(path_text):
        raise PathContainmentError(f"Windows drive roots are not allowed: {path_text!r}")

    candidate = Path(path)
    if any(part in {"", ".."} for part in candidate.parts):
        raise PathContainmentError(f"Path must not contain empty or parent segments: {path_text!r}")
    if not candidate.is_absolute():
        candidate = output_root / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        resolved = candidate.parent.resolve(strict=False) / candidate.name

    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise PathContainmentError(
            f"Path {resolved!s} is outside output directory {output_root!s}."
        ) from exc
    return resolved


def _reject_invalid_relative_path_text(path_text: str) -> None:
    if not path_text:
        raise PathContainmentError("Path must not be empty.")
    if "\\" in path_text:
        raise PathContainmentError(f"Backslash path separators are not allowed: {path_text!r}")
    if _has_windows_drive(path_text):
        raise PathContainmentError(f"Windows drive roots are not allowed: {path_text!r}")
    if Path(path_text).is_absolute():
        raise PathContainmentError(f"Path must be relative: {path_text!r}")
    parts = path_text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PathContainmentError(f"Path contains an invalid segment: {path_text!r}")


def assert_relative_path_contained(path: str | Path, *, output_dir: str | Path) -> Path:
    """Validate a manifest/public path as POSIX-relative and contained."""
    path_text = os.fspath(path)
    _reject_invalid_relative_path_text(path_text)
    return assert_path_contained(Path(path_text), output_dir=output_dir)


def fsync_path(path: str | Path) -> None:
    target = Path(path)
    try:
        fd = os.open(target, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_parent(path: str | Path) -> None:
    parent = Path(path).parent
    try:
        fd = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def temp_sibling_path(path: str | Path, *, suffix: str = ".tmp") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    os.close(fd)
    return Path(temp_path)


def atomic_write_json(
    path: str | Path,
    payload: Mapping,
    *,
    output_dir: str | Path,
    validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    target = assert_path_contained(path, output_dir=output_dir)
    if validator is not None:
        validator(payload)
    temp_path = temp_sibling_path(target, suffix=".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        fsync_parent(target)
        with target.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if validator is not None:
            if not isinstance(loaded, Mapping):
                raise ValueError("Reopened JSON payload must be an object.")
            validator(loaded)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


__all__ = [
    "PathContainmentError",
    "assert_relative_path_contained",
    "assert_path_contained",
    "atomic_write_json",
    "fsync_parent",
    "fsync_path",
    "temp_sibling_path",
]
