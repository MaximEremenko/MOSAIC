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


def serialize_json_payload(payload: Mapping) -> bytes:
    """Serialize a manifest payload to the canonical on-disk byte form.

    This is the single source of truth for the durable JSON encoding used by
    ``atomic_write_json`` (and therefore ``write_manifest``). Any other code path
    that needs to reproduce the exact committed bytes — e.g. a no-overwrite CAS
    commit — MUST route through this helper so the two paths can never drift.

    The encoding is ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
    followed by a single trailing newline, encoded as UTF-8.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def atomic_write_json(
    path: str | Path,
    payload: Mapping,
    *,
    output_dir: str | Path | None = None,
    validator: Callable[[Mapping[str, Any]], None] | None = None,
    indent: int | None = None,
) -> None:
    """Atomically write ``payload`` as JSON (tmp file + fsync + rename).

    ``output_dir`` enables the containment check; callers writing to
    already-validated absolute paths (residual-field manifests) may omit it.
    ``indent=None`` produces the canonical compact ``serialize_json_payload``
    bytes; ``indent=N`` produces ``json.dumps(..., indent=N, sort_keys=True)``
    with no trailing newline — the residual-field manifest byte format, which
    must stay stable because existing manifests are re-read and re-written
    mid-run.
    """
    if output_dir is not None:
        target = assert_path_contained(path, output_dir=output_dir)
    else:
        # temp_sibling_path creates the parent directory.
        target = Path(path)
    if validator is not None:
        validator(payload)
    if indent is None:
        data = serialize_json_payload(payload)
    else:
        data = json.dumps(payload, indent=indent, sort_keys=True).encode("utf-8")
    temp_path = temp_sibling_path(target, suffix=".tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(data)
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


def atomic_create_no_overwrite(target: str | Path, data: bytes) -> bool:
    """Atomically create ``target`` with ``data`` only if it does not yet exist.

    This is a true first-writer-wins compare-and-swap (CAS): the first caller to
    create the file wins; every later caller is told the file already exists and
    leaves the existing bytes untouched. Unlike ``atomic_write_json`` (which uses
    ``os.replace`` = atomic OVERWRITE, with a check-then-write TOCTOU window when
    callers gate it on ``exists()``), this primitive never overwrites and has no
    TOCTOU window.

    Mechanism: ``data`` is written to a unique temp file in the SAME directory as
    ``target`` (so the temp and target share a filesystem, a hard requirement for
    ``os.link``), flushed and ``os.fsync``-ed for durability. We then attempt
    ``os.link(temp, target)``:

    * success  -> this caller is the first writer; returns ``True`` and the parent
      directory is fsync-ed so the new directory entry is durable.
    * ``FileExistsError`` -> another writer already created ``target``; returns
      ``False`` and the existing file is left exactly as-is.

    The temp file is ALWAYS unlinked in a ``finally`` (the hard link keeps the
    committed inode alive after the temp name is removed).

    ``os.link`` is the classic POSIX atomic no-overwrite CAS and is the intended
    path for the Linux cluster and WSL. WINDOWS CAVEAT: hard-link / replace
    semantics differ on Windows (and ``os.link`` requires NTFS + privileges), so
    this primitive is not the supported durable path there; the ``os.replace``
    based writers above remain for non-CAS atomic overwrites.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_sibling_path(target_path, suffix=".cas.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, target_path)
        except FileExistsError:
            return False
        fsync_parent(target_path)
        return True
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
    "atomic_create_no_overwrite",
    "atomic_write_json",
    "fsync_parent",
    "fsync_path",
    "serialize_json_payload",
    "temp_sibling_path",
]
