from __future__ import annotations

from pathlib import Path
from typing import Any


def first_present(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(payload, dict) or key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_s = str(value).strip().lower()
    if value_s in {"1", "true", "yes", "on"}:
        return True
    if value_s in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_path_from(base_dir: Path | str, raw_path: Path | str) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return path


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result
