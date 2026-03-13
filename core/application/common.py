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


def normalize_post_mode(mode: Any) -> str:
    mode_norm = str(mode or "").strip().lower()
    if mode_norm in {
        "chemical",
        "chem",
        "checmical",
        "occupational",
        "occupancy",
        "occupantioal",
    }:
        return "chemical"
    return "displacement"


def pad_interval(interval: dict[str, Any], dim: int) -> dict[str, Any]:
    base: dict[str, Any] = {}
    if "h_range" in interval:
        base["h_range"] = interval["h_range"]
        base["h_start"], base["h_end"] = interval["h_range"]
    if dim >= 2:
        base["k_range"] = interval.get("k_range", (0, 0))
        base["k_start"], base["k_end"] = base["k_range"]
    else:
        base["k_range"] = (0, 0)
        base["k_start"] = 0
        base["k_end"] = 0
    if dim == 3:
        base["l_range"] = interval.get("l_range", (0, 0))
        base["l_start"], base["l_end"] = base["l_range"]
    else:
        base["l_range"] = (0, 0)
        base["l_start"] = 0
        base["l_end"] = 0
    return base

