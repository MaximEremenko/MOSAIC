from __future__ import annotations

from typing import Any


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
