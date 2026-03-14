from __future__ import annotations

from typing import Any


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
