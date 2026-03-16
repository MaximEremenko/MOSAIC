from __future__ import annotations

from typing import Any

from core.processing_mode import normalize_processing_mode


def normalize_post_mode(mode: Any) -> str:
    return normalize_processing_mode(mode)
