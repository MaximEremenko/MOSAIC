from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodingRequest:
    output_dir: str
    parameters: dict[str, Any]
    client: Any


__all__ = ["DecodingRequest"]
