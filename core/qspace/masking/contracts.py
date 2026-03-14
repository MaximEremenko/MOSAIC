from __future__ import annotations

from typing import Protocol

import numpy as np


class MaskStrategy(Protocol):
    def generate_mask(self, data: np.ndarray) -> np.ndarray:
        ...
