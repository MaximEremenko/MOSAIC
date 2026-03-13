from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


class CoefficientCenteringService:
    def center(
        self, coeff: np.ndarray, refnumbers: np.ndarray | None, mode: Any
    ) -> np.ndarray:
        coeff = np.asarray(coeff, float)
        if mode in (None, "", "none", False):
            return coeff
        mode_s = str(mode).strip().lower()
        if mode_s in {"global", "mean", "avg"}:
            return coeff - float(np.mean(coeff))
        if mode_s in {"refnumber", "refnumbers", "site", "sites"}:
            if refnumbers is None:
                return coeff - float(np.mean(coeff))
            ref = np.asarray(refnumbers)
            out = coeff.copy()
            for refnum in np.unique(ref):
                mask = ref == refnum
                if np.any(mask):
                    out[mask] -= float(np.mean(coeff[mask]))
            return out
        raise ValueError(f"Unknown coeff centering mode: {mode!r}")
