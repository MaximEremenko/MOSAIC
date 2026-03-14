from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.application.postprocessing.mode import normalize_post_mode


@dataclass(frozen=True)
class PostprocessingProcessorState:
    parameters: dict[str, Any]
    mode: str
    original_coords: np.ndarray | None
    average_coords: np.ndarray | None
    u_true_all: np.ndarray | None


def build_postprocessing_processor_state(
    parameters: dict[str, Any] | None,
) -> PostprocessingProcessorState:
    normalized = dict(parameters or {})

    rspace_info = normalized.get("rspace_info") or {}
    mode_raw = (
        normalized.get("postprocessing_mode")
        or normalized.get("postprocess_mode")
        or normalized.get("mode")
        or rspace_info.get("postprocessing_mode")
        or rspace_info.get("postprocess_mode")
        or rspace_info.get("mode")
        or "displacement"
    )
    mode = normalize_post_mode(mode_raw)

    normalized.setdefault("normalize_amplitudes_by", "ntotal")
    normalized.setdefault("coords_are_fractional", False)
    normalized.setdefault("ls_weight_gamma", 0.35)
    normalized.setdefault("dog_lambda_reg", 1e-3)
    normalized.setdefault("linear_max_training_samples", None)
    normalized.setdefault("q_window_kind", "cheb")
    normalized.setdefault("q_window_at_db", 100.0)
    normalized.setdefault("edge_guard_frac", 0.10)

    original_coords = None
    average_coords = None
    if mode == "displacement":
        if "original_coords" not in normalized:
            raise KeyError("parameters['original_coords'] is required for displacement mode.")
        if "average_coords" not in normalized:
            raise KeyError(
                "parameters['average_coords'] is required for displacement mode. "
                "Add avg_coords.to_numpy() to params in main.py."
            )
        original_coords = np.asarray(normalized["original_coords"], float)
        average_coords = np.asarray(normalized["average_coords"], float)

    u_true_all = None
    if "displacements_from_config" in normalized:
        u_true_all = np.asarray(normalized["displacements_from_config"], float)

    return PostprocessingProcessorState(
        parameters=normalized,
        mode=mode,
        original_coords=original_coords,
        average_coords=average_coords,
        u_true_all=u_true_all,
    )
