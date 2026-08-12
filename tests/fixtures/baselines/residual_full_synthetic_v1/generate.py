from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GENERATOR_COMMAND = "python3 tests/fixtures/baselines/residual_full_synthetic_v1/generate.py"
ANCHOR_COMMIT = "f019f76"


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _provenance() -> dict[str, object]:
    return {
        "generator_command": GENERATOR_COMMAND,
        "anchor_commit": ANCHOR_COMMIT,
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "input_files": [],
        "synthetic_only": True,
        "public_representative_fixture": {
            "status": "deferred",
            "reason": "No committed representative public fixture has been selected.",
        },
    }


def _imag_over_real(values: np.ndarray) -> float:
    real_norm = float(np.linalg.norm(np.real(values)))
    imag_norm = float(np.linalg.norm(np.imag(values)))
    return imag_norm / real_norm


def main() -> None:
    q_grid_zero_l = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    q_grid_positive_l = np.array([[0.0, 0.0, 0.25]], dtype=np.float64)
    q_amp_delta = np.array([2.0 + 0.0j], dtype=np.complex128)
    q_amp_average = np.array([1.0 + 0.0j], dtype=np.complex128)
    rifft_grid = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    grid_shape_nd = np.array([[2]], dtype=np.int64)
    inverse_outputs = np.array(
        [[5.0 + 2.0j, -1.0 + 1.0j], [6.0 - 3.0j, 2.0 - 4.0j]],
        dtype=np.complex128,
    )
    positive_half_delta_expected = inverse_outputs[0] + np.conj(inverse_outputs[0])
    positive_half_average_expected = inverse_outputs[1] + np.conj(inverse_outputs[1])
    zero_plane_delta_expected = inverse_outputs[0].copy()
    zero_plane_average_expected = inverse_outputs[1].copy()
    point_ids_expected = np.array([0, 1], dtype=np.int64)

    arrays = {
        "q_grid_zero_l": q_grid_zero_l,
        "q_grid_positive_l": q_grid_positive_l,
        "q_amp_delta": q_amp_delta,
        "q_amp_average": q_amp_average,
        "rifft_grid": rifft_grid,
        "grid_shape_nd": grid_shape_nd,
        "inverse_outputs": inverse_outputs,
        "positive_half_delta_expected": positive_half_delta_expected,
        "positive_half_average_expected": positive_half_average_expected,
        "zero_plane_delta_expected": zero_plane_delta_expected,
        "zero_plane_average_expected": zero_plane_average_expected,
        "point_ids_expected": point_ids_expected,
    }
    np.savez(ROOT / "baseline_arrays.npz", **arrays)

    metadata = {
        "schema": "mosaic.residual_full_synthetic_baseline",
        "schema_version": 1,
        "description": "Synthetic compact baseline for residual-field half-space reconstruction.",
        "generator": "tests/fixtures/baselines/residual_full_synthetic_v1/generate.py",
        "provenance": _provenance(),
        "thresholds": {
            "positive_half_delta_imag_over_real": 1e-12,
            "positive_half_average_imag_over_real": 1e-12,
            "zero_plane_preserves_imag": True,
        },
        "metrics": {
            "positive_half_delta_imag_over_real": _imag_over_real(
                positive_half_delta_expected
            ),
            "positive_half_average_imag_over_real": _imag_over_real(
                positive_half_average_expected
            ),
            "zero_plane_delta_imag_norm": float(
                np.linalg.norm(np.imag(zero_plane_delta_expected))
            ),
            "zero_plane_average_imag_norm": float(
                np.linalg.norm(np.imag(zero_plane_average_expected))
            ),
        },
        "array_sha256": {
            name: _array_sha256(value)
            for name, value in sorted(arrays.items())
        },
    }
    (ROOT / "baseline.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
