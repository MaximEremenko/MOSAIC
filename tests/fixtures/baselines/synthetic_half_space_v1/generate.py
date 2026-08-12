from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GENERATOR_COMMAND = "python3 tests/fixtures/baselines/synthetic_half_space_v1/generate.py"
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


def main() -> None:
    q_grid_zero_l = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    q_grid_positive_l = np.array(
        [[0.0, 0.0, 0.25], [1.0, 0.0, 0.50]],
        dtype=np.float64,
    )
    q_grid_misleading_positive_role = q_grid_zero_l.copy()
    q_grid_misleading_zero_role = q_grid_positive_l.copy()
    complex_values = np.array([1.25 + 2.5j, -3.0 + 4.0j], dtype=np.complex128)
    positive_half_expected = complex_values + np.conj(complex_values)
    zero_plane_expected = complex_values.copy()

    arrays = {
        "q_grid_zero_l": q_grid_zero_l,
        "q_grid_positive_l": q_grid_positive_l,
        "q_grid_misleading_positive_role": q_grid_misleading_positive_role,
        "q_grid_misleading_zero_role": q_grid_misleading_zero_role,
        "complex_values": complex_values,
        "positive_half_expected": positive_half_expected,
        "zero_plane_expected": zero_plane_expected,
    }
    np.savez(ROOT / "baseline_arrays.npz", **arrays)

    real_norm = float(np.linalg.norm(np.real(positive_half_expected)))
    imag_norm = float(np.linalg.norm(np.imag(positive_half_expected)))
    metadata = {
        "schema": "mosaic.synthetic_half_space_baseline",
        "schema_version": 1,
        "description": "Synthetic compact baseline for half-space RIFFT reconstruction.",
        "generator": "tests/fixtures/baselines/synthetic_half_space_v1/generate.py",
        "provenance": _provenance(),
        "thresholds": {
            "positive_half_imag_over_real": 1e-12,
            "zero_plane_preserves_imag": True,
        },
        "metrics": {
            "positive_half_imag_norm": imag_norm,
            "positive_half_real_norm": real_norm,
            "positive_half_imag_over_real": imag_norm / real_norm,
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
