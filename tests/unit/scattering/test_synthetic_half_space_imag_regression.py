from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.scattering.accumulation import apply_half_space_conjugate_reconstruction
from core.scattering.half_space import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
)


BASELINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "baselines"
    / "synthetic_half_space_v1"
)


def _load_baseline():
    metadata = json.loads((BASELINE_DIR / "baseline.json").read_text(encoding="utf-8"))
    arrays = np.load(BASELINE_DIR / "baseline_arrays.npz")
    return metadata, arrays


def _assert_required_provenance(metadata):
    provenance = metadata["provenance"]
    assert provenance["generator_command"] == (
        "python3 tests/fixtures/baselines/synthetic_half_space_v1/generate.py"
    )
    assert provenance["anchor_commit"] == "f019f76"
    assert provenance["input_files"] == []
    assert provenance["synthetic_only"] is True
    assert provenance["public_representative_fixture"]["status"] == "deferred"
    dependency_versions = provenance["dependency_versions"]
    assert dependency_versions["python"]
    assert dependency_versions["numpy"]


def test_baseline_metadata_records_reproducibility_provenance():
    metadata, _arrays = _load_baseline()

    _assert_required_provenance(metadata)


def test_positive_half_reconstructs_to_real_without_imaginary_zeroing():
    metadata, arrays = _load_baseline()
    reconstructed = apply_half_space_conjugate_reconstruction(
        arrays["complex_values"],
        arrays["q_grid_positive_l"],
        HALF_SPACE_ROLE_POSITIVE_HALF,
    )

    np.testing.assert_allclose(reconstructed, arrays["positive_half_expected"])
    ratio = np.linalg.norm(np.imag(reconstructed)) / np.linalg.norm(np.real(reconstructed))
    assert ratio < metadata["thresholds"]["positive_half_imag_over_real"]


def test_zero_plane_preserves_imaginary_signal():
    _metadata, arrays = _load_baseline()
    reconstructed = apply_half_space_conjugate_reconstruction(
        arrays["complex_values"],
        arrays["q_grid_zero_l"],
        HALF_SPACE_ROLE_ZERO_PLANE,
    )

    np.testing.assert_allclose(reconstructed, arrays["zero_plane_expected"])
    assert np.linalg.norm(np.imag(reconstructed)) > 0


def test_explicit_half_space_role_overrides_q_grid_third_coordinate():
    _metadata, arrays = _load_baseline()

    positive_from_zero_l_grid = apply_half_space_conjugate_reconstruction(
        arrays["complex_values"],
        arrays["q_grid_misleading_positive_role"],
        HALF_SPACE_ROLE_POSITIVE_HALF,
    )
    zero_from_positive_l_grid = apply_half_space_conjugate_reconstruction(
        arrays["complex_values"],
        arrays["q_grid_misleading_zero_role"],
        HALF_SPACE_ROLE_ZERO_PLANE,
    )

    np.testing.assert_allclose(positive_from_zero_l_grid, arrays["positive_half_expected"])
    np.testing.assert_allclose(zero_from_positive_l_grid, arrays["zero_plane_expected"])
