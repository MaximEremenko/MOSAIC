from __future__ import annotations

import numpy as np

from core.scattering.planning import build_scientific_digest


def _parameters(**overrides):
    base = {
        "supercell": np.array([4.0, 4.0, 4.0]),
        "vectors": np.eye(3),
        "reciprocal_space_intervals": [
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
        ],
        "reciprocal_space_intervals_all": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
        ],
        "mask_parameters": {"kind": "none"},
        "mask_strategy": "none",
        "charge": 1.0,
        "use_coeff": False,
        "output_dir": "/run/a",
        "backend": "cpu",
        "eps": 1e-6,
    }
    base.update(overrides)
    return base


def test_scientific_digest_ignores_execution_and_output_fields():
    left = build_scientific_digest(_parameters())
    right = build_scientific_digest(
        _parameters(output_dir="/run/b", backend="cuda", eps=1e-12)
    )

    assert left == right
    assert len(left) == 64


def test_scientific_digest_changes_when_science_changes():
    baseline = build_scientific_digest(_parameters())
    charge_changed = build_scientific_digest(_parameters(charge=2.0))
    interval_changed = build_scientific_digest(
        _parameters(
            reciprocal_space_intervals=[
                {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            ],
            reciprocal_space_intervals_all=[
                {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            ],
        )
    )

    assert charge_changed != baseline
    assert interval_changed != baseline
