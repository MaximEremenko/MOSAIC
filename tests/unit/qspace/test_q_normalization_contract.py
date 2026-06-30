"""W6.2 #1: the single authoritative q-normalization record + byte estimators.

Pins that planned/accepted counts are multiplicity-free and comparable, that masking is
explicit, that multiplicity is applied exactly once, and that the output-byte estimate
matches the durable chunk payload layout.
"""
import pytest

from core.qspace.normalization import (
    QNormalizationContract,
    estimate_attempt_output_bytes,
    estimate_qgrid_input_bytes,
)


def test_contract_records_mask_rejection_and_applies_multiplicity_once():
    # A positive-half interval (multiplicity 2) whose mask dropped 10 of 100 points.
    c = QNormalizationContract(
        planned_count=100,
        accepted_count=90,
        multiplicity=2,
        half_space_role="positive_half",
        interval_id=3,
        q_digest="a" * 64,
    )
    assert c.mask_rejected == 10  # masking is explicit, not silent
    # multiplicity applied exactly once, on the ACCEPTED (masked) base
    assert c.reciprocal_point_count == 180
    # planned and accepted are both multiplicity-free, hence directly comparable
    assert c.accepted_count <= c.planned_count


def test_contract_multiplicity_one_is_identity():
    c = QNormalizationContract(planned_count=50, accepted_count=50, multiplicity=1)
    assert c.mask_rejected == 0
    assert c.reciprocal_point_count == 50


def test_accepted_cannot_exceed_planned():
    with pytest.raises(ValueError, match="accepted masked count cannot exceed"):
        QNormalizationContract(planned_count=10, accepted_count=11, multiplicity=1)


def test_rejects_bad_multiplicity_and_negative_counts():
    with pytest.raises(ValueError, match="multiplicity must be >= 1"):
        QNormalizationContract(planned_count=10, accepted_count=5, multiplicity=0)
    with pytest.raises(ValueError, match="must be non-negative"):
        QNormalizationContract(planned_count=-1, accepted_count=0, multiplicity=1)


def test_estimate_attempt_output_bytes_matches_payload_layout():
    # 4 real-space samples: delta + average (complex128 = 16B each) + point_ids (int64=8B);
    # grid_shape_nd is 4 rows x 1 col int64.
    got = estimate_attempt_output_bytes(
        real_space_sample_count=4, grid_rows=4, grid_cols=1
    )
    assert got == 4 * 2 * 16 + 4 * 8 + 4 * 1 * 8


def test_estimate_qgrid_input_bytes():
    # 90 accepted reciprocal points in 3-D, float64 coordinates.
    assert estimate_qgrid_input_bytes(accepted_count=90, dim=3) == 90 * 3 * 8
