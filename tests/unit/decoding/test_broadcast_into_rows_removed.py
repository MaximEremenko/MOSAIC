"""Guard against the broadcast_into_rows dead branch returning.

The ``broadcast_into_rows`` parameter was removed from
``compute_and_save_displacements`` (and its plumbing from
``PointDataPostprocessingProcessor``) because the branch body referenced names
that were never in scope (``ids_all``, ``amplitudes``), making it an
unconditional ``NameError`` when triggered via
``processor.parameters["broadcast_displacement_into_rows"] = True``.

These tests verify:
1. ``compute_and_save_displacements`` no longer accepts a
   ``broadcast_into_rows`` keyword argument.
2. ``PointDataPostprocessingProcessor.compute_and_save_displacements`` no longer
   accepts a ``broadcast_into_rows`` keyword argument.
3. Calling ``compute_and_save_displacements`` without the removed parameter
   still produces the expected ``out_table`` dict (``central_point_id`` and
   ``u`` keys).
"""

from __future__ import annotations

import inspect
import os
import types

import numpy as np
import pytest

from core.decoding.displacement import compute_and_save_displacements
from core.decoding.decode_chunk import PointDataPostprocessingProcessor


# ---------------------------------------------------------------------------
# Parameter-signature tests
# ---------------------------------------------------------------------------

def test_compute_and_save_displacements_has_no_broadcast_into_rows_param():
    """The dead ``broadcast_into_rows`` parameter must not appear in the
    public signature of ``compute_and_save_displacements``."""
    sig = inspect.signature(compute_and_save_displacements)
    assert "broadcast_into_rows" not in sig.parameters, (
        "broadcast_into_rows was re-added to compute_and_save_displacements; "
        "this parameter references out-of-scope names and must stay removed."
    )


def test_processor_compute_and_save_displacements_has_no_broadcast_into_rows_param():
    """The dead ``broadcast_into_rows`` parameter must not appear on the
    ``PointDataPostprocessingProcessor.compute_and_save_displacements`` method."""
    sig = inspect.signature(PointDataPostprocessingProcessor.compute_and_save_displacements)
    assert "broadcast_into_rows" not in sig.parameters, (
        "broadcast_into_rows was re-added to "
        "PointDataPostprocessingProcessor.compute_and_save_displacements."
    )


def test_passing_broadcast_into_rows_raises_type_error():
    """Callers that pass ``broadcast_into_rows`` must get a ``TypeError``
    (unexpected keyword argument) rather than silently ignoring the flag."""
    with pytest.raises(TypeError, match="broadcast_into_rows"):
        compute_and_save_displacements(
            object(),  # processor — will never be used, TypeError fires first
            chunk_id=0,
            rifft_saver=None,
            point_data_list=[],
            broadcast_into_rows=True,
        )


# ---------------------------------------------------------------------------
# Functional smoke test: the function still produces out_table correctly
# ---------------------------------------------------------------------------

def _make_processor(tmp_path):
    """Build a minimal namespace that satisfies compute_and_save_displacements."""
    return types.SimpleNamespace(
        parameters={
            "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0)}],
            "supercell": np.array([4]),
            "vectors": np.array([[1.0]]),
            "q_window_kind": "cheb",
            "q_window_at_db": 100.0,
            "edge_guard_frac": 0.1,
            "ls_weight_gamma": 0.35,
            "linear_max_training_samples": None,
        },
        original_coords=np.array([[0.0]], dtype=np.float64),
        average_coords=np.array([[0.0]], dtype=np.float64),
        u_true_all=None,
        decoder_source_policy=types.SimpleNamespace(assignment="single"),
        _decoder_M=np.array([[1.0]], dtype=np.float64),
        _feature_dim=1,
        _decoder_family=None,
        _decoder_feature_dims=None,
    )


def test_compute_and_save_displacements_returns_out_table(monkeypatch, tmp_path):
    """compute_and_save_displacements must return an out_table dict with
    ``central_point_id`` and ``u`` keys after removing broadcast_into_rows."""
    processor = _make_processor(tmp_path)

    point_data_list = [
        {
            "central_point_id": 0,
            "coordinates": np.array([0.0], dtype=np.float64),
            "dist_from_atom_center": np.array([0.2], dtype=np.float64),
            "step_in_frac": np.array([0.05], dtype=np.float64),
        }
    ]

    # Stub out the heavy IO helpers so the test stays unit-level.
    monkeypatch.setattr(
        "core.decoding.displacement_inputs.load_chunk_residual_field_and_grid",
        lambda *args, **kwargs: (
            {},
            np.array([[0.0, 1.0]], dtype=np.float64),  # amplitudes (N, 2)
            np.array([[0.0, 0.0]], dtype=np.float64),  # rifft_space_grid (N, D+1)
        ),
    )
    monkeypatch.setattr(
        "core.decoding.displacement_inputs.build_feature_sets",
        lambda *args, **kwargs: (
            [np.array([1.0], dtype=np.float64)],  # features_all
            [0],                                   # cids_all
            [None],                                # decoder_keys_all (unused here)
            [],                                    # features_train
            [],                                    # u_train
            [],                                    # training_decoder_keys
        ),
    )

    saved = {}

    class _FakeSaver:
        def save_data(self, table, path):
            saved["table"] = table
            saved["path"] = path

        def get_output_dir(self, chunk_id):
            return str(tmp_path)

    monkeypatch.setattr(
        "core.decoding.displacement.write_displacements_csv",
        lambda path, ids, U: None,
    )

    out_table = compute_and_save_displacements(
        processor,
        chunk_id=0,
        rifft_saver=_FakeSaver(),
        point_data_list=point_data_list,
        output_dir=str(tmp_path),
    )

    assert "central_point_id" in out_table, "out_table must contain central_point_id"
    assert "u" in out_table, "out_table must contain u"
    assert out_table["central_point_id"][0] == 0
    assert out_table["u"].shape == (1, 1)
