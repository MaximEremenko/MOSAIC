"""Confirm that the preallocated column-stack helper is bitwise identical
to ``np.stack(..., axis=1)`` for 1-D input vectors. No arithmetic, just byte
copies.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.decoding.decoder_service import _stack_features_into_columns


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128])
@pytest.mark.parametrize("P,N", [(1, 1), (4, 3), (257, 11), (13, 257)])
def test_preallocated_stack_matches_np_stack(dtype, P, N):
    rng = np.random.default_rng(0xC0FFEE ^ P ^ N)
    if np.issubdtype(dtype, np.complexfloating):
        features = [
            (rng.standard_normal(P) + 1j * rng.standard_normal(P)).astype(dtype)
            for _ in range(N)
        ]
    else:
        features = [rng.standard_normal(P).astype(dtype) for _ in range(N)]

    ref = np.stack(features, axis=1)
    out = _stack_features_into_columns(features)

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    np.testing.assert_array_equal(out, ref)


def test_preallocated_stack_returns_writable_array():
    feats = [np.arange(5, dtype=np.float64) + k for k in range(3)]
    out = _stack_features_into_columns(feats)
    assert out.flags.writeable
    out[0, 0] = -1.0  # must not raise
