"""Regression tests for the memoized qspace_psf_in_r.

The cache is pure memoization: same inputs -> same outputs. These tests pin
byte-for-byte equivalence between a cached call and an uncached reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.decoding import features as features_mod
from core.decoding.features import qspace_psf_in_r


@pytest.fixture(autouse=True)
def _reset_psf_cache():
    features_mod._qspace_psf_in_r_cached.cache_clear()
    yield
    features_mod._qspace_psf_in_r_cached.cache_clear()


def _reference_psf(size_aver, hkl, guard_frac, kind, at_db):
    features_mod._qspace_psf_in_r_cached.cache_clear()
    return np.array(
        qspace_psf_in_r(
            size_aver,
            hkl,
            guard_frac=guard_frac,
            window_kind=kind,
            window_at_db=at_db,
        ),
        copy=True,
    )


@pytest.mark.parametrize(
    "size_aver,hkl",
    [
        (np.array([16]), (3.0, 0.0, 0.0)),
        (np.array([12, 10]), (2.5, 1.5, 0.0)),
        (np.array([8, 8, 8]), (2.0, 2.0, 1.0)),
    ],
)
def test_cached_matches_reference_bitwise(size_aver, hkl):
    ref = _reference_psf(size_aver, hkl, 0.1, "cheb", 100.0)
    psf1 = qspace_psf_in_r(
        size_aver, hkl, guard_frac=0.1, window_kind="cheb", window_at_db=100.0
    )
    psf2 = qspace_psf_in_r(
        size_aver, hkl, guard_frac=0.1, window_kind="cheb", window_at_db=100.0
    )
    np.testing.assert_array_equal(psf1, ref)
    np.testing.assert_array_equal(psf2, ref)
    assert psf1 is psf2  # memoization identity


def test_different_keys_produce_independent_entries():
    # Use a setup where guard_frac actually changes the k-band taper:
    # hmax_eff (here 20) must be well below the Nyquist of size_aver/2 (here 32),
    # so the guard band lies inside the representable frequency range.
    kw_a = dict(guard_frac=0.1, window_kind="cheb", window_at_db=100.0)
    kw_b = dict(guard_frac=0.4, window_kind="cheb", window_at_db=100.0)
    a = qspace_psf_in_r(np.array([64, 64]), (20.0, 20.0, 0.0), **kw_a)
    b = qspace_psf_in_r(np.array([64, 64]), (20.0, 20.0, 0.0), **kw_b)
    assert a is not b
    assert not np.array_equal(a, b)


def test_size_aver_input_types_share_cache_entry():
    kw = dict(guard_frac=0.1, window_kind="cheb", window_at_db=100.0)
    a = qspace_psf_in_r(np.array([8, 8]), (1.0, 1.0, 0.0), **kw)
    b = qspace_psf_in_r([8, 8], (1.0, 1.0, 0.0), **kw)
    c = qspace_psf_in_r((8, 8), (1.0, 1.0, 0.0), **kw)
    assert a is b is c
