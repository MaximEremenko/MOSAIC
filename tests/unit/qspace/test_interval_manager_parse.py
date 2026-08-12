"""Security and round-trip tests for ReciprocalSpaceIntervalManager parsing.

These tests guard against the previous use of ``eval()`` on interval strings
loaded from HDF5 attributes (config-derived / on-disk input), which allowed
arbitrary code execution. Parsing now goes through
``ReciprocalSpaceIntervalManager._parse_interval_str`` (``ast.literal_eval``
plus structural validation).
"""

import os

import h5py
import pytest

from core.qspace.intervals.manager import ReciprocalSpaceIntervalManager


def _make_manager(tmp_path, supercell=(2, 2, 2)):
    hdf5_path = os.path.join(str(tmp_path), "intervals.h5")
    return ReciprocalSpaceIntervalManager(hdf5_path, parameters={}, supercell=supercell)


# --- Valid inputs still parse correctly --------------------------------------

def test_parse_valid_interval_string():
    parse = ReciprocalSpaceIntervalManager._parse_interval_str
    interval = parse("{'h_range': (0.0, 1.0), 'k_range': (-2.0, 2.0), 'l_range': (0.0, 0.0)}")
    assert interval == {
        "h_range": (0.0, 1.0),
        "k_range": (-2.0, 2.0),
        "l_range": (0.0, 0.0),
    }
    # All endpoints normalised to float.
    for start, end in interval.values():
        assert isinstance(start, float) and isinstance(end, float)


def test_parse_accepts_integer_endpoints_and_bytes():
    parse = ReciprocalSpaceIntervalManager._parse_interval_str
    interval = parse(b"{'h_range': (0, 3)}")
    assert interval == {"h_range": (0.0, 3.0)}


def test_round_trip_matches_str_repr():
    """str(interval) -> _parse_interval_str round-trips to an equal dict."""
    parse = ReciprocalSpaceIntervalManager._parse_interval_str
    original = {"h_range": (0.5, 4.5), "k_range": (-1.0, 1.0)}
    assert parse(str(original)) == original


# --- Malicious / malformed inputs are rejected (cannot execute) --------------

@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('echo pwned')",
        "[].__class__.__base__",
        "{'h_range': (0.0, 1.0)}; __import__('os').system('echo pwned')",
        "open('pwned.txt', 'w').write('x')",
        "exec(\"print('x')\")",
        "1 + 1",                       # not a dict
        "['not', 'a', 'dict']",        # not a dict
        "{'h_range': (0.0, 1.0, 2.0)}",  # wrong arity
        "{'h_range': 1.0}",            # range not a pair
        "{1: (0.0, 1.0)}",            # non-string key
        "{'h_range': ('a', 'b')}",     # non-numeric endpoints
        "not valid python at all",
    ],
)
def test_parse_rejects_malicious_or_malformed_input(payload):
    with pytest.raises(ValueError):
        ReciprocalSpaceIntervalManager._parse_interval_str(payload)


def test_malicious_string_does_not_execute(tmp_path):
    """A code-execution payload stored in the HDF5 attr must not run on load."""
    sentinel = os.path.join(str(tmp_path), "pwned.txt")
    payload = f"open({sentinel!r}, 'w').write('pwned')"

    manager = _make_manager(tmp_path)
    with h5py.File(manager.hdf5_file_path, "w") as h5file:
        grp = h5file.create_group("reciprocal_space_intervals")
        grp.attrs["0"] = payload

    # load_from_hdf5 must fail gracefully (returns False) and never execute code.
    assert manager.load_from_hdf5() is False
    assert not os.path.exists(sentinel), "payload executed: parser ran arbitrary code"


def test_load_from_hdf5_round_trip(tmp_path):
    """Save then load reproduces the original intervals (behavior preserved)."""
    manager = _make_manager(tmp_path)
    manager.reciprocal_space_intervals = [
        {"h_range": (0.0, 1.0), "k_range": (-2.0, 2.0), "l_range": (0.0, 0.0)},
        {"h_range": (1.0, 2.0), "k_range": (0.0, 1.0), "l_range": (-1.0, 1.0)},
    ]
    manager.save_to_hdf5()

    reloaded = _make_manager(tmp_path)
    assert reloaded.load_from_hdf5() is True
    assert reloaded.reciprocal_space_intervals == manager.reciprocal_space_intervals
