from __future__ import annotations

import hashlib

import numpy as np

from core.storage.fingerprint import file_sha256, payload_sha256


def test_file_sha256_hashes_exact_file_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


def test_payload_sha256_is_dataset_order_and_endian_stable():
    native = np.array([1, 2, 3], dtype=np.int32)
    big_endian = native.astype(">i4")

    left = payload_sha256(
        schema="mosaic.test",
        expected_set_digest="abc",
        datasets={"b": native, "a": np.array([1.5], dtype=np.float64)},
        attrs={"z": 1, "a": 2},
    )
    right = payload_sha256(
        schema="mosaic.test",
        expected_set_digest="abc",
        datasets={"a": np.array([1.5], dtype=np.float64), "b": big_endian},
        attrs={"a": 2, "z": 1},
    )

    assert left == right
