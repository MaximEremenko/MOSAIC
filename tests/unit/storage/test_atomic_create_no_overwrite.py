from __future__ import annotations

import pytest

from core.storage.atomic import atomic_create_no_overwrite


def test_first_writer_creates_and_writes_bytes(tmp_path):
    target = tmp_path / "commit.json"
    payload = b'{"schema":"x","schema_version":1}\n'

    created = atomic_create_no_overwrite(target, payload)

    assert created is True
    assert target.read_bytes() == payload
    # The hard-link temp file is removed in the finally block.
    assert list(tmp_path.glob(".commit.json.*.cas.tmp")) == []


def test_second_writer_skips_and_leaves_bytes_intact(tmp_path):
    target = tmp_path / "commit.json"
    first = b'{"schema":"x","schema_version":1}\n'
    second = b'{"schema":"y","schema_version":2}\n'

    assert atomic_create_no_overwrite(target, first) is True

    created = atomic_create_no_overwrite(target, second)

    assert created is False
    # The existing committed bytes are untouched -- no overwrite.
    assert target.read_bytes() == first
    assert list(tmp_path.glob(".commit.json.*.cas.tmp")) == []


def test_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "nested" / "deep" / "commit.json"
    payload = b"hello\n"

    created = atomic_create_no_overwrite(target, payload)

    assert created is True
    assert target.read_bytes() == payload
