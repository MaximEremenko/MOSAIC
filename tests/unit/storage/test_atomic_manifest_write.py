from __future__ import annotations

import json

import pytest

from core.storage.atomic import PathContainmentError, atomic_write_json


def test_atomic_write_json_writes_valid_json_and_cleans_temp(tmp_path):
    target = tmp_path / "manifest.json"

    atomic_write_json(target, {"schema_version": 1, "schema": "x"}, output_dir=tmp_path)

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "schema": "x",
        "schema_version": 1,
    }
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_atomic_write_json_validator_runs_before_and_after_reopen(tmp_path):
    target = tmp_path / "manifest.json"
    calls = []

    def validator(payload):
        calls.append(dict(payload))

    atomic_write_json(
        target,
        {"schema": "x", "schema_version": 1},
        output_dir=tmp_path,
        validator=validator,
    )

    assert calls == [
        {"schema": "x", "schema_version": 1},
        {"schema": "x", "schema_version": 1},
    ]


def test_atomic_write_json_pre_validation_failure_preserves_existing_manifest(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"schema":"old","schema_version":1}\n', encoding="utf-8")

    def validator(payload):
        raise ValueError("invalid manifest")

    with pytest.raises(ValueError, match="invalid manifest"):
        atomic_write_json(
            target,
            {"schema": "new", "schema_version": 1},
            output_dir=tmp_path,
            validator=validator,
        )

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "schema": "old",
        "schema_version": 1,
    }
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_atomic_write_json_rejects_outside_output_dir(tmp_path):
    with pytest.raises(PathContainmentError):
        atomic_write_json(tmp_path.parent / "manifest.json", {}, output_dir=tmp_path)
