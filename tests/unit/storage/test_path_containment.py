from __future__ import annotations

import pytest

from core.storage.atomic import (
    PathContainmentError,
    assert_path_contained,
    assert_relative_path_contained,
)


def test_assert_path_contained_accepts_relative_child(tmp_path):
    resolved = assert_path_contained("run/manifest.json", output_dir=tmp_path)

    assert resolved == tmp_path.resolve() / "run" / "manifest.json"


def test_assert_path_contained_accepts_internal_absolute_child(tmp_path):
    child = tmp_path / "run" / "manifest.json"

    resolved = assert_path_contained(child, output_dir=tmp_path)

    assert resolved == child.resolve()


def test_assert_relative_path_contained_accepts_manifest_relative_child(tmp_path):
    resolved = assert_relative_path_contained("run/manifest.json", output_dir=tmp_path)

    assert resolved == tmp_path.resolve() / "run" / "manifest.json"


def test_assert_path_contained_rejects_parent_escape(tmp_path):
    with pytest.raises(PathContainmentError):
        assert_path_contained("../escape.json", output_dir=tmp_path)


def test_assert_relative_path_contained_rejects_absolute_child(tmp_path):
    with pytest.raises(PathContainmentError, match="relative"):
        assert_relative_path_contained(tmp_path / "run" / "manifest.json", output_dir=tmp_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "a//b.json",
        "a/./b.json",
        "../escape.json",
        "C:/escape.json",
        r"a\b.json",
        "a/b/",
    ],
)
def test_assert_relative_path_contained_rejects_invalid_manifest_path_segments(
    tmp_path,
    bad_path,
):
    with pytest.raises(PathContainmentError):
        assert_relative_path_contained(bad_path, output_dir=tmp_path)


def test_assert_path_contained_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)

    with pytest.raises(PathContainmentError):
        assert_path_contained(link, output_dir=tmp_path)
