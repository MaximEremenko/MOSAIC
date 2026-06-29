from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import release_evidence


REPO_ROOT = Path(release_evidence.__file__).resolve().parents[1]

# Every path the explicit ``final_plan.md`` section 13 layout must account for.
EXPECTED_LAYOUT = {
    "commit.txt",
    "ci_urls.txt",
    "dist_sha256.txt",
    "pip_freeze.txt",
    "environment.json",
    "fs_capability.json",
    "public_manifest.json",
    "stage_commits/",
    "chunk_commits/",
    "decoder_commit.json",
    "restart_recovery.log",
    "cross_host_visibility.log",
    "gpu_validation_report.json",
    "hpc_runbook_signoff.txt",
}


def _make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        output_dir=str(tmp_path / "release_evidence"),
        label="testlabel",
        dist_dir=str(tmp_path / "dist"),
        scope="local",
        ci_url=[],
        ci_urls_file=None,
        hpc_smoke_json=[],
        run_output_dir=None,
        run_digest=None,
        restart_recovery_log=None,
        cross_host_visibility_log=None,
        gpu_validation_report=None,
        hpc_runbook_signoff=None,
        run_standard_commands=False,
        fail_on_command_error=False,
        fail_on_missing_required=True,
        command_timeout_seconds=120,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _index_by_path(index: dict) -> dict[str, dict]:
    return {entry["path"]: entry for entry in index["bundle_index"]}


def test_bundle_covers_every_expected_layout_path(tmp_path):
    args = _make_args(tmp_path)
    result = release_evidence.build_bundle(args, REPO_ROOT)
    by_path = _index_by_path(result["index"])

    # The index must account for every explicit layout item, nothing silently
    # dropped.
    assert EXPECTED_LAYOUT <= set(by_path), (
        EXPECTED_LAYOUT - set(by_path)
    )

    # The canonical index file exists and round-trips.
    evidence_path = result["evidence_path"]
    assert evidence_path.exists()
    loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "mosaic.release_evidence"
    assert loaded["scope"] == "local"


def test_locally_derived_evidence_is_present_with_hashes(tmp_path):
    # Provide a dist artifact and a CI URL so dist_sha256 / ci_urls are present.
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "mosaic-0.0.0.tar.gz").write_bytes(b"synthetic sdist")
    args = _make_args(tmp_path, ci_url=["https://ci.example/run/1"])

    result = release_evidence.build_bundle(args, REPO_ROOT)
    evidence_dir = Path(result["index"]["evidence_dir"])
    by_path = _index_by_path(result["index"])

    for name in ("dist_sha256.txt", "pip_freeze.txt", "environment.json", "ci_urls.txt"):
        entry = by_path[name]
        assert entry["present"] is True, (name, entry)
        assert (evidence_dir / name).exists()
        assert entry["sha256"], (name, entry)

    # commit.txt is derived from git; in a git checkout it should be present.
    assert by_path["commit.txt"]["present"] is True

    # dist_sha256.txt records the sha of the synthetic artifact.
    dist_text = (evidence_dir / "dist_sha256.txt").read_text(encoding="utf-8")
    assert "mosaic-0.0.0.tar.gz" in dist_text


def test_hpc_inputs_are_explicitly_missing_when_absent(tmp_path):
    args = _make_args(tmp_path, scope="local")
    result = release_evidence.build_bundle(args, REPO_ROOT)
    by_path = _index_by_path(result["index"])

    for name in (
        "restart_recovery.log",
        "cross_host_visibility.log",
        "gpu_validation_report.json",
        "hpc_runbook_signoff.txt",
        "fs_capability.json",
        "public_manifest.json",
    ):
        entry = by_path[name]
        assert entry["present"] is False, (name, entry)
        assert entry["missing_reason"], (name, entry)
        # Under the local scope these HPC/runtime items are not required.
        assert entry["required"] is False, (name, entry)


def test_run_artifacts_are_copied_from_synthetic_run_dir(tmp_path):
    output_dir = tmp_path / "processed_point_data"
    run_digest = "deadbeef"
    run_root = output_dir / ".mosaic" / "runs" / run_digest
    (run_root / "decoding").mkdir(parents=True)
    (run_root / "fs_capability.json").write_text('{"fs": "ok"}', encoding="utf-8")
    (output_dir / "public_manifest.json").write_text('{"public": true}', encoding="utf-8")
    (run_root / "decoding" / "decoder_commit.json").write_text('{"decoder": 1}', encoding="utf-8")

    stage_dir = run_root / "residual_field"
    (stage_dir / "chunks" / "chunk_000").mkdir(parents=True)
    (stage_dir / "stage_commit.json").write_text('{"stage": "residual_field"}', encoding="utf-8")
    (stage_dir / "chunks" / "chunk_000" / "chunk_commit.json").write_text(
        '{"chunk": 0}', encoding="utf-8"
    )

    args = _make_args(
        tmp_path,
        scope="runtime",
        run_output_dir=str(output_dir),
        run_digest=run_digest,
    )
    result = release_evidence.build_bundle(args, REPO_ROOT)
    evidence_dir = Path(result["index"]["evidence_dir"])
    by_path = _index_by_path(result["index"])

    assert by_path["fs_capability.json"]["present"] is True
    assert by_path["public_manifest.json"]["present"] is True
    assert by_path["decoder_commit.json"]["present"] is True
    assert (evidence_dir / "fs_capability.json").exists()

    stage_entry = by_path["stage_commits/"]
    chunk_entry = by_path["chunk_commits/"]
    assert stage_entry["present"] is True
    assert stage_entry["member_count"] == 1
    assert chunk_entry["present"] is True
    assert chunk_entry["member_count"] == 1
    assert (evidence_dir / "stage_commits" / "residual_field_stage_commit.json").exists()
    assert (
        evidence_dir / "chunk_commits" / "residual_field_chunk_000_chunk_commit.json"
    ).exists()


def test_missing_required_evidence_fails_gate_for_claimed_scope(tmp_path):
    # Claim the full scope without supplying HPC/run inputs: the gate must fail.
    argv = [
        "--output-dir",
        str(tmp_path / "release_evidence"),
        "--label",
        "full",
        "--dist-dir",
        str(tmp_path / "dist"),
        "--scope",
        "full",
    ]
    rc = release_evidence.main(argv)
    assert rc == 2

    index = json.loads(
        (tmp_path / "release_evidence" / "full" / "release_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["complete_for_scope"] is False
    assert "restart_recovery.log" in index["missing_required"]


def test_local_scope_passes_without_hpc_evidence(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "mosaic-0.0.0-py3-none-any.whl").write_bytes(b"synthetic wheel")
    argv = [
        "--output-dir",
        str(tmp_path / "release_evidence"),
        "--label",
        "local",
        "--dist-dir",
        str(dist_dir),
        "--scope",
        "local",
        "--ci-url",
        "https://ci.example/run/2",
    ]
    rc = release_evidence.main(argv)
    assert rc == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
