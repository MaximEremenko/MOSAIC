from __future__ import annotations

from scripts.ci_ast_check import check_paths


def _write(tmp_path, name: str, source: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_ci_ast_check_flags_forbidden_patterns(tmp_path):
    source = _write(
        tmp_path,
        "bad.py",
        """
import hashlib
from somewhere import apply_half_space_conjugate_mirror as mirror
from core.scattering.half_space import HALF_SPACE_ROLE_LEGACY

def run(arr):
    hashlib.sha1(b"data")
    mirror(arr)
    arr.imag = 0.0
    load(allow_legacy=True)
    resolve(legacy_layout=True)
    return HALF_SPACE_ROLE_LEGACY
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "weak-hash" in codes
    assert "forbidden-conjugate-mirror" in codes
    assert "blanket-imag-zeroing" in codes
    assert "current-run-legacy-fallback" in codes
    assert "old-intermediate-fallback" in codes


def test_ci_ast_check_flags_weak_hash_import_aliases(tmp_path):
    source = _write(
        tmp_path,
        "weak_hash.py",
        """
from hashlib import md5, sha1 as digest

def run():
    md5(b"data")
    return digest(b"more")
""",
    )

    codes = [violation.code for violation in check_paths([source])]

    assert codes.count("weak-hash") >= 2


def test_ci_ast_check_flags_old_npz_repair_path(tmp_path):
    source = _write(
        tmp_path,
        "old_npz.py",
        """
def repair(path):
    if path.with_suffix(".npz").exists():
        return path.with_suffix(".npz")
    return path
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "old-intermediate-npz-fallback" in codes


def test_ci_ast_check_flags_old_codebase_artifact_paths_in_core(tmp_path):
    source = _write(
        tmp_path,
        "core/residual_field/artifacts.py",
        """
from pathlib import Path

def old_path(output_dir, chunk_id):
    return Path(output_dir) / "residual_shards" / f"chunk_{chunk_id}"
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "old-codebase-artifact-format" in codes


def test_ci_ast_check_flags_public_paths_in_current_run_workers(tmp_path):
    source = _write(
        tmp_path,
        "core/residual_field/tasks.py",
        """
from pathlib import Path

def run(output_dir, chunk_id):
    return Path(output_dir) / "processed_point_data" / f"residual_chunk_{chunk_id}_amplitudes.hdf5"
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "worker-public-path" in codes


def test_ci_ast_check_flags_old_codebase_fallback_functions_in_workers(tmp_path):
    source = _write(
        tmp_path,
        "core/scattering/execution.py",
        """
from core.scattering.artifacts import reduce_scattering_shards_for_chunk

def run():
    return reduce_scattering_shards_for_chunk()
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "old-codebase-artifact-fallback" in codes


def test_ci_ast_check_allows_new_codebase_checkpoint_terms(tmp_path):
    source = _write(
        tmp_path,
        "core/residual_field/execution.py",
        """
def summarize(checkpoint_metrics, attempt_manifest, chunk_commit, backend):
    backend.persist_shard_checkpoint
    return {
        "checkpoint_metrics": checkpoint_metrics,
        "attempt_manifest": attempt_manifest,
        "chunk_commit": chunk_commit,
    }
""",
    )

    assert check_paths([source]) == []


def test_ci_ast_check_allows_current_backend_checkpoint_api_in_worker(tmp_path):
    source = _write(
        tmp_path,
        "core/residual_field/tasks.py",
        """
def run(backend, work_unit):
    return backend.persist_shard_checkpoint(work_unit)
""",
    )

    assert check_paths([source]) == []


def test_ci_ast_check_allows_public_paths_in_publisher(tmp_path):
    source = _write(
        tmp_path,
        "core/storage/publisher.py",
        """
def public_names(chunk_id):
    return [
        "processed_point_data",
        "public_manifest.json",
        f"point_data_chunk_{chunk_id}_amplitudes.hdf5",
        f"residual_chunk_{chunk_id}_amplitudes.hdf5",
    ]
""",
    )

    assert check_paths([source]) == []


def test_ci_ast_check_flags_half_space_role_repair_from_q_grid(tmp_path):
    source = _write(
        tmp_path,
        "role_repair.py",
        """
def repair_role(q_grid):
    half_space_role = q_grid[:, 2]
    return half_space_role
""",
    )

    codes = {violation.code for violation in check_paths([source])}

    assert "missing-half-space-metadata" in codes


def test_ci_ast_check_allows_q_grid_coordinate_math(tmp_path):
    source = _write(
        tmp_path,
        "math.py",
        """
def l_component(q_grid):
    return q_grid[:, 2]
""",
    )

    assert check_paths([source]) == []
