"""The structure content digest, and the stability the run tree depends on.

This digest addresses reusable work: the scattering run tree, the work-unit
checkpoints, the stage-1 payload store and the residual reducer-progress
manifests all fold it in. Two properties therefore matter equally:

* it must MOVE when the structure moves — otherwise a re-run credits the
  previous structure's results (measured before this existed: a completed
  case republished byte-identical displacements for changed coordinates);
* it must NOT move otherwise — a digest that drifted between invocations
  would send every restart to a fresh family and destroy kill/resume.

The second is the easier one to break silently, so it is tested across a
fresh interpreter as well as within one.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from core.residual_field.planning import build_residual_field_parameter_digest
from core.structure.identity import (
    StructureIdentityConflict,
    build_structure_content_digest,
    enforce_output_dir_structure,
    structure_content_digest_from_structure,
    structure_identity_path,
)


def _source(seed: int = 0, n: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "original_coords": rng.random((n, 3)),
        "average_coords": rng.random((n, 3)),
        "cells_origin": np.zeros((n, 3)),
        "elements": np.array(["Ti", "O"] * (n // 2)),
        "refnumbers": np.arange(n),
        "vectors": np.eye(3) * 5.0,
        "supercell": np.array([2, 2, 2]),
        "coeff": None,
    }


class TestMovesWithTheStructure:
    def test_any_changed_coordinate_moves_the_digest(self):
        base = _source()
        moved = dict(base)
        shifted = base["original_coords"].copy()
        shifted[0, 1] += 1e-9
        moved["original_coords"] = shifted
        assert build_structure_content_digest(base) != build_structure_content_digest(
            moved
        )

    @pytest.mark.parametrize(
        ("key", "replacement"),
        [
            ("average_coords", np.full((32, 3), 0.25)),
            ("cells_origin", np.ones((32, 3))),
            ("elements", np.array(["Si"] * 32)),
            ("refnumbers", np.arange(32) + 1),
            ("vectors", np.eye(3) * 6.0),
            ("supercell", np.array([2, 2, 4])),
            ("coeff", np.ones(32)),
        ],
    )
    def test_every_member_participates(self, key, replacement):
        base = _source()
        changed = dict(base)
        changed[key] = replacement
        assert not np.array_equal(
            np.asarray(base[key], dtype=object), np.asarray(replacement, dtype=object)
        ) or base[key] is None
        assert build_structure_content_digest(base) != build_structure_content_digest(
            changed
        )

    def test_absent_and_present_never_collide(self):
        base = _source()
        without = {k: v for k, v in base.items() if k != "refnumbers"}
        assert build_structure_content_digest(base) != build_structure_content_digest(
            without
        )


class TestStability:
    """The start/stop guarantee: recomputing the identity for an unchanged
    structure must return the same value, or every restart lands in a new
    checkpoint family and resume is lost."""

    def test_repeated_calls_agree(self):
        source = _source()
        first = build_structure_content_digest(source)
        for _ in range(5):
            assert build_structure_content_digest(source) == first

    def test_equal_but_distinct_arrays_agree(self):
        """A reload parses fresh arrays; equal values must digest equally."""
        a, b = _source(), _source()
        assert a["original_coords"] is not b["original_coords"]
        assert build_structure_content_digest(a) == build_structure_content_digest(b)

    def test_stable_across_a_fresh_interpreter(self):
        """Guards against any hash-seed or iteration-order dependence: a
        restart is a new process, which is exactly the resume case."""
        script = textwrap.dedent(
            """
            import numpy as np
            from core.structure.identity import build_structure_content_digest
            n = 32
            rng = np.random.default_rng(0)
            source = {
                "original_coords": rng.random((n, 3)),
                "average_coords": rng.random((n, 3)),
                "cells_origin": np.zeros((n, 3)),
                "elements": np.array(["Ti", "O"] * (n // 2)),
                "refnumbers": np.arange(n),
                "vectors": np.eye(3) * 5.0,
                "supercell": np.array([2, 2, 2]),
                "coeff": None,
            }
            print(build_structure_content_digest(source))
            """
        )
        digests = set()
        for seed in ("0", "1", "random"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            digests.add(result.stdout.strip())
        assert len(digests) == 1, f"digest drifted across processes: {digests}"
        assert digests.pop() == build_structure_content_digest(_source())

    def test_reads_a_loaded_structure_object(self):
        source = _source()
        structure = type("S", (), source)()
        assert structure_content_digest_from_structure(
            structure
        ) == build_structure_content_digest(source)

    def test_a_missing_member_raises_instead_of_digesting_none(self):
        """Reading identity members with a default is how
        structure_content_digest came to be declared everywhere and
        populated nowhere; a renamed field must not drop out silently."""
        source = _source()
        del source["refnumbers"]
        structure = type("S", (), source)()
        with pytest.raises(AttributeError, match="refnumbers"):
            structure_content_digest_from_structure(structure)


class TestNonArrayMembers:
    """A loaded structure is not all ndarrays: `coeff` arrives as a pandas
    DataFrame. Digesting it naively raised TypeError at the first stage of
    every run — a unit fixture of plain numpy would never have caught it."""

    def test_dataframe_coefficients_digest(self):
        pd = pytest.importorskip("pandas")
        source = _source()
        source["coeff"] = pd.DataFrame({"Ti": np.arange(32.0), "O": np.ones(32)})
        digest = build_structure_content_digest(source)
        assert digest == build_structure_content_digest(dict(source))

    def test_dataframe_values_participate(self):
        pd = pytest.importorskip("pandas")
        base, changed = _source(), _source()
        base["coeff"] = pd.DataFrame({"Ti": np.arange(32.0)})
        changed["coeff"] = pd.DataFrame({"Ti": np.arange(32.0) + 1.0})
        assert build_structure_content_digest(
            base
        ) != build_structure_content_digest(changed)

    def test_column_labels_participate(self):
        pd = pytest.importorskip("pandas")
        base, changed = _source(), _source()
        base["coeff"] = pd.DataFrame({"Ti": np.arange(32.0)})
        changed["coeff"] = pd.DataFrame({"O": np.arange(32.0)})
        assert build_structure_content_digest(
            base
        ) != build_structure_content_digest(changed)

    def test_series_members_digest(self):
        pd = pytest.importorskip("pandas")
        source = _source()
        source["elements"] = pd.Series(["Ti", "O"] * 16)
        assert isinstance(build_structure_content_digest(source), str)


class TestOutputDirectoryGuard:
    """The one check a newly-added cache cannot slip past.

    Reuse is decided in at least five places (run tree, reducer-progress
    manifests, decoder commit, point data, loose residual artifacts) and
    most asked only whether a file exists. Rather than make each one
    structure-aware and hope the next one is too, the directory as a whole
    belongs to exactly one structure."""

    def test_first_run_records_the_identity(self, tmp_path):
        enforce_output_dir_structure(tmp_path, "a" * 64)
        assert structure_identity_path(tmp_path).exists()

    def test_matching_rerun_proceeds(self, tmp_path):
        enforce_output_dir_structure(tmp_path, "a" * 64)
        for _ in range(3):
            enforce_output_dir_structure(tmp_path, "a" * 64)

    def test_a_different_structure_is_refused(self, tmp_path):
        enforce_output_dir_structure(tmp_path, "a" * 64)
        with pytest.raises(StructureIdentityConflict) as excinfo:
            enforce_output_dir_structure(tmp_path, "b" * 64)
        message = str(excinfo.value)
        assert "fresh_start" in message, "the message must name the way out"
        assert "output_directory" in message

    def test_refusal_leaves_the_recorded_identity_intact(self, tmp_path):
        enforce_output_dir_structure(tmp_path, "a" * 64)
        with pytest.raises(StructureIdentityConflict):
            enforce_output_dir_structure(tmp_path, "b" * 64)
        recorded = json.loads(structure_identity_path(tmp_path).read_text())
        assert recorded["source_structure_digest"] == "a" * 64

    def test_an_unreadable_record_fails_closed(self, tmp_path):
        """The record is written atomically, so an unreadable one is not an
        ordinary torn write — and treating it as absent would let through
        exactly the case the guard exists to stop."""
        path = structure_identity_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ truncated")
        with pytest.raises(StructureIdentityConflict):
            enforce_output_dir_structure(tmp_path, "a" * 64)

    def test_a_record_without_a_digest_fails_closed(self, tmp_path):
        path = structure_identity_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema": "mosaic.structure_identity"}')
        with pytest.raises(StructureIdentityConflict):
            enforce_output_dir_structure(tmp_path, "a" * 64)

    def test_a_wiped_directory_starts_over(self, tmp_path):
        """What fresh_start does: remove the directory, and with it the
        record, so the next structure may claim it."""
        import shutil

        enforce_output_dir_structure(tmp_path, "a" * 64)
        shutil.rmtree(tmp_path)
        tmp_path.mkdir()
        enforce_output_dir_structure(tmp_path, "b" * 64)


def _params(content_digest):
    """The REAL parameter types, not a friendly stand-in.

    A dict-shaped fake hid the production bug this test exists for:
    WorkflowRuntimeInfo stores operator keys in `.extra` and exposes them
    only via `.get`, so the digest's `getattr` lookup returned None for
    every structure and the family never moved."""
    from core.models import WorkflowParameters

    return WorkflowParameters(
        schema_version=1,
        struct_info={"dimension": 3, "filename": "s.rmc6f"},
        peak_info={},
        rspace_info={"mode": "displacement", "num_chunks": 4},
        runtime_info={"source_structure_digest": content_digest},
    )


class TestResidualParameterDigest:
    def test_reads_the_digest_through_the_real_runtime_info(self):
        params = _params("a" * 64)
        assert params.runtime_info.get("source_structure_digest") == "a" * 64

    def test_follows_the_structure_content(self):
        """The reducer-progress manifests live OUTSIDE .mosaic/runs/, so the
        run digest cannot scope them — this digest must carry the structure
        itself or a swapped structure reuses their credits."""
        assert build_residual_field_parameter_digest(
            _params("a" * 64)
        ) != build_residual_field_parameter_digest(_params("b" * 64))

    def test_a_missing_digest_does_not_masquerade_as_a_structure(self):
        assert build_residual_field_parameter_digest(
            _params(None)
        ) != build_residual_field_parameter_digest(_params("a" * 64))

    def test_is_stable_for_one_structure(self):
        params = _params("a" * 64)
        first = build_residual_field_parameter_digest(params)
        assert build_residual_field_parameter_digest(params) == first
        assert build_residual_field_parameter_digest(_params("a" * 64)) == first
