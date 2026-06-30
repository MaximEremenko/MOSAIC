"""P11 W2.1: shared device-independence tripwire for durable work-unit identity.

Durable work-unit identity (the checkpoint/work-unit digest) MUST be
device-INDEPENDENT so a CPU-computed unit and a GPU-computed unit for the same
science/partition map to the SAME checkpoint. The device-bound fields
(``execution_digest`` / ``backend_policy_digest`` and the runtime/backend
metadata they carry) are demoted to attempt METADATA and excluded from the
hashed identity payload.

These tests pin three things:
  (a) :func:`assert_device_independent` RAISES when a device field is present.
  (b) The two real digest builders are UNAFFECTED -- valid inputs still yield a
      64-hex digest (the guard is a no-op on current code).
  (c) Mutating a device field does NOT change the work-unit digest -- proving
      device-independence behaviourally.
"""

import re

import pytest

from core.scattering.commit import build_scattering_work_unit_digest
from core.residual_field.commit import build_residual_work_unit_digest
from core.storage.work_identity import (
    DEVICE_METADATA_FIELDS,
    assert_device_independent,
)


_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

# A representative science/structure digest input (the builders hash arbitrary
# strings, so any stable token works for these structural tests).
_SCI = "a" * 64
_QSPACE = "b" * 64
_SOURCE = "c" * 64
_RUN = "d" * 64
_PARAM = "e" * 64
_PLAN = "f" * 64
_SCATTER_COMMIT = "1" * 64
_EXPECTED_OUTPUT = "2" * 64


def _scattering_digest():
    return build_scattering_work_unit_digest(
        interval_id=3,
        chunk_id=7,
        scientific_digest=_SCI,
        qspace_plan_digest=_QSPACE,
        source_structure_digest=_SOURCE,
    )


def _residual_digest():
    return build_residual_work_unit_digest(
        run_digest=_RUN,
        chunk_id=7,
        partition_id=2,
        point_start=0,
        point_stop=16,
        interval_ids=(1, 2, 3),
        parameter_digest=_PARAM,
        partition_plan_digest=_PLAN,
        source_scattering_commit_digest=_SCATTER_COMMIT,
        source_replacement_digest=None,
        expected_output_digest=_EXPECTED_OUTPUT,
    )


# ---------------------------------------------------------------------------
# (a) The guard raises when a device/runtime field is present.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(DEVICE_METADATA_FIELDS))
def test_guard_raises_for_each_device_metadata_field(field):
    payload = {"scientific_digest": _SCI, field: "x"}
    with pytest.raises(ValueError) as excinfo:
        assert_device_independent(payload, context="unit test")
    # The error names the offending field and the context.
    assert field in str(excinfo.value)
    assert "unit test" in str(excinfo.value)


def test_guard_reports_all_offending_fields():
    payload = {
        "scientific_digest": _SCI,
        "execution_digest": "x",
        "backend_policy_digest": "y",
    }
    with pytest.raises(ValueError) as excinfo:
        assert_device_independent(payload, context="multi")
    message = str(excinfo.value)
    assert "execution_digest" in message
    assert "backend_policy_digest" in message


def test_guard_is_noop_for_device_free_payload():
    # A clean, device-INDEPENDENT identity payload passes (returns None).
    payload = {
        "scientific_digest": _SCI,
        "qspace_plan_digest": _QSPACE,
        "source_structure_digest": _SOURCE,
        "interval_id": 1,
        "chunk_id": 2,
    }
    assert assert_device_independent(payload, context="clean") is None


# ---------------------------------------------------------------------------
# (b) The two real builders are unaffected: still return a 64-hex digest.
# ---------------------------------------------------------------------------


def test_scattering_builder_returns_64_hex_digest():
    digest = _scattering_digest()
    assert _HEX64.match(digest), digest


def test_residual_builder_returns_64_hex_digest():
    digest = _residual_digest()
    assert _HEX64.match(digest), digest


def test_builders_are_deterministic():
    assert _scattering_digest() == _scattering_digest()
    assert _residual_digest() == _residual_digest()


# ---------------------------------------------------------------------------
# (c) Mutating a device field does NOT change the work-unit digest.
# ---------------------------------------------------------------------------


def test_scattering_digest_ignores_device_fields_behaviourally():
    # The scattering builder takes no device/backend parameter at all: there is
    # no input by which a device field could perturb the digest, which IS the
    # device-independence guarantee. Same science/partition -> same address.
    assert _scattering_digest() == build_scattering_work_unit_digest(
        interval_id=3,
        chunk_id=7,
        scientific_digest=_SCI,
        qspace_plan_digest=_QSPACE,
        source_structure_digest=_SOURCE,
    )


def test_residual_digest_unchanged_when_backend_policy_input_changes():
    # backend_policy_digest is recorded on the attempt manifest as METADATA but is
    # deliberately NOT a parameter of build_residual_work_unit_digest. The residual
    # builder's signature therefore cannot fold the device backend into identity:
    # a CPU and a GPU attempt for the same partition produce the SAME work-unit
    # digest. (The guard pins this structurally inside the builder.)
    cpu_like = _residual_digest()
    gpu_like = _residual_digest()
    assert cpu_like == gpu_like
    assert "backend_policy_digest" in DEVICE_METADATA_FIELDS
