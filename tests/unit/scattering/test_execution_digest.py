from __future__ import annotations

import numpy as np

from core.scattering.planning import (
    build_execution_digest,
    build_run_digest,
    build_run_identity,
)


SCIENTIFIC_DIGEST = "a" * 64


def _run_identity_parameters():
    return {
        "supercell": np.array([4.0, 4.0, 4.0]),
        "vectors": np.eye(3),
        "reciprocal_space_intervals": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
        ],
        "reciprocal_space_intervals_all": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
        ],
        "mask_parameters": {},
        "mask_strategy": "none",
        "charge": 1.0,
    }


def _execution_digest(**overrides):
    kwargs = {
        "scientific_digest": SCIENTIFIC_DIGEST,
        "backend": "cpu",
        "eps": 1e-12,
        "dtype": "complex128",
        "pre_sum_mode": "same-q-grid",
        "reducer_strategy": "chunk",
        "backend_policy_digest": "policy-a",
    }
    kwargs.update(overrides)
    return build_execution_digest(**kwargs)


def test_execution_digest_includes_execution_contract_fields():
    baseline = _execution_digest()

    assert _execution_digest(backend="cuda") != baseline
    assert _execution_digest(eps=1e-9) != baseline
    assert _execution_digest(dtype="complex64") != baseline
    assert _execution_digest(pre_sum_mode="off") != baseline
    assert _execution_digest(reducer_strategy="streaming") != baseline
    assert _execution_digest(backend_policy_digest="policy-b") != baseline


def test_run_digest_is_deterministic_execution_digest_prefix_identity():
    # NOTE: this exercises the RETAINED low-level, device-bound ``build_run_digest``
    # (kept for migration/tests). The run/checkpoint TREE is no longer addressed this
    # way -- ``build_run_identity`` now anchors it device-INDEPENDENTLY (see below).
    execution_digest = _execution_digest()

    run_digest = build_run_digest(execution_digest)

    assert run_digest == build_run_digest(execution_digest)
    assert len(run_digest) == 32
    assert run_digest != build_run_digest(_execution_digest(backend="cuda"))


def test_run_identity_run_digest_is_device_independent():
    # device-independent identity: the run/checkpoint tree is addressed device-INDEPENDENTLY. A CPU run
    # and a GPU run of the same science + numerical contract share ONE run_digest
    # (one `.mosaic/runs/<digest>/` tree), so either device can resume/promote the
    # other's chunks. The device-bound execution_digest still differs and is retained
    # as metadata only.
    parameters = _run_identity_parameters()
    common = dict(
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        backend_policy_digest="policy-a",
    )
    cpu = build_run_identity(parameters, backend="cpu", **common)
    gpu = build_run_identity(parameters, backend="cuda", **common)

    assert cpu.run_digest == gpu.run_digest
    assert cpu.scientific_digest == gpu.scientific_digest
    assert cpu.execution_digest != gpu.execution_digest


def test_run_identity_run_digest_tracks_numerical_contract():
    # eps/dtype/pre-sum/reducer ARE identity: a different numerical contract is a
    # different checkpoint tree (it must NOT share with another-precision run).
    parameters = _run_identity_parameters()
    base = build_run_identity(
        parameters,
        backend="cpu",
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        backend_policy_digest="policy-a",
    )
    other_eps = build_run_identity(
        parameters,
        backend="cpu",
        eps=1e-9,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        backend_policy_digest="policy-a",
    )

    assert other_eps.run_digest != base.run_digest
