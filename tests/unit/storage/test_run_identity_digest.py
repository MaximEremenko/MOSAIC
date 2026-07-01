"""device-independent run/checkpoint identity (`build_run_identity_digest`).

The whole point of is that a durable checkpoint is addressed by the science +
numerical contract, NOT by the device that computed it -- so CPU and GPU runs of
the same science share one run tree and one set of checkpoint addresses, and
cross-device promotion is decided by scientific-invariant validation rather than
output-byte equality (impossible across CPU/GPU, and even across GPU launches).
"""

import pytest

from core.storage.digests import (
    build_execution_digest,
    build_run_identity_digest,
)


_SCI_A = "a" * 64
_SCI_B = "b" * 64


def _run_identity(**overrides):
    kwargs = dict(
        scientific_digest=_SCI_A,
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="grid",
        reducer_strategy="manifest",
        schema_version=1,
    )
    kwargs.update(overrides)
    return build_run_identity_digest(**kwargs)


def test_run_identity_is_device_independent_unlike_execution_digest():
    # The device-bound execution digests differ ONLY by backend ...
    common = dict(
        scientific_digest=_SCI_A,
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="grid",
        reducer_strategy="manifest",
        schema_version=1,
        domain="mosaic.scattering.execution.v1",
    )
    cpu_exec = build_execution_digest(backend="cpu", **common)
    gpu_exec = build_execution_digest(backend="cuda", **common)
    assert cpu_exec != gpu_exec  # device-bound identity separates CPU and GPU

    # ... but the run identity takes no backend at all: there is no input by
    # which CPU and GPU of the same science/contract could ever diverge, so they
    # address the SAME checkpoint.
    assert _run_identity() == _run_identity()


def test_run_identity_is_deterministic():
    assert _run_identity() == _run_identity()


def test_run_identity_changes_with_science_and_numerical_contract():
    base = _run_identity()
    assert _run_identity(scientific_digest=_SCI_B) != base
    assert _run_identity(eps=1e-6) != base
    assert _run_identity(dtype="complex64") != base
    assert _run_identity(pre_sum_mode="none") != base
    assert _run_identity(reducer_strategy="local") != base


def test_run_identity_rejects_non_sha256_scientific_digest():
    with pytest.raises(ValueError):
        _run_identity(scientific_digest="not-a-digest")


def test_run_identity_length_is_32():
    assert len(_run_identity()) == 32
