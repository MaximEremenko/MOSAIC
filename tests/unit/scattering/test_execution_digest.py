from __future__ import annotations

from core.scattering.planning import build_execution_digest, build_run_digest


SCIENTIFIC_DIGEST = "a" * 64


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
    execution_digest = _execution_digest()

    run_digest = build_run_digest(execution_digest)

    assert run_digest == build_run_digest(execution_digest)
    assert len(run_digest) == 32
    assert run_digest != build_run_digest(_execution_digest(backend="cuda"))
