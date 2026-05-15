from __future__ import annotations

import json

import numpy as np

from core.scattering.half_space import (
    HALF_SPACE_ROLE_POSITIVE_HALF,
    HALF_SPACE_ROLE_ZERO_PLANE,
)
from core.scattering.planning import (
    build_qspace_plan,
    build_run_identity,
    write_qspace_plan,
    write_run_manifest,
)


def _parameters():
    return {
        "supercell": np.array([4.0, 4.0, 4.0]),
        "vectors": np.eye(3),
        "reciprocal_space_intervals": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
        ],
        "reciprocal_space_intervals_all": [
            {"id": 1, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.0, 0.0)},
            {"id": 2, "h_range": (0.0, 0.0), "k_range": (0.0, 0.0), "l_range": (0.25, 0.25)},
        ],
        "mask_parameters": {},
        "mask_strategy": "none",
        "charge": 1.0,
    }


def _identity(parameters):
    return build_run_identity(
        parameters,
        backend="cpu",
        eps=1e-12,
        dtype="complex128",
        pre_sum_mode="same-q-grid",
        reducer_strategy="chunk",
        backend_policy_digest="policy-a",
    )


def test_qspace_plan_records_explicit_half_space_metadata():
    parameters = _parameters()
    identity = _identity(parameters)

    plan = build_qspace_plan(
        parameters=parameters,
        identity=identity,
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
    )

    by_id = {interval.interval_id: interval for interval in plan.intervals}
    assert by_id[1].half_space_role == HALF_SPACE_ROLE_ZERO_PLANE
    assert by_id[1].reciprocal_multiplicity == 1
    assert by_id[1].l_coverage == "L=0"
    assert by_id[2].half_space_role == HALF_SPACE_ROLE_POSITIVE_HALF
    assert by_id[2].reciprocal_multiplicity == 2
    assert by_id[2].l_coverage == "positive-L"
    assert len(by_id[1].q_grid_digest) == 64
    assert len(by_id[2].q_grid_digest) == 64
    assert by_id[1].q_grid_digest != by_id[2].q_grid_digest


def test_run_manifest_and_qspace_plan_are_written_under_run_digest(tmp_path):
    parameters = _parameters()
    identity = _identity(parameters)
    plan = build_qspace_plan(
        parameters=parameters,
        identity=identity,
        B_=np.eye(3),
        mask_params={},
        MaskStrategy=None,
    )

    run_path = write_run_manifest(
        tmp_path,
        identity,
        execution_contract={
            "backend": "cpu",
            "eps": 1e-12,
            "dtype": "complex128",
            "pre_sum_mode": "same-q-grid",
            "reducer_strategy": "chunk",
        },
    )
    qspace_path = write_qspace_plan(tmp_path, plan)

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    qspace_payload = json.loads(qspace_path.read_text(encoding="utf-8"))

    assert run_path.relative_to(tmp_path).as_posix() == (
        f".mosaic/runs/{identity.run_digest}/run_manifest.json"
    )
    assert qspace_path.relative_to(tmp_path).as_posix() == (
        f".mosaic/runs/{identity.run_digest}/qspace_plan.json"
    )
    assert run_payload["run_digest"] == identity.run_digest
    assert qspace_payload["run_digest"] == identity.run_digest
    assert qspace_payload["intervals"][0]["half_space_role"] == HALF_SPACE_ROLE_ZERO_PLANE
