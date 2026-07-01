"""Q-normalization sidecar (q_normalization.json) round-trip.

Pins ``to_payload``/``from_payload`` on the contract and the
``write_q_normalization_sidecar`` / ``load_q_normalization_contract`` pair. The sidecar
is the authority the commit-time reconciliation reads; it lives NEXT TO qspace_plan.json
(same run-scoped directory) but is a SEPARATE file so qspace_plan_digest is untouched.
"""
from __future__ import annotations

import json

from core.qspace.normalization import (
    Q_NORMALIZATION_SIDECAR_NAME,
    QNormalizationContract,
    load_q_normalization_contract,
    q_normalization_sidecar_payload,
    write_q_normalization_sidecar,
)
from core.storage.attempt_store import qspace_plan_path


def test_contract_to_from_payload_round_trip():
    contract = QNormalizationContract(
        planned_count=100,
        accepted_count=90,
        multiplicity=2,
        half_space_role="positive_half",
        interval_id=3,
        q_digest="a" * 64,
    )
    restored = QNormalizationContract.from_payload(contract.to_payload())
    assert restored == contract
    assert restored.reciprocal_point_count == 180
    assert restored.mask_rejected == 10


def test_contract_payload_handles_optional_fields():
    contract = QNormalizationContract(planned_count=10, accepted_count=10, multiplicity=1)
    payload = contract.to_payload()
    assert payload["interval_id"] is None
    assert payload["q_digest"] is None
    assert QNormalizationContract.from_payload(payload) == contract


def test_sidecar_payload_is_ordered_by_interval_id():
    contracts = [
        QNormalizationContract(planned_count=5, accepted_count=5, multiplicity=1, interval_id=2),
        QNormalizationContract(planned_count=5, accepted_count=5, multiplicity=1, interval_id=1),
    ]
    payload = q_normalization_sidecar_payload(contracts)
    assert [item["interval_id"] for item in payload["intervals"]] == [1, 2]


def test_write_and_load_sidecar_round_trip(tmp_path):
    run_digest = "deadbeef" * 4
    contracts = [
        QNormalizationContract(
            planned_count=100,
            accepted_count=90,
            multiplicity=2,
            half_space_role="positive_half",
            interval_id=1,
            q_digest="b" * 64,
        ),
        QNormalizationContract(
            planned_count=4,
            accepted_count=4,
            multiplicity=1,
            half_space_role="zero_plane",
            interval_id=2,
            q_digest="c" * 64,
        ),
    ]

    sidecar_path = write_q_normalization_sidecar(tmp_path, run_digest, contracts)

    # Lives in the SAME run-scoped directory as qspace_plan.json, distinct filename.
    assert sidecar_path.name == Q_NORMALIZATION_SIDECAR_NAME
    assert sidecar_path.parent == qspace_plan_path(tmp_path, run_digest).parent

    loaded_1 = load_q_normalization_contract(tmp_path, run_digest, 1)
    loaded_2 = load_q_normalization_contract(tmp_path, run_digest, 2)
    assert loaded_1 == contracts[0]
    assert loaded_2 == contracts[1]
    assert loaded_1.reciprocal_point_count == 180


def test_load_returns_none_when_sidecar_absent(tmp_path):
    # Back-compat: older runs predate the sidecar; loader returns None (caller skips).
    assert load_q_normalization_contract(tmp_path, "f" * 32, 1) is None


def test_load_returns_none_for_unknown_interval(tmp_path):
    run_digest = "abcd" * 8
    write_q_normalization_sidecar(
        tmp_path,
        run_digest,
        [QNormalizationContract(planned_count=1, accepted_count=1, multiplicity=1, interval_id=5)],
    )
    assert load_q_normalization_contract(tmp_path, run_digest, 999) is None


def test_sidecar_is_valid_compact_json(tmp_path):
    run_digest = "0123456789abcdef" * 2
    path = write_q_normalization_sidecar(
        tmp_path,
        run_digest,
        [QNormalizationContract(planned_count=2, accepted_count=1, multiplicity=1, interval_id=0)],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "mosaic.qspace.q_normalization"
    assert payload["intervals"][0]["planned_count"] == 2
    assert payload["intervals"][0]["accepted_count"] == 1
