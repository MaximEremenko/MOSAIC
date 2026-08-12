"""Shared device-independence tripwire for durable work-unit identity .

Durable work-unit identity (the checkpoint/work-unit digest and the attempt
identity tuple) MUST be DEVICE-INDEPENDENT: a CPU-computed unit and a
GPU-computed unit for the same science/partition map to the SAME address and
promote to the same checkpoint. The device-bound fields (``execution_digest``,
``backend_policy_digest``, and the runtime/backend metadata they carry) are
demoted to attempt METADATA and are deliberately EXCLUDED from the hashed
work-unit digest and the identity tuple.

Today that exclusion is enforced only by convention in two modules that do NOT
share code (``core/scattering/commit.py`` and ``core/residual_field/commit.py``).
:func:`assert_device_independent` is the single shared tripwire that pins the
invariant structurally: a future edit that folds a device/runtime field into the
hashed identity payload FAILS LOUDLY instead of silently splitting CPU and GPU
checkpoints.

This module is a PURE LEAF: it imports only ``typing`` / ``collections.abc`` so
it can never participate in an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping


# Canonical device/runtime denylist. Every name here is a field that encodes the
# DEVICE or RUNTIME that computed a work unit -- it is legitimate attempt
# METADATA but must NEVER appear in the hashed work-unit identity payload.
#
# Confirmed against core/scattering/commit.py, core/residual_field/commit.py,
# core/storage/digests.py and core/runtime/gpu_admission.py:
#   - execution_digest      : scattering attempt field; includes ``backend``.
#   - backend_policy_digest : scattering + residual attempt field; carries
#                             ``backend_kind``.
#   - backend / backend_kind: the device kind (cpu/cuda) folded into the above.
#   - device                : generic device identifier.
#   - scheduler_kind        : runtime_provenance key (gpu_admission.py).
#   - thread_count          : CPU thread-policy concept (worker_nthreads, etc.).
#   - nufft_policy          : runtime_provenance key (gpu_admission.py).
#   - cuda_probe            : runtime_provenance_for_attempt parameter.
#   - runtime_provenance    : the whole attempt-metadata provenance dict.
DEVICE_METADATA_FIELDS = frozenset(
    {
        "execution_digest",
        "backend_policy_digest",
        "backend",
        "backend_kind",
        "device",
        "scheduler_kind",
        "thread_count",
        "nufft_policy",
        "cuda_probe",
        "runtime_provenance",
    }
)


def assert_device_independent(payload: Mapping[str, object], *, context: str) -> None:
    """Fail closed if any device/runtime field leaked into an identity payload.

    ``payload`` is the dict that is about to be hashed into a durable work-unit
    digest. If it contains any key in :data:`DEVICE_METADATA_FIELDS`, identity has
    been made device-dependent (CPU and GPU units would split into different
    checkpoints) -- this raises a clear :class:`ValueError` naming the offending
    fields and the ``context``.

    On current code this is a NO-OP: the two real builders already exclude device
    fields, so this is a structural tripwire that pins the invariant, not a change
    in behaviour.
    """
    offending = sorted(DEVICE_METADATA_FIELDS.intersection(payload.keys()))
    if offending:
        raise ValueError(
            f"Work-unit identity payload for {context} must be device-INDEPENDENT, "
            f"but it contains device/runtime metadata field(s): "
            f"{', '.join(offending)}. These belong on the attempt manifest as "
            f"metadata and must NOT be folded into the hashed work-unit identity "
            f"(see core/storage/work_identity.py)."
        )


__all__ = [
    "DEVICE_METADATA_FIELDS",
    "assert_device_independent",
]
