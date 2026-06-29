from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Literal, cast


NufftExecutionPolicy = Literal[
    "auto",
    "gpu-required",
    "cpu-only",
    "allow-fallback",
]

VALID_NUFFT_POLICIES: tuple[NufftExecutionPolicy, ...] = (
    "auto",
    "gpu-required",
    "cpu-only",
    "allow-fallback",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_GPU_LAUNCH_WORKER_COMMANDS = {"dask-cuda-worker"}
_COMPLEX128_ALIASES = {
    "complex128",
    "complex",
    "complex_",
    "cdouble",
    "np.complex128",
    "numpy.complex128",
    "<c16",
}


# A device-bound identity may only be stamped when the device is *known* to
# have run the work:
#   * ``cpu``  — CPU was enforced (``cpu-only``/``auto``), so CPU bytes are
#     definitively CPU bytes.
#   * ``cuda`` — GPU was enforced with no fallback (``gpu-required``), so the
#     bytes are definitively GPU bytes (the wrapper raises rather than fall
#     back under ``gpu_only``).
# ``allow-fallback`` is *not* device-bound at resolve time: the wrapper may
# silently fall back to CPU, so the realized device is unknown until execution
# completes.  We represent that as the sentinel ``"pending"`` backend, which
# carries a conservative recorded identity of ``cpu`` (see ``backend``) so that
# CPU bytes can never be mis-recorded under a ``cuda`` identity if Phase C has
# not yet stamped the realized device.  Phase C MUST call
# ``with_realized_backend(...)`` with the device that actually ran before the
# identity is committed; only then may a ``cuda`` identity appear for a
# fallback-eligible run.
RealizedBackend = Literal["cpu", "cuda"]
_BackendState = Literal["cpu", "cuda", "pending"]


@dataclass(frozen=True)
class NufftExecutionSettings:
    requested_policy: NufftExecutionPolicy
    execution_policy: NufftExecutionPolicy
    # ``backend_state`` is the resolve-time intent: ``"pending"`` means the
    # realized device is not yet known (fallback-eligible).  ``backend`` (a
    # property) is what gets recorded into identity and is always a concrete,
    # truthful device — never ``cuda`` unless GPU execution is guaranteed or
    # has been confirmed via ``with_realized_backend``.
    backend_state: _BackendState
    eps: float
    dtype: Literal["complex128"]
    deterministic_mode: str
    thread_count: int | None
    # Whether the wrapper should *attempt* GPU first.  Decoupled from the
    # recorded identity: ``allow-fallback`` attempts GPU (``True``) but may
    # realize as CPU, so its recorded backend is not forced to ``cuda``.
    attempt_gpu: bool = False

    @property
    def realized_backend_known(self) -> bool:
        """True iff the recorded ``backend`` is a confirmed realized device."""
        return self.backend_state != "pending"

    @property
    def backend(self) -> RealizedBackend:
        """Device identity to record.

        Device-bound when known (``cpu``/``cuda``).  For a still-``pending``
        fallback-eligible run we record the conservative ``cpu`` so CPU bytes
        are never recorded under ``cuda`` before Phase C confirms the realized
        device via :meth:`with_realized_backend`.
        """
        if self.backend_state == "cuda":
            return "cuda"
        return "cpu"

    def with_realized_backend(self, realized: RealizedBackend) -> "NufftExecutionSettings":
        """Stamp the device that actually executed the work (Phase C seam).

        Only meaningful for a ``pending`` (fallback-eligible) resolution.  For
        an already device-bound resolution the realized device must match the
        bound device, otherwise the policy was violated (e.g. ``gpu-required``
        somehow ran on CPU) and we fail closed rather than mis-record identity.
        """
        if realized not in ("cpu", "cuda"):
            raise ValueError("realized backend must be 'cpu' or 'cuda'.")
        if self.backend_state != "pending" and self.backend_state != realized:
            raise ValueError(
                "realized backend "
                f"{realized!r} contradicts device-bound policy backend "
                f"{self.backend_state!r}; cannot record identity under a device "
                "that did not run the work."
            )
        if self.backend_state == realized:
            return self
        return NufftExecutionSettings(
            requested_policy=self.requested_policy,
            execution_policy=self.execution_policy,
            backend_state=realized,
            eps=self.eps,
            dtype=self.dtype,
            deterministic_mode=self.deterministic_mode,
            thread_count=self.thread_count,
            attempt_gpu=self.attempt_gpu,
        )

    @property
    def prefer_cpu(self) -> bool:
        return not self.attempt_gpu

    @property
    def gpu_only(self) -> bool:
        return self.attempt_gpu

    @property
    def execute_kwargs(self) -> dict[str, bool]:
        return {
            "prefer_cpu": self.prefer_cpu,
            "gpu_only": self.gpu_only,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "requested_policy": self.requested_policy,
            "execution_policy": self.execution_policy,
            "backend": self.backend,
            "realized_backend_known": self.realized_backend_known,
            "eps": float(self.eps),
            "dtype": self.dtype,
            "deterministic_mode": self.deterministic_mode,
            "thread_count": self.thread_count,
        }


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    raw = env.get(name)
    return raw is not None and str(raw).strip().lower() in _TRUE_VALUES


def _env_int(env: Mapping[str, str], name: str, default: int = 0) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def normalize_nufft_policy(value: object) -> NufftExecutionPolicy:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in VALID_NUFFT_POLICIES:
        return cast(NufftExecutionPolicy, normalized)
    valid = ", ".join(VALID_NUFFT_POLICIES)
    raise ValueError(f"NUFFT execution policy must be one of: {valid}.")


def normalize_nufft_dtype(value: object | None = None) -> Literal["complex128"]:
    raw = "complex128" if value is None else str(value).strip().lower()
    raw = raw.replace("'", "").replace('"', "")
    if raw in _COMPLEX128_ALIASES:
        return "complex128"
    raise ValueError(
        "MOSAIC durable NUFFT execution currently supports dtype=complex128 only."
    )


def gpu_launch_requested(*, env: Mapping[str, str] | None = None) -> bool:
    environment = _env(env)
    backend = str(environment.get("DASK_BACKEND", "")).strip().lower()
    worker_command = str(environment.get("DASK_WORKER_COMMAND", "")).strip().lower()
    return (
        backend == "cuda-local"
        or worker_command in _GPU_LAUNCH_WORKER_COMMANDS
        or _env_int(environment, "GPUS_PER_JOB", 0) > 0
    )


def resolve_nufft_policy(
    requested: object | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> NufftExecutionPolicy:
    """Resolve NUFFT execution policy with runtime environment precedence.

    Precedence:
      1. ``MOSAIC_NUFFT_CPU_ONLY=1`` forces ``cpu-only``.
      2. Explicit caller policy, when supplied.
      3. ``MOSAIC_NUFFT_EXECUTION_POLICY``.
      4. GPU launch environment implies ``gpu-required``.
      5. Default ``auto``.
    """
    environment = _env(env)
    if _env_truthy(environment, "MOSAIC_NUFFT_CPU_ONLY"):
        return "cpu-only"
    if requested is not None:
        return normalize_nufft_policy(requested)
    raw_policy = environment.get("MOSAIC_NUFFT_EXECUTION_POLICY")
    if raw_policy is not None and str(raw_policy).strip():
        return normalize_nufft_policy(raw_policy)
    if gpu_launch_requested(env=environment):
        return "gpu-required"
    return "auto"


def _env_thread_count(env: Mapping[str, str]) -> int | None:
    for name in ("MOSAIC_NUFFT_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def resolve_nufft_execution_settings(
    requested: object | None = None,
    *,
    eps: object = 1e-12,
    dtype: object | None = "complex128",
    env: Mapping[str, str] | None = None,
) -> NufftExecutionSettings:
    environment = _env(env)
    policy = resolve_nufft_policy(requested, env=environment)
    # ``backend_state`` is the resolve-time device intent; ``attempt_gpu`` is
    # whether the wrapper tries GPU first.  Only ``cpu-only``/``auto`` (CPU
    # enforced) and ``gpu-required`` (GPU enforced, no fallback) may bind a
    # device identity at resolve time.  ``allow-fallback`` resolves to
    # ``pending`` because the wrapper may silently fall back to CPU; the
    # realized device must be stamped post-execution via
    # ``with_realized_backend`` (Phase C) before identity is committed.
    if policy == "auto":
        backend_state: _BackendState = "cpu"
        execution_policy: NufftExecutionPolicy = "cpu-only"
        attempt_gpu = False
    elif policy == "cpu-only":
        backend_state = "cpu"
        execution_policy = "cpu-only"
        attempt_gpu = False
    elif policy == "gpu-required":
        # GPU enforced with no fallback: the wrapper raises rather than fall
        # back under ``gpu_only``, so a ``cuda`` identity is definitively true.
        backend_state = "cuda"
        execution_policy = "gpu-required"
        attempt_gpu = True
    elif policy == "allow-fallback":
        # Fallback-eligible: attempt GPU but DO NOT bind a ``cuda`` identity.
        # The wrapper may fall back to CPU; the realized device is unknown
        # until execution completes.  Recorded backend stays the conservative
        # ``cpu`` until Phase C confirms the realized device.
        backend_state = "pending"
        execution_policy = "allow-fallback"
        attempt_gpu = True
    else:  # pragma: no cover - normalize_nufft_policy keeps this closed.
        raise ValueError(f"Unsupported NUFFT policy: {policy!r}")
    try:
        resolved_eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("NUFFT eps must be a finite float.") from exc
    if not (resolved_eps > 0.0):
        raise ValueError("NUFFT eps must be positive.")
    return NufftExecutionSettings(
        requested_policy=policy,
        execution_policy=execution_policy,
        backend_state=backend_state,
        eps=resolved_eps,
        dtype=normalize_nufft_dtype(dtype),
        deterministic_mode="stable-v1",
        thread_count=_env_thread_count(environment),
        attempt_gpu=attempt_gpu,
    )


def nufft_execute_kwargs(policy: object) -> dict[str, bool]:
    resolved = normalize_nufft_policy(policy)
    return {
        "prefer_cpu": resolved == "cpu-only",
        "gpu_only": resolved in {"gpu-required", "allow-fallback"},
    }


def nufft_task_retries(policy: object, default_retries: int) -> int:
    resolved = normalize_nufft_policy(policy)
    if resolved in {"gpu-required", "allow-fallback"}:
        return 0
    return int(default_retries)


def is_nufft_gpu_resource_failure(error: object) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "cuda",
            "cudart",
            "cufft",
            "cufinufft",
            "cupy",
            "gpu execution forced",
            "no cuda device",
            "out of memory",
            "memory allocation",
            "cuda_error_out_of_memory",
            "cudaerroroutofmemory",
            "insufficient resources",
            "too many resources requested",
            "launch-resource-exhausted",
            "budget",
            "driver shutting down",
            "device-side assert",
            "illegal memory access",
        )
    )


def should_resubmit_cpu_fallback(
    policy: object,
    failure: object,
    *,
    already_resubmitted: bool,
) -> bool:
    return (
        normalize_nufft_policy(policy) == "allow-fallback"
        and not bool(already_resubmitted)
        and is_nufft_gpu_resource_failure(failure)
    )


__all__ = [
    "NufftExecutionSettings",
    "NufftExecutionPolicy",
    "RealizedBackend",
    "VALID_NUFFT_POLICIES",
    "gpu_launch_requested",
    "is_nufft_gpu_resource_failure",
    "normalize_nufft_dtype",
    "normalize_nufft_policy",
    "nufft_execute_kwargs",
    "nufft_task_retries",
    "resolve_nufft_execution_settings",
    "resolve_nufft_policy",
    "should_resubmit_cpu_fallback",
]
