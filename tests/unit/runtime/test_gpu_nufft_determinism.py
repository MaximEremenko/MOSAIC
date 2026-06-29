"""GPU cuFINUFFT type-3 determinism characterization (real-GPU only).

These tests run ONLY when a CUDA device + cupy + cufinufft are present; they
skip cleanly otherwise so the suite stays green on CPU-only CI.

Empirical finding (RTX 5000 Ada, cc 8.9, cupy 14.0.1, cuFINUFFT 2.5.1):
the cuFINUFFT type-3 transform is **irreducibly non-bit-reproducible** at
production point counts, because the spread step accumulates colliding points
with ``atomicAdd`` whose float summation order depends on warp scheduling.
No combination of gpu_method / gpu_sort / gpu_kerevalmeth / gpu_maxsubprobsize
makes it bit-identical across launches.

Consequences for MOSAIC:
* ``_select_attempts_by_interval`` (core/scattering/commit.py:618-622) requires
  a single ``payload_sha256`` per interval — this is unsatisfiable for two
  honest GPU attempts of the same work unit.
* The fix is per-(device-class, driver, cuFINUFFT-version) identity plus a
  tolerance-based agreement check, NOT bit-equality (see
  scratchpad/gpu_determinism_results.md).

The "production scale is bit-reproducible" expectation is therefore marked
``xfail(strict=True)``: it documents the known non-determinism and will turn
the suite RED (alerting us) if cuFINUFFT ever becomes deterministic here, at
which point the exact-payload model could be revisited.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
cufinufft = pytest.importorskip("cufinufft")


def _gpu_present() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_present(), reason="No CUDA device available"
)


# Fixed, explicit plan params — the whole point is that fixing them does NOT
# make the transform bit-reproducible at scale.
_PLAN_KW = dict(gpu_method=1, gpu_sort=1, gpu_kerevalmeth=1, gpu_maxsubprobsize=1024)


def _make_type3(seed: int, n_src: int, n_tgt: int):
    rng = np.random.default_rng(seed)
    src = ((rng.random((n_src, 3)) - 0.5) * 2 * np.pi).astype(np.float64)
    tgt = ((rng.random((n_tgt, 3)) - 0.5) * 50.0).astype(np.float64)
    w = (rng.standard_normal(n_src) + 1j * rng.standard_normal(n_src)).astype(
        np.complex128
    )
    return src, tgt, w


def _run_once(src, tgt, w, *, eps=1e-12, plan_kw=None):
    """One fresh-plan type-3 inverse (isign=-1) transform; returns host bytes."""
    plan_kw = _PLAN_KW if plan_kw is None else plan_kw
    d_src = [cp.ascontiguousarray(cp.asarray(src[:, i])) for i in range(3)]
    d_tgt = [cp.ascontiguousarray(cp.asarray(tgt[:, i])) for i in range(3)]
    d_w = cp.ascontiguousarray(cp.asarray(w))
    plan = cufinufft.Plan(
        3, 3, n_trans=1, eps=eps, isign=-1, dtype="complex128", **plan_kw
    )
    try:
        plan.setpts(d_src[0], d_src[1], d_src[2], d_tgt[0], d_tgt[1], d_tgt[2])
        out = plan.execute(d_w)
        cp.cuda.runtime.deviceSynchronize()
        host = cp.asnumpy(out)
    finally:
        try:
            plan._destroy_plan(plan._plan)
            plan._plan = None
        except Exception:
            pass
        cp.get_default_memory_pool().free_all_blocks()
    return host


def _bit_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    )


def test_gpu_type3_runs_on_device_and_matches_cpu():
    """Runnability: the GPU path executes on the device and is numerically
    correct (rel-L2 well below 1e-9 at eps=1e-12)."""
    finufft = pytest.importorskip("finufft")
    src, tgt, w = _make_type3(seed=0, n_src=2000, n_tgt=1500)
    gpu = _run_once(src, tgt, w)
    cpu = finufft.nufft3d3(
        src[:, 0], src[:, 1], src[:, 2], w,
        tgt[:, 0], tgt[:, 1], tgt[:, 2], eps=1e-12, isign=-1,
    )
    assert gpu.dtype == np.complex128 and gpu.shape == (1500,)
    rel = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    assert rel < 1e-9, f"GPU vs CPU rel-L2 too large: {rel:.3e}"


def test_gpu_type3_tiny_is_bit_reproducible():
    """Positive control: with few, well-separated points there are no spread
    collisions, so the transform IS bit-identical across fresh-plan launches."""
    src, tgt, w = _make_type3(seed=0, n_src=8, n_tgt=4)
    ref = _run_once(src, tgt, w)
    for _ in range(5):
        assert _bit_equal(ref, _run_once(src, tgt, w))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "cuFINUFFT type-3 is irreducibly non-bit-reproducible at production "
        "scale (atomic-add spread collisions). If this ever PASSES, cuFINUFFT "
        "became deterministic here and the exact-payload commit model can be "
        "revisited. See scratchpad/gpu_determinism_results.md."
    ),
)
def test_gpu_type3_production_scale_is_bit_reproducible():
    """Same inputs + same fixed plan params, repeated. Expected to FAIL
    (xfail-strict) because the GPU transform is non-deterministic at scale."""
    src, tgt, w = _make_type3(seed=7, n_src=200_000, n_tgt=160_000)
    ref = _run_once(src, tgt, w)
    for _ in range(3):
        assert _bit_equal(ref, _run_once(src, tgt, w)), (
            "GPU transform was NOT bit-identical across launches"
        )
