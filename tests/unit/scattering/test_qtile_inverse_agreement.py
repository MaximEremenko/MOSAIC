"""Numerical viability proof that q-tiling the inverse NUFFT is sound.

Q-tiling splits the reciprocal q_grid into contiguous sub-ranges. Because the inverse
(type-3) NUFFT SUMS over q-points, the full inverse result equals the SUM of the per-tile
partials -- but the per-tile-then-sum path uses a DIFFERENT floating-point reduction order,
so it agrees with the un-tiled result only within the PREDICTED forward-error tolerance
(``core.storage.agreement.predict_agreement_rtol`` / ``relative_l2``), never bitwise.

This test proves that agreement holds within the project's OWN agreement gate, so a future
(separately science-signed-off) increment can drive execution with q-tiling. It uses the
EXISTING kernel only -- ``execute_inverse_cunufft_batch_materialize_once`` from
``core.adapters.cunufft_wrapper`` -- with the SAME kwargs the production task uses
(``core/scattering/tasks.py::compute_scattering_interval_chunk_arrays``: a stacked
``[delta, average]`` weight pair, ``eps``, ``prefer_cpu``). It edits no kernel / hot path and
wires nothing into execution; it only CALLS the kernel and SUMS partials in the test body.

Determinism: forced CPU (``MOSAIC_NUFFT_CPU_ONLY=1`` + ``prefer_cpu=True``) so finufft's
single-threaded type-3 path gives a stable reduction; the agreement assertion is the hard
gate and the not-bitwise observation is soft (only asserted when K>1 actually changes the
reduction order, which it does for this problem).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.storage.agreement import (
    cancellation_kappa,
    predict_agreement_rtol,
    relative_l2,
)
from core.scattering.work_unit_tiling import plan_qspace_tiles


# Force the deterministic CPU path BEFORE importing the wrapper so the module-level
# CPU-only switch is read at import. Mirrors the import-cycle test's harness.
@pytest.fixture(scope="module")
def cunufft_cpu(monkeypatch_module):
    monkeypatch_module.setenv("MOSAIC_NUFFT_CPU_ONLY", "1")
    from core.adapters import cunufft_wrapper

    cunufft_wrapper.set_cpu_only(True)
    yield cunufft_wrapper
    # set_cpu_only(True) flips process-wide globals (_GPU_AVAILABLE, cp);
    # mp.undo() only restores the env var, so re-probe the GPU backend here
    # or every later GPU wrapper test in the session fails.
    cunufft_wrapper.set_cpu_only(False)


@pytest.fixture(scope="module")
def monkeypatch_module():
    # module-scoped monkeypatch (pytest's built-in `monkeypatch` is function-scoped).
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _build_problem(*, dim: int, n_q: int, n_real: int, seed: int):
    """Deterministic small problem: a reciprocal q_grid, complex128 weights for the two
    channels (delta = q_amp - q_amp_av, and q_amp_av), and seeded real-space coords.

    Coordinate magnitudes are kept small (a few cycles) so the type-3 transform is
    well-conditioned and the eps-bounded agreement is exercised, not blow-up.
    """
    rng = np.random.default_rng(seed)
    q_grid = rng.uniform(-3.0, 3.0, size=(n_q, dim)).astype(np.float64)
    real_coords = rng.uniform(-2.0, 2.0, size=(n_real, dim)).astype(np.float64)

    # Two physical channels mirroring the production weights: q_amp and q_amp_av; the
    # transformed channels are delta = q_amp - q_amp_av and the average q_amp_av.
    q_amp = (rng.standard_normal(n_q) + 1j * rng.standard_normal(n_q)).astype(np.complex128)
    # Small displacement regime: q_amp_av close to q_amp so ||delta|| << ||average|| and the
    # delta channel's cancellation amplification (kappa) is exercised (matches the science).
    q_amp_av = q_amp + 1e-3 * (
        rng.standard_normal(n_q) + 1j * rng.standard_normal(n_q)
    ).astype(np.complex128)
    delta = q_amp - q_amp_av
    weights = np.stack([delta, q_amp_av], axis=0)  # (2, n_q): [delta, average]
    return q_grid, real_coords, weights


@pytest.mark.parametrize("dim", [2, 3])
def test_qtile_inverse_sum_agrees_within_predicted_tolerance(cunufft_cpu, dim, capsys):
    eps = 1e-12
    n_q = 384
    n_real = 200
    q_grid, real_coords, weights = _build_problem(
        dim=dim, n_q=n_q, n_real=n_real, seed=20260630 + dim
    )
    accepted_count = int(q_grid.shape[0])

    common = dict(
        real_coords=real_coords,
        eps=eps,
        prefer_cpu=True,
    )

    # 1) UN-TILED inverse on the FULL q_grid (same call shape as the production task).
    untiled = cunufft_cpu.execute_inverse_cunufft_batch_materialize_once(
        q_coords=q_grid,
        weights=weights,
        **common,
    )
    untiled = np.asarray(untiled, dtype=np.complex128)
    assert untiled.shape == (2, n_real)

    # 2) Split the q_grid into K tiles. Force a small byte budget so K > 1: budget is a
    #    fraction of the whole q-input so we get several tiles.
    whole_qbytes = accepted_count * dim * 8
    byte_budget = max(1, whole_qbytes // 5)
    tiles = plan_qspace_tiles(
        accepted_count=accepted_count, dim=dim, byte_budget=byte_budget
    )
    assert len(tiles) > 1  # the proof is vacuous if we don't actually split
    # exact, contiguous, non-overlapping cover of [0, accepted_count)
    assert tiles[0][0] == 0 and tiles[-1][1] == accepted_count
    assert sum(stop - start for start, stop in tiles) == accepted_count

    # 3) Run the SAME inverse per tile on the q-sub-grid + weight-sub-columns and SUM.
    tiled = np.zeros_like(untiled)
    for start, stop in tiles:
        partial = cunufft_cpu.execute_inverse_cunufft_batch_materialize_once(
            q_coords=q_grid[start:stop],
            weights=weights[:, start:stop],
            **common,
        )
        np.add(tiled, np.asarray(partial, dtype=np.complex128), out=tiled)

    # 4) Assert per-channel agreement within the PREDICTED tolerance, mirroring
    #    commit.py::_assert_scattering_partials_agree: kappa=1 for the average channel and
    #    cancellation_kappa(avg, delta) for the delta channel; M = accepted_count.
    avg_untiled = untiled[1]
    delta_untiled = untiled[0]
    avg_tiled = tiled[1]
    delta_tiled = tiled[0]

    rtol_avg = predict_agreement_rtol(
        eps=eps, summation_terms=accepted_count, kappa=1.0
    )
    rtol_delta = predict_agreement_rtol(
        eps=eps,
        summation_terms=accepted_count,
        kappa=cancellation_kappa(avg_untiled, delta_untiled),
    )

    rel_avg = relative_l2(avg_untiled, avg_tiled)
    rel_delta = relative_l2(delta_untiled, delta_tiled)

    # Print measured rel-L2 vs the tolerance bound for the report.
    with capsys.disabled():
        print(
            f"\n[qtile-agreement dim={dim} K={len(tiles)} M={accepted_count}] "
            f"average: rel_l2={rel_avg:.3e} <= rtol={rtol_avg:.3e} | "
            f"delta: rel_l2={rel_delta:.3e} <= rtol={rtol_delta:.3e} "
            f"(kappa_delta={cancellation_kappa(avg_untiled, delta_untiled):.3e})"
        )

    assert rel_avg <= rtol_avg, (
        f"average channel rel-L2 {rel_avg:.3e} exceeds predicted tolerance "
        f"{rtol_avg:.3e} (eps={eps:.1e}, M={accepted_count})"
    )
    assert rel_delta <= rtol_delta, (
        f"delta channel rel-L2 {rel_delta:.3e} exceeds predicted tolerance "
        f"{rtol_delta:.3e} (eps={eps:.1e}, M={accepted_count})"
    )

    # SOFT observation (documented, not a hard gate): q-tiling changes the reduction order,
    # so the tiled sum is generally NOT bitwise-identical to the un-tiled result. We only
    # check this when K>1 actually split the sum; if a problem happened to be bitwise-equal
    # we do NOT fail (the hard gate is the agreement-within-tolerance check above).
    if len(tiles) > 1:
        bitwise_equal = np.array_equal(untiled, tiled)
        if bitwise_equal:
            # Not a failure: just record that this instance happened to reduce identically.
            with capsys.disabled():
                print(
                    f"[qtile-agreement dim={dim}] note: tiled sum was bitwise-identical "
                    "to the un-tiled result for this instance (still within tolerance)."
                )
