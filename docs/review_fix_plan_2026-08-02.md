# Review Fix Plan — 2026-08-02

Executes the 41 confirmed findings of the deep review run after the
2026-08-01 campaign (7 lenses, 49 raw findings, 8 refuted by adversarial
verification), plus 4 findings confirmed independently.

Constraints unchanged: FP64 + NUFFT eps=1e-12 untouched; numerical gate is
max|diff| < 1e-10 Å on chunk site displacements; summation order is free;
local branch only, no push.

## The organising observation

The findings are not 41 unrelated defects. They are five repetitions of two
patterns, plus a tail of dead code:

1. **A declared identity key that nothing populates.** `_SCIENTIFIC_KEYS`
   reserves slots for the structure, the form-factor family, the
   coefficient knobs and `charge`. `structure_content_digest` was fixed in
   99ada1e; the rest still have no writer. Any of them changes the physics
   and none of them moves a digest.
2. **A reuse decision made by existence.** `ensure_*` helpers refuse to
   update an existing value; artifact loaders take a `parameter_digest`
   and ignore it; a snapshot that fails to load is skipped rather than
   raised.

Both patterns fail OPEN — they serve stale or partial data and publish it.
Every phase below is ordered so the fail-open cases close first.

## Phase 1 — identity completeness (7 findings, 3 HIGH)

The keys that decide whether stage-1 work may be reused.

| id | severity | finding |
|---|---|---|
| A1 | HIGH | `scattering_weights_digest` / `scattering_calculator_version` declared in `_SCIENTIFIC_KEYS` (planning.py:61) with no writer since introduction. With `coefficients.use=false` — the only mode where `ff_factory` is consulted — changing the form-factor family leaves every digest byte-identical and republishes the previous family's displacements. |
| A2 | HIGH | `use_coeff`, `coeff_center_by`, `chemical_filtered_ordering` and `charge` likewise absent from the scattering digest. `chemical_filtered_ordering` substitutes `average_coords` for `original_coords` in the amplitude adapter — a different sum — with no digest movement. |
| A3 | HIGH | **Regression introduced by 99ada1e.** `build_source_structure_digest` short-circuits on `structure_content_digest`; its fallback hashed the ADAPTER-modified `original_coords` and `coeff`, so supplying the stamp made the centered-coefficient contribution dead. The coefficients now enter no identity in either digest layer. |
| E1 | HIGH | **Gap in afc2dc9.** An interval that flips non-empty → mask-empty on a rerun leaves the previous run's `precomputed_intervals/interval_<id>.hdf5` in place; the residual stage folds it with no identity check. Same hole on the store-adoption path (execution.py:352-357). |
| E2 | LOW | **Gap in 99ada1e.** The point-data structure stamp covers only `FromAveragePointProcessor`; the `central` and `full_list` processors are unguarded. |
| E3 | LOW | The structure guard fails OPEN on an unreadable `structure_identity.json` — a corrupt record silently re-claims the directory. |
| E4 | LOW | `structure_content_digest_from_structure` uses `getattr(..., None)`, so a renamed field drops out of the identity silently — the precise failure mode that caused the original bug. |

**Approach.** Populate the reserved keys where `parameters` is assembled,
rather than adding new ones: the slots exist and their absence is the bug.
A3 needs the adapter-modified arrays back in an identity — either stop
short-circuiting or digest the adapter inputs explicitly. E3/E4 become
fail-closed.

**Gate.** Unit suite + host-regress. New E2E: change the form-factor family
and separately `coeff_center_by` in a reused directory; both must recompute
and match a fresh-directory run.

## Phase 2 — wrong numbers and silent corruption (4 findings, 4 HIGH)

| id | severity | finding |
|---|---|---|
| B1 | HIGH | `ensure_total_reciprocal_points` (artifacts.py:151) keeps any existing non-sentinel value, so a run with a larger reciprocal-space extent normalizes its residual by the PREVIOUS run's total. This is a wrong-number path, not a stale-file path. |
| B2 | LOW | `ensure_grid_shape` is the same anti-pattern; the decode path regenerates the grid, so it surfaces in merge-invariant checks rather than in the published numbers. |
| C1 | HIGH | The points-axis concat assembler (assembly.py:274-283) skips a partition whose snapshot will not load, publishes the truncated chunk as COMMITTED, then unlinks the dropped snapshot. Surfaces downstream as an opaque IndexError. |
| D1 | HIGH | `resolve_worker_scratch_root` builds a per-worker token from `get_worker()`, but is called on the DRIVER where that raises and the token degrades to `"local"`. Every worker on a node then shares one live-accumulator memmap path, and `_allocate_live_arrays` opens it `mode="w+"` (truncate) — a remapped owner truncates a live accumulator mid-fold. |
| F1 | HIGH | `load_existing_materialized_state` (artifacts.py:1037) accepts a `parameter_digest` and never uses it for lookup; both disk reads are keyed by chunk id alone, so a run with a different digest inherits the previous run's applied-interval set and never rewrites the residual payload. Measured: artifacts unchanged (mtime in the prior run's window) while a clean run differed by 7.2e+05. |

**Approach.** `ensure_*` writes when the value differs, not only when
absent. The assembler raises on an unloadable snapshot instead of
continuing. The scratch root resolves per-worker (on the worker). The
residual artifacts carry their parameter digest and refuse a mismatch —
the same stamp-and-validate shape as the payload identity.

**Gate.** Unit suite + host-regress + resume-check (Phase 2 touches the
credit path) + swap-check. New E2E: change a residual knob in a reused
directory; artifacts must be rewritten and match a fresh-directory run.

## Phase 3 — durability and degradation (5 findings, MEDIUM)

- Final residual chunk payload renamed without fsync while the manifest
  that credits it is fsynced (artifacts.py:203) — crash-ordering inversion.
- A transient `client.run()` failure downgrades a multi-node run to
  "single host", skipping the cross-host and file-lock probes
  (fs_capability.py:137), and the fail-closed lock probe cannot detect the
  `nolock` mounts it names (fs_capability.py:203).
- Streaming resume credit granted on artifact EXISTENCE with no binding to
  the credited intervals (run_state_cache.py:512).
- Per-chunk scattered atom recarrays have no re-scatter recovery, unlike
  the streaming context (execution.py:1861).
- Non-streaming finalize marks the shared SQLite cache from N workers
  concurrently, after the durable commit (execution.py:2049).

## Phase 4 — legacy and dead code (20 findings, LOW)

Dead modules (`scattering/precompute.py`, `scattering/chunk_processing.py`,
`decoding/mode.py`, `qspace/masking/shape_gpu.py`, `storage/contracts.py`,
`storage/hdf5_data_storage.py`); nine dead re-exports in
`scattering/runtime.py` left by the deleted stage2_replacement; a second
atomic-JSON writer in `decoder_cache.py` bypassing the consolidated one
(no fsync); an unreachable legacy `.npz` store reader whose docstring
claims the opposite; three unreachable JSON schemas, one requiring a field
no manifest carries; dead resume/completeness predicates from the deleted
layouts; `MOSAIC_NUFFT_CPU_ONLY` parsed three ways; contract specs with no
enforcement point; four manifest validators that can `return None` with the
suite still green; `.gitignore` misses the 3D demo run trees, and tracked
demo configs hard-code one developer's absolute paths; docs listing as open
five items the campaign implemented.

## Refuted (8)

Recorded so they are not re-raised: the adversarial pass rejected them as
unreachable, already fixed by the campaign, or misread. Details in the
workflow journal.

## Gates

Full unit suite + host-regress.sh (small_all + small_sphere, PASS < 1e-10 Å)
after every phase. Phases 1-2 additionally run resume-check.sh (kill
mid-residual, resume, run 3 must resubmit zero batches) and swap-check.sh
(the structure guard must still refuse and fresh_start must still rebuild
correctly).
