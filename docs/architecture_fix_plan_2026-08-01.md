# Architecture Fix Campaign — 2026-08-01

Executes the 48 confirmed findings of `docs/architecture_review_2026-07-31.md`.
Constraints carried over: FP64 (complex128) + NUFFT eps=1e-12 untouched; numerical
gate is max|diff| < 1e-10 Å on chunk site displacements (host-regress.sh) after
every phase; summation order is free; local branch only, no push.

## Outcome (all six phases complete)

| Phase | Commit | Unit suite | Gate (max diff vs reference) |
|---|---|---|---|
| 1 correctness seams | `73abf73` | pass | 4.6e-15 |
| 2 deployment/resources | `ed4d823` | pass | 3.7e-15 |
| 3 identity break (digest v3) | `e453e12` | pass | 4.1e-15 + kill/resume |
| 4 deletion/consolidation/layering | `085fb82`, `6ac5245` | 1500 | 5.4e-15 |
| 5 durable_shared retirement | `ca334d4` | 1475 | 3.7e-15 + 3-node sim 1.7e-12 |
| 6 god-module split | `2a16884` | 1485 | 4.0e-15 |

All 48 findings fixed (M14 followed the six phases — see below). Net ≈ −5,900 lines. Module sizes after phase 6:
`execution.py` 2731 → 2127, `backend.py` 3108 → 1946, with `run_loop.py` (816),
`assembly.py` (509) and `snapshot_writer.py` (123) carrying the extracted parts.

Post-campaign follow-ups (same branch):

- **M14** — the recorded non-fix was subsequently done (`f941373`): one
  codec (`scattering/interval_payload.py`) owns the stage-1 payload format
  for both durable modes, so the streaming store inherits the artifact
  writer's fsync'd commit protocol and either mode's writer produces a file
  the other's reader accepts. Legacy `.npz` store entries stay readable.
- **Multi-node GPU** — what this plan first recorded as an environmental
  sim issue turned out to be a real bug on the documented multi-node MPI
  path (`f917927`): stage-1 payloads were memory-mapped from the SHARED
  filesystem and handed to CUDA, whose DMA from network-faulting pages
  fails (`cudaErrorDevicesUnavailable` on every GPU fold). Payloads on a
  network filesystem are now read materialized. The original failing
  configuration — 6 ranks, 3 nodes, 4 GPUs, NFS store — completes clean and
  matches the host reference at 4.9e-15.
- **Cross-mode stage-1 reuse** — the last deliberate non-fix is now done.
  Every payload carries a `payload_identity` stamp (see below), so each
  mode can reuse what the other left on disk: streaming reads
  `precomputed_intervals/` before recomputing, and precompute adopts
  `stage1_payload_store/<payload_identity>/` entries as its own artifacts.
  A payload whose identity cannot be PROVEN to match is recomputed, never
  served. Verified end to end on CaTiO3-small (320 intervals), each phase
  reusing the previous one's output directory:

  | phase | mode | stage-1 transforms | evidence |
  |---|---|---|---|
  | A | streaming | 320 | store filled |
  | B | precompute, A's store | **0** | "Adopted 320 stage-1 payload(s)"; 0 written, 320 cached |
  | C | streaming, B's artifacts | **0** | its own store stayed empty |
  | D | streaming, B's artifacts, ONE atom moved | 320 | store filled — artifacts refused |

  B and C match the reference at 5.940e-15. D lands 1.317e-03 away from it,
  which is the point: the physics really did change, and serving B's
  payloads would have returned the reference answer for a structure that no
  longer produces it.

## Stage-1 payload identity

`precomputed_intervals/interval_<id>.hdf5` is keyed by interval id alone.
Every run sharing an output directory writes the same paths, and
`fresh_start` defaults to False — so reuse across runs is the default, and
file existence has never been evidence that the current run produced the
file. Neither is the SQLite `precomputed` column: it is not digest-scoped
and no rebuild resets it.

The stamp is therefore in the FILE:

    payload_identity = H(scientific_digest, source_structure_digest,
                         eps, dtype, pre_sum_mode)

which is what a forward transform actually depends on. `run_digest` cannot
serve here, in both directions:

- **It carries too much.** `run_digest` is built with `reducer_strategy`,
  which is `"stage2-streaming"` in streaming mode and `"attempt-commit"`
  in precompute mode. The same CaTiO3 case measured `1760a0ae…` streaming
  vs `cb86ad49…` precompute — so the two modes addressed different store
  directories and could never have reused each other's payloads whatever
  the stamp said. The reducer consumes stage-1 output; it does not change
  it, so it is excluded.
- **It carries too little.** `run_digest` is **invariant to the atomic
  coordinates**: nothing in the pipeline populates the
  `structure_content_digest` / `structure_file_sha256` / `structure_digest`
  keys its scientific payload hashes. Measured, not inferred: while the
  store leaf was still the run digest, phase D — one atom displaced by 0.01
  fractional units, a change worth 1.3e-03 Å in the output — resolved to
  the SAME leaf `1760a0ae…` as the unperturbed phase A. Two structures,
  one store directory. Under the payload identity they separate
  (`5873b359…` vs `47f2d59b…`).

`pre_sum_mode` reaches no stage-1 code today; it is folded in anyway,
because a false miss costs one recompute and a false hit is wrong physics.

The store's directory leaf moved from `run_digest` to this identity for
the same two reasons: a run-digest leaf split the modes apart and merged
two structures together. Existing stores under the old leaf go cold and
recompute — they are a cache, and the recompute is what correctness costs.

Readers apply it by what their directory can prove:

| reader | directory proves identity | unstamped file |
|---|---|---|
| `stage1_payload_store/<payload_identity>/` | yes (identity leaf) | accepted |
| `precomputed_intervals/` | no (shared path) | refused |

A contradicting stamp is refused in both. The store still checks the stamp
it does not need, because `MOSAIC_STREAMING_PAYLOAD_STORE` can relocate the
base onto a path whose scoping is the operator's problem.

## Phase order and rationale

1. **Correctness seams** (residual core, sequential edits — the two god-files):
   - H9 barrier rescue: extend bounded-wait + dead-owner remap to the flush /
     inspect / finalize barriers (flush remap = no-op → coverage validation
     fail-stops loudly; inspect/finalize remap to any live worker).
   - L15 all-workers-dead horizon: abort the drain loop after a wall-clock
     horizon with zero live workers instead of spinning on 45 s timeouts.
   - H1 opportunistic-recovery guard: defer finalize of NON-partitioned
     families whose interval coverage cannot be proven (mirror the partitioned
     deferral) so a partial owner-level chunk is never published COMMITTED.
   - M2 finalize takes the per-target lock around live-accumulator snapshots.
   - M1 fsync snapshot payloads before rename; snapshot loaders treat a torn
     (unreadable) file as absent, matching the stage-1 store's self-healing.
   - M21 submission interleave ordered by resolved owner, not slot, so
     placement stays spread when worker count divides chunk count.
   - M24 set dask `retries=0` unconditionally (driver owns retry); cap
     repeated KilledWorker of the SAME key below the infra budget; document
     the post-nanny-restart `gpu_tripped` inconsistency.
   - L2/M22 delete dead reducer/backend code and the non-owner-local else
     branches guarded alive by their own removal raises.
   - H8 async snapshot writer works on file-mode accumulators (capture from
     the memmap view, no copy) — the hkl40 case it was built for.

2. **Deployment / resource layer** (file-disjoint from phase 1 targets):
   - H2 backend-correct job directives (SLURM/PBS/LSF/SGE) + concat-merge of
     job_extra_directives so the GPU directive survives.
   - H7 register heap-trim plugin + worker env on job-queue backends.
   - H6 backpressure probes host truth (available_memory_bytes via client.run)
     instead of the deliberately-zero dask memory_limit.
   - M6 `max_workers: auto` resolves per backend (mpi: skip/world-size;
     job-queue: SLURM grant) instead of the literal 4.
   - M7+L13 RAM/VRAM divisors use processes-on-THIS-host (OMPI/SLURM local
     size, else scheduler_info grouped by host); promote the dead
     `_same_node_local_nufft_capacity` into that role.
   - M19 pool-cap divisor keyed on processes-sharing-this-device, not
     DASK_PROCESSES.
   - M18 tile sizing after `_transform_enter()` from `_current_tile_budget()`.
   - L12 VRAM floors relative (tile floor clamped to reservable pool).
   - M17 lattice cache budget scales with host RAM / worker count; cleared at
     stage boundary; evicts under host pressure.
   - M20 shadowed knobs: one prefetch name (old one warns), split
     MOSAIC_STREAMING_STAGE1_PARALLEL consumers, one threads default.
   - M5 local_restartable scratch defaults to the dask worker local_directory,
     shared output dir only for durable state.

3. **Checkpoint-identity break — snapshot layout v2** (one deliberate break,
   bundling everything the review said to do "at the next layout break"):
   - H3 identity knobs (subchunk slot count, shard grid budget,
     intervals-per-shard, lattice-FFT switch) enter the parameter digest;
     a changed knob now lands in a DISJOINT family (recompute, non-destructive)
     instead of deleting checkpoints. Budget stays a host-independent constant
     (review L14: do not host-scale it).
   - M23 slot assignment becomes rank-in-canonical-order mod n_slots (kills
     the sha256 binomial skew that concentrated the rod case in one slot).
   - M4 non-streaming partition plan derives from a config constant, not live
     scheduler capacity → durable-mode resume survives GPU-count changes.
   - L5 owner-epoch fencing in the snapshot key + manifest union rejects
     superseded epochs (dask-mpi zombie-owner window).
   - L3 resume-credit scan uses the typed manifest loader and re-derives
     artifact paths from output_dir (relocation-safe credits).
   - L4 fix the sbatch store-sharing overclaim (store is mask-scoped).

4. **Redundancy deletion**:
   - H5 delete legacy pair execution + stage2_replacement (config-dead, own
     scheduler/committer, 8 cross-package private imports disappear with it).
   - M11 scattering/persistence.py + orphaned artifact functions; M12
     compute_streamed_interval_tasks (tests port to lazy loaders); M13
     local-window inverse (~250 unvalidated lines); M16 pre-tile-owner
     launcher .sh scripts; L10 test-only lattice helpers relocated to tests.
   - M15 inspect.signature probes → direct calls (typos raise again).
   - M9/L6 layering: ScatteringIntervalArtifactPolicy → scattering,
     is_same_node_local_client → core/runtime, _residual_work_unit_digest →
     planning; one _current_worker_addresses.
   - L8 BudgetedLRU + bounded_prefetch replace 4 hand-rolled caches and 2
     window idioms; L9 one atomic-JSON writer (storage/atomic), one env-parser
     home, one fsync pair; L11 DatabaseManager Protocol replaces hasattr
     probing.

5. **durable_shared retirement (M10)** — the review's own path: validate a
   non-streaming multi-node run on local_restartable in the docker cluster
   sim, flip the locality-heuristic default, then delete SHARED_DURABLE_LAYOUT
   and the generation-checkpoint machinery (~17 forks). If validation fails,
   stop at "no longer the default" and keep the branch explicitly configured.

6. **Structural, pure code motion (H4 + layering)** — last, so no behavioral
   fix has to merge across the motion: run_residual_field_stage closures →
   run-loop class (residual_field/run_loop.py); backend assembly →
   assembly.py; _LocalSnapshotWriter → snapshot_writer.py; run_state_cache →
   core/workflow (storage becomes a leaf); M8 one documented precedence —
   OPERATOR ENV WINS over JSON config, config over default (the rule every
   launcher already assumes) — applied at the flip sites; identity-bearing
   resolved values are covered by the digest (phase 3), so no separate plan
   manifest record is needed; M3 flock failure logged loudly + capability
   probe covers locking and fails closed for multi-node runs.

## Explicit non-fixes (decision recorded, per the review's own verdicts)

- **M14 two stage-1 payload formats**: initially deferred as a store-format
  break with hot-path perf implications; subsequently DONE (`f941373`), and
  the cross-mode reuse it unblocked is now done too (see the follow-ups
  above). Nothing from the review's 48 findings remains open.
- Findings L7/L14 ("sound, do not disturb") are constraints on the above, not
  work items.

## Open finding, NOT fixed here: run_digest ignores the structure

Discovered while scoping the payload identity above, and larger than the
item it was found under.

`run_digest` addresses the whole durable run tree — `.mosaic/runs/<digest>/`
chunk commits, the streaming payload store, resume credits. It is built
from `scientific_digest`, which hashes the structure only through
`structure_content_digest` / `structure_file_sha256` / `structure_digest`,
and **no code path in the repository ever sets any of those keys**
(`grep` finds only the two definitions in `scattering/planning.py`).
`build_source_structure_digest` does hash the coordinates, but its result
is carried as metadata beside the identity — never folded into it.

Reproduction (two parameter sets differing only in `original_coords`):

    build_run_identity(params(1), ...).run_digest
      == build_run_identity(params(2), ...).run_digest   ->  True
    build_source_structure_digest(params(1))
      == build_source_structure_digest(params(2))        ->  False

Consequence: point one output directory at a different structure with the
same cell, supercell, intervals and mask, and the run resolves to the SAME
run tree. Scattering chunk commits validate (their recorded payload hashes
match the files the previous run left at the same paths), the SQLite cache
is credited from them, and the stage can be reported complete with the
PREVIOUS structure's results. `fresh_start` defaults to False, so this is
the default resume path, not an opt-in one.

Stage-1 payloads are no longer exposed to it — that is exactly what
`payload_identity` closes, and the cross-mode check's phase D demonstrates
the refusal on a real structure change. The chunk/commit layer above them
is still exposed.

Not fixed here because the fix is a second checkpoint-identity break:
folding `source_structure_digest` into `_build_run_identity_digest` changes
every run digest, so every existing `.mosaic/runs/` tree, payload store and
resume credit goes cold and recomputes — including the published hkl40
case. That is non-destructive (a disjoint family, per the phase-3
principle) but it is a deliberate, costly decision that belongs to the
operator, not to a follow-up commit. Recommended as its own change.

## Gates

After each phase: full unit suite + host-regress.sh (small_all + small_sphere,
PASS < 1e-10 Å). Phase 3 additionally proves resume: kill mid-run, resume,
0 resubmitted units for completed chunks, gate PASS. Phase 5 runs in the
cluster sim. Final: full suite, gate, sim spot-checks.
