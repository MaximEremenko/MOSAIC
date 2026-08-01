# Architecture Fix Campaign — 2026-08-01

Executes the 48 confirmed findings of `docs/architecture_review_2026-07-31.md`.
Constraints carried over: FP64 (complex128) + NUFFT eps=1e-12 untouched; numerical
gate is max|diff| < 1e-10 Å on chunk site displacements (host-regress.sh) after
every phase; summation order is free; local branch only, no push.

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
   core/workflow (storage becomes a leaf); M8 single documented
   config-over-env precedence at the two flip sites, resolved values recorded
   in the stage plan manifest; M3 flock failure logged + capability probe
   covers locking.

## Explicit non-fixes (decision recorded, per the review's own verdicts)

- **M14 two stage-1 payload formats**: both are live (HDF5 serves every
  non-streaming example, npz serves streaming); consolidation is a store
  format break with hot-path perf implications. Deferred to the next store
  version; documented here as accepted debt.
- Findings L7/L14 ("sound, do not disturb") are constraints on the above, not
  work items.

## Gates

After each phase: full unit suite + host-regress.sh (small_all + small_sphere,
PASS < 1e-10 Å). Phase 3 additionally proves resume: kill mid-run, resume,
0 resubmitted units for completed chunks, gate PASS. Phase 5 runs in the
cluster sim. Final: full suite, gate, sim spot-checks.
