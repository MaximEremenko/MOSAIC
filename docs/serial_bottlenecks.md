# Serial-bottleneck inventory (hkl40 streaming hot path)

Produced by a 29-agent audited sweep (every finding adversarially verified for
hot-path reachability and cost arithmetic). Principle: on a 4-GPU / 96-core
node, any phase holding one core or one GPU while the rest idle is a defect.

Already fixed on `feature/tile-owner-architecture` (excluded below): q-space
plan build (threaded), per-site decode extraction (fork pool), stage-1
prologue (durable payload store + multi-GPU prewarm + windowed loaders),
submission interleave, 2-in-flight double buffering, finalize snapshot loads
(npz mmap), rifft grid build (blockwise memmap), decoder ridge solve
(threadpoolctl lifts the 1-thread BLAS pin for the driver-side solve).

## Verified findings, ranked

### Major (minutes each)

1. **Streaming finalize collapses onto ONE worker.** All 4 chunk finalizes
   pin to the same owner (slot-keyed ownership maps every finalize target to
   one worker). Per chunk: ~34 GB snapshot read + summation + 13.5 GB write
   ≈ 40–80 s → 3–5 min serialized on one worker while three idle.
   *Fix:* key finalize ownership by chunk (round-robin owners), or fan out
   in the driver with a process per chunk. Watch SQLite: route
   `_mark_residual_intervals_saved` through one writer.

2. **decoder.source='current' runs the whole per-chunk input pipeline
   twice** (training-collection pass, then decode pass): double 6.78 GB h5
   reads, double grid regeneration, double groups build (~10+ min total).
   *Fix:* cache the per-chunk prepared inputs (features are already
   computed in the training pass — reuse for decode).

3. **groups-index build is a pure-Python loop over 211.7M rows** (~42 s per
   chunk-pass, ×8 passes ≈ 5–6 min, GIL-bound).
   *Fix:* vectorize — `np.argsort(ids_all, kind='stable')` + boundary
   `np.searchsorted` gives the same site→row-index groups in ~2 s.

4. **End-of-stage flush barrier**: ~271 GB of accumulator snapshot writes
   with zero compute overlap (90–135 s, GPUs idle).
   *Fix:* stream snapshots during the run's final batches (cadence-aligned)
   or overlap the flush with the first finalize reads.

5. **Finalize metadata materializes 211.7M point ids as Python tuples,
   twice per chunk** (~45 s/chunk + GC churn; `contracts/accumulation`).
   *Fix:* carry (start, stop) ranges or ndarrays through
   `ResidualFieldPartialResult`; never `tuple(int(x) for x in 211M)`.

6. **recover_pending finalizes pending chunks serially in the driver**
   (~80–90 s/chunk).
   *Fix:* same per-chunk fan-out as (1); shares its risk notes.

### Moderate (30 s – 2 min)

7. **54 GB output pre-allocation, single-threaded** (measured 51 s at
   1.07 GB/s; both columns derivable — ids are run-length metadata, values
   are fill).
   *Fix:* per-chunk process fan-out (HDF5 global lock defeats threads), or
   eliminate: write an RLE sidecar and materialize on first real write.

8. **SQLite row-at-a-time**: 9,216 interval-chunk status updates, each a
   fresh connection + single-row fsync commit (~46 s NVMe; minutes on
   slower fsync); the resume-time cache rebuild replays up to 18k more.
   *Fix:* batch with `executemany` in one transaction per chunk; keep one
   connection; WAL.

9. **Shard lattice scatter runs on the single consumer thread** (~155M
   fancy-index adds per shard).
   *Fix:* scatter per-interval partials into per-thread grids and reduce, or
   move the scatter to the GPU (cupyx scatter_add) since the grid is bound
   for the GPU anyway.

10. **Snapshot writes run inline under the per-target lock**
    (accept_local_contribution blocks folds for 4.5–7 s per 8.5 GB savez).
    *Fix:* snapshot from a copy-on-write view outside the lock, or offload
    to a writer thread.

### Minor / structural notes

- Type-2 tile loop: pageable D2H + coordinate wrap are synchronous with the
  transform stream; pinned buffers + async copy would hide ~10–15% of tile
  time. The 2-in-flight transforms are tile-granularity concurrent (ledger
  arbitrated) — adequate.
- Startup redundancy across cases: each case re-does structure load, DB
  setup, cluster spawn (~2 min). A shared long-lived cluster + shared
  preparation would make case starts ~seconds (matters once cases pipeline).
- Refuted by verification (do not chase): run-state scan hashing (measured
  0.12 s — the feared 135 GB path is durable-mode-only); quiescence glob
  loops (cold on this config); submit-loop O(N²) maps (negligible at ≤64
  units).
