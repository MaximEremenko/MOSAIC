# Tile-Owner Architecture (branch `feature/tile-owner-architecture`)

Status: phase 1 implemented and validated end-to-end on `demo_3D_displacement_CaTiO3_hkl40`
(4x RTX 4090, 61 GB RAM, NVMe). FP64 (`complex128`, `eps=1e-12`) untouched throughout.

## Governing principle

**The problem size must never set the memory footprint — budgets must.** Every
hkl40 failure before this branch traced to an object sized by the problem:
whole-chunk accumulators (6.78 GB), whole-shard payload lists (8.4 GB), full
rifft target grids (5.1 GB), full transform outputs (6.8 GB), full snapshot
materializations (8.5 GB). The architecture bounds every resident object to
O(budget) or O(tile), so the same code runs on a small-RAM laptop with one GPU
and on a multi-node cluster.

## What changed (one commit each)

1. **Budget-bounded work units** (`3fd44b6`)
   - `execute_type2_on_lattice(tile_consumer=...)`: completed target tiles
     stream to the caller; the full `(n_trans, n_tgt)` result and the full-size
     wrapped-coordinate array are never materialized. `real_coords` may be a
     read-only memmap.
   - `_execute_lattice_groups` folds tiles into a (delta, average) pair with
     per-tile conjugate reconstruction; large pairs live in unlinked scratch
     memmaps (evictable page cache, not anonymous RSS).
   - Streaming stage-1 inputs are zero-arg lazy loaders: the two-pass lattice
     builder holds ONE ~57 MB payload at a time.
   - The rifft target grid builds center-by-center into a reusable scratch
     `.npy` memmap (atomic rename, keyed by digest/chunk/partition).
   - **Spatial shard packing**: interval ids run k-fastest, so contiguous-id
     shards span the full reciprocal volume (a 147-id run = a 31 GiB dense
     lattice grid, which failed RAM admission and silently fell back to
     type-3). Shards now pack greedily by bounding-box grid bytes after a
     spatial sort — deterministic given the config, so shard identity (and the
     checkpoint ledger) is invariant to worker count.
   - Infrastructure casualties (KilledWorker/cancel/comm) no longer count
     toward the GPU circuit breaker that used to flip whole runs to CPU-only.

2. **Shared point-id vector** (`87a9eca`): one scratch-memmap
   `start + arange(count)` per (start, count) per process, instead of a fresh
   1.69 GB int64 array per accumulator (slots x chunks copies). Configs moved
   to 4 workers x 1 thread and a FIXED `residual_streaming_subchunks: 8` so
   checkpoints resume across 1/2/4/8-GPU layouts.

3. **Dask RSS limiter off for cuda-local** (`38e0969`): file-backed memmap
   pages count into RSS; the default per-worker limit read them as worker
   memory and the nanny killed healthy workers at 95%. `DASK_MEMORY_LIMIT`
   still overrides.

4. **Memmap snapshot loads** (`212e...`): `np.savez` members are ZIP_STORED, so
   finalize maps each snapshot array in place at its zip offset instead of
   materializing 8.5 GB per snapshot (~50 GB anonymous across a chunk's slots —
   the reproducible driver OOM right after "Residual-field finished").

## Measured outcome (hkl40 `all` case)

| metric | before | after |
|---|---|---|
| run survives | never past ~batch 10 | complete end-to-end |
| GPUs used | 2 (by config, RAM-forced) | 4 |
| worker restarts / failed batches | dozens / thousands (cascade -> CPU-only abort) | 0 / 0 |
| shard dense grid | up to 31.3 GiB (silent type-3 fallback) | <= 3.94 GiB, always lattice type-2 |
| residual compute wall | n/a (died) | ~1:29 h (32 units; ~5.9 GPU-min is stage-1) |
| output | none | 12,288 sites, RMS u = 0.224 A, 0 NaN |

## Known remaining inefficiencies (next steps, in value order)

All five items previously listed here were implemented during the
2026-08-01 fix campaign and its follow-ups; the list is kept for the
record, marked with what closed it.

1. ~~**Durable stage-1 store (Phase 0)**~~ — DONE. Payloads persist under
   `stage1_payload_store/<payload_identity>/`, computed once per interval
   across shards, owners, restarts and cluster sizes, and reusable from the
   precompute mode's artifacts as well (`afc2dc9`).
2. ~~**Tail serialization**~~ — DONE. Ownership is per (chunk, slot) and the
   submission interleave is ordered by RESOLVED owner, so placement stays
   spread when worker count divides chunk count (campaign M21).
3. ~~**Batch-major submission window**~~ — DONE by the same change; prefetch
   is no longer the workaround.
4. ~~**Finalize assembler**~~ — DONE. The points-axis concat assembler
   streams partitions through the memmap outputs, holding roughly one
   partition plus the output buffers rather than all partitions at once.
5. ~~**Decoder/extraction stage**~~ — DONE. Per-chunk decode is parallel and
   carries prepared inputs from the training pass instead of redoing the
   per-chunk pipeline.

## Operational notes

- The notebook launcher cell is the reference environment: loader path for the
  venv's CUDA 12 wheels (system CUDA 13 is ABI-incompatible with cufinufft),
  every scratch root kept off the 31 GB tmpfs `/tmp`, and the measured-safe
  knob values with comments explaining each.
- `processes: false` in `runtime.dask` is load-bearing for the CuPy pool cap
  (see `_apply_cupy_pool_cap`); do not "fix" it to true.
- Changing `MOSAIC_RESIDUAL_SHARD_GRID_BUDGET_BYTES` changes shard identity and
  invalidates checkpoints; treat it as part of the run's parameters.
