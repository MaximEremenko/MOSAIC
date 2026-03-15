# HPC Parallel Execution Plan

This document describes the target execution architecture for MOSAIC when run
on clusters with Dask, multiple workers, and optionally multiple GPUs.

It is complementary to:

- [README.md](../README.md)
- [input_schema.md](input_schema.md)
- [REFACTORING_PLAN.md](../REFACTORING_PLAN.md)

The goal here is not package naming. The goal is to make the current
scientific-stage architecture:

- pause-friendly
- rerun-friendly
- crash-resistant
- cluster-friendly
- multi-GPU-friendly
- disk-space-friendly

---

## Scientific Basis

The manuscript workflow is:

1. load an atomistic configuration
2. compute the diffuse scattering amplitude `ΔA(Q)`
3. apply a phase-preserving mask `W_F(Q)`
4. evaluate the inverse Fourier transform to obtain a residual field `R_F(r)`
5. decode site-resolved displacement or occupancy information

This implies a natural parallel dataflow:

- `scattering` owns masked reciprocal-space artifacts
- `residual_field` owns inverse-transform materialization and accumulation
- `decoding` consumes final residual-field artifacts only

---

## Current Safety Goal

The old unsafe pattern was:

- many worker tasks updating the same final chunk HDF5 artifacts directly

Examples:

- `point_data_chunk_<id>_amplitudes.hdf5`
- `point_data_chunk_<id>_amplitudes_av.hdf5`
- `point_data_chunk_<id>_applied_interval_ids.hdf5`

Even with locks, this is fragile for:

- crash recovery
- stale artifact reuse
- shared filesystem contention
- multi-GPU scaling
- deterministic replay

The target design therefore moves toward:

- immutable worker outputs
- single-writer reducers
- explicit manifests and retry semantics

---

## Target Dataflow

### Stage 1: Scattering Map

Each map task owns one reciprocal-space interval.

Input:

- structure coordinates
- form-factor data
- interval definition
- reciprocal-space mask definition

Output:

- one immutable interval artifact

Typical artifact:

- `interval_<id>.npz`

Contents:

- `q_grid`
- `q_amp`
- `q_amp_av`
- interval id
- schema/version metadata

### Stage 2: Residual-Field Map

Each map task owns one `(interval_id, chunk_id)` or one
`(interval_batch, chunk_id)` work unit.

Input:

- one or more scattering interval artifacts
- one chunk’s patch-center grid data

Output:

- one immutable residual shard checkpoint

Suggested artifact:

- `residual_chunk_<chunk>__batch_<batch>__interval_<interval>.npz`

or, preferably for batching:

- one shard per `(chunk, batch-of-intervals, worker-attempt)`

Contents:

- chunk id
- contributing interval ids
- point ids
- grid shape
- residual values
- residual-average values
- reciprocal-point count
- schema/version
- idempotency key

### Stage 2: Residual-Field Reduce

Exactly one reducer owns each final chunk artifact family.

Input:

- all committed residual shard checkpoints for one chunk

Output:

- final chunk artifact set

Examples:

- `point_data_chunk_<id>_amplitudes.hdf5`
- `point_data_chunk_<id>_amplitudes_av.hdf5`
- `point_data_chunk_<id>_shapeNd.hdf5`
- `point_data_chunk_<id>_amplitudes_nreciprocal_space_points.hdf5`
- `point_data_chunk_<id>_amplitudes_ntotal_reciprocal_space_points.hdf5`
- `point_data_chunk_<id>_applied_interval_ids.hdf5`

The reducer is the **only** writer of the final chunk artifacts.

### Stage 3: Decoding

`decoding` consumes only final residual-field chunk artifacts.

It does not depend directly on scattering interval artifacts.

---

## Storage Strategy

### Durable Storage

Use durable shared storage for:

- scattering interval artifacts
- committed residual shard checkpoints
- final chunk artifacts
- manifests

### Worker-Local Scratch

Use worker-local scratch for:

- temporary shard build files
- temporary in-memory partials spilled to local disk
- non-committed worker intermediates

This reduces pressure on the shared filesystem and improves multi-GPU cluster
behavior.

### Final Output Format

Recommended:

- immutable partials as `npz`
- final user-facing chunk outputs as HDF5

This balances:

- cluster safety
- low metadata overhead
- compatibility with current downstream code

---

## Manifest and State Model

Each stage must expose explicit manifest/state objects.

### Scattering Interval Manifest

Tracks:

- interval id
- schema/version
- artifact path
- completion status
- producer version

### Residual Shard Manifest

Tracks:

- chunk id
- contributing interval ids
- parameter digest
- shard artifact paths
- completion status
- retry/idempotency key

### Final Chunk Manifest

Tracks:

- chunk id
- expected shard inputs
- completed shard inputs
- final artifact paths
- validation status
- cleanup eligibility

---

## Completion Rules

Artifact existence alone must never imply completion.

A work unit is complete only when:

1. required artifact files exist
2. artifact schema/version validates
3. manifest marks completion
4. replay/idempotency markers agree
5. any persisted state index agrees with the manifest

SQLite or any other database should be treated as:

- scheduling/index state

not:

- the sole authority for artifact correctness

---

## Retry, Pause, and Resume Semantics

### Pause/Stop

Safe stopping means:

- unfinished work units remain retryable
- completed shard checkpoints remain valid
- final chunk outputs are only marked complete after reducer commit and
  manifest validation

### Resume

On resume:

1. discover manifests
2. skip completed interval artifacts
3. skip completed residual shards
4. skip fully reduced chunks
5. continue from remaining work only

### Rerun

Support two rerun modes:

- `fresh_start`
  - clear generated outputs and manifests
- manifest-aware resume
  - reuse valid artifacts and continue only unfinished work

---

## Single-Writer Rule

The single-writer rule is mandatory:

- map workers do not write final chunk artifacts
- reducers do not write scattering interval artifacts
- one reducer owns one final chunk artifact family

Locks may still exist as safety nets, but the design should not rely on shared
multi-writer mutation as the primary execution model.

---

## Dask and Multi-GPU Rules

### Worker Layout

For GPU runs:

- one worker per GPU
- `threads_per_worker = 1`
- explicit GPU affinity via `CUDA_VISIBLE_DEVICES`
- bounded in-flight tasks per worker

### Resource Labelling

Workers should advertise explicit resources for NUFFT/GPU work so scheduling is
predictable.

### Serialization Discipline

Worker tasks should receive only:

- a compact work unit
- references to artifacts
- minimal execution parameters

Do not send:

- whole configuration blobs
- orchestration-only objects
- large global Python structures

---

## Disk-Space Strategy

Pure immutable shard output is safest, but it can create too many small files.

The preferred compromise is:

- worker-local accumulation over a small batch of intervals
- one compact shard checkpoint per chunk/batch
- reducer merges compact shard checkpoints
- shard checkpoints become reclaimable after final chunk commit

This gives:

- safer restart semantics
- bounded file count
- bounded shared-storage pressure

---

## What Is Still Missing for Optimal HPC Use

The current codebase is on the right path but is not yet fully optimal for
large-scale HPC execution.

Main remaining items:

1. make shard-first stage-2 execution the only normal path
2. separate worker-local scratch from durable storage explicitly
3. reduce overlap between database state and artifact completion state
4. add real distributed integration tests:
   - multi-worker Dask
   - retry/replay
   - manifest-based resume
   - reducer correctness
   - CPU-only distributed smoke
   - optional CUDA-local smoke

---

## Recommended Next Technical Milestones

1. Introduce explicit residual shard reducer ownership everywhere.
2. Add scratch-storage support for worker-local shard writes.
3. Make final artifact commit protocol atomic and manifest-driven.
4. Add distributed integration tests for retries, resume, and reducer
   correctness.

This is the path from “good architecture” to “truly optimal for HPC cluster
parallelism.”
