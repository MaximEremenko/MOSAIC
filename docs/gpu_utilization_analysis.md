# GPU Utilization and Execution-Mode Plan

This document replaces the earlier single-run diagnosis with a current
execution plan for improving GPU utilization **without sacrificing
restartability**.

It is complementary to:

- [hpc_parallel_plan.md](hpc_parallel_plan.md)
- [cunufft_wrapper_plan.md](cunufft_wrapper_plan.md)
- [qspace_equation_mask_plan.md](qspace_equation_mask_plan.md)

## Target Scale

This plan is intentionally aimed at the actual expected operating range:

- default: 1 node with 1–4 GPUs
- hard case: up to roughly 10–15 nodes

It is **not** trying to solve extreme-scale 1000-node orchestration.
That keeps the design practical and avoids over-engineering.

## Current Situation

The current active workflow is:

```text
WorkflowService.run()
  1. ScatteringStage.execute()
  2. ResidualFieldStage.execute()
  3. DecodingStage.execute()
```

The scientific-stage split is good and should remain.

However, the runtime behavior still uses a durability-heavy transport path even
for local runs:

- scattering writes interval `.npz` artifacts
- residual field reloads those artifacts
- residual field writes shard checkpoints
- reducers reload shard checkpoints and write final chunk artifacts
- decoding reads the final chunk artifacts

This is safe and restartable, but it is not the best default for same-node
throughput.

## What Is Already Good

- The shard/reducer model in `residual_field` is much safer than the old
  shared-mutable HDF5 path.
- The cuFINUFFT wrapper is now materially stronger:
  - adaptive budgeting
  - batched and super-batched inverse execution
  - telemetry and benchmark harnesses
- Q-space masking is no longer a blind CPU bottleneck:
  - blockwise masking exists
  - equation-mask GPU acceleration exists
- Decoder-consistency work has closed the immediate patch-shape correctness
  hole for the single-decoder path and added explicit family-mode support.

These are important foundations and should not be undone.

## Main Live Bottlenecks

The main remaining utilization problems are now architectural and transport
related:

1. forward-to-inverse stage barrier
2. interval `.npz` disk round-trip
3. shard checkpoint writes on local runs
4. repeated Dask serialization of large shared arrays
5. redundant forward NUFFT work per element
6. small but still-live Python overheads around masking and point selection

## Key Principle

The solution is **not**:

- “put everything in RAM”

because large 3D outputs can be enormous.

The solution is also **not**:

- “use one pipeline for both local and cluster and hope it is fast enough”

The right model is:

- same logical map → reduce pipeline
- different transport / accumulator backends depending on execution mode

## Correct Architecture

Keep the same logical flow:

```text
interval -> scattering result
interval-batch + chunk -> residual contribution
chunk-owned reducer -> final chunk artifact
decoder consumes final chunk artifact
```

The difference between local and HPC should be:

- where partial state lives
- how durable it is
- how aggressively it is checkpointed

not:

- whether restart exists

Single-node runs should also be restartable.

## Race-Safety Requirements

Any optimization in this plan must preserve the current safety guarantees and
must not reintroduce shared-write races.

Mandatory invariants:

1. **one reducer owns one final chunk artifact family**
   - never allow multiple workers to mutate the same final chunk outputs
2. **worker map tasks produce only isolated partial state**
   - in-memory payloads
   - local scratch partials
   - or immutable durable checkpoints
3. **final outputs are only marked complete after reducer commit**
   - artifact write
   - manifest/progress update
   - validation
4. **restart must not depend on filename existence alone**
   - manifest/progress state must agree with artifact state
5. **local-fast optimizations must not bypass reducer ownership**
   - local mode may use different transport/backing storage
   - but not a different race model

This means:

- no return to shared mutable HDF5 accumulation from many workers
- no direct multi-writer updates to `point_data_chunk_*` artifact families
- no “fast path” that compromises chunk-level reducer ownership

## Two Restartable Modes

### 1. Local Restartable Mode

Goals:

- fast
- restartable
- optimized for same-node execution

Characteristics:

- use local scratch / output directory for restartable reducer state
- avoid unnecessary interval `.npz` round-trips when forward and inverse stay on
  the same node
- avoid unnecessary durable shard writes when chunk-local accumulation can remain
  on local scratch until reducer finalization
- keep single-writer chunk reducers

This mode is about:

- minimizing shared-filesystem style overhead
- while still surviving process crashes or reruns

### 2. Durable Distributed HPC Mode

Goals:

- restartable across workers/jobs/nodes
- safe under long runtimes and preemption

Characteristics:

- immutable interval artifacts when needed
- immutable residual shard checkpoints
- reducer progress manifests
- worker-local scratch for temporary build
- durable commit to shared storage

This mode is about:

- resilience and correctness first
- throughput second

## What Should Not Differ

These should stay common to both modes:

- scientific-stage boundaries
- the ability to persist scattering / q-space interval artifacts when explicitly requested
- chunk-owned reducer model
- final artifact contracts
- decoder semantics
- mask semantics

## Most Important Remaining Code Issues

These are the main live gaps in the current branch:

### 1. Forward → inverse transport is still disk-first

For local runs, interval `.npz` artifacts are still the default transport
between scattering and residual field.

This is the largest same-node utilization loss.

Important clarification:

- scattering interval artifacts are still scientifically useful and must remain
  a supported output surface
- publication examples, debugging workflows, and reproducibility runs may
  legitimately want those saved
- the optimization goal is therefore **not** to remove them from the product
- the goal is to stop using them as the mandatory transport path for every run

### 2. Shard checkpoint writes are still always on the hot path

For local runs, the reducer can often use local restartable accumulation
instead of full checkpoint round-trips for every partial.

### 3. Shared arrays are still passed repeatedly to scattering tasks

Large atom-coordinate arrays should be scattered once, not serialized per task.

### 4. Forward stage still contains redundant element work

The `q_av` computation is still repeated per element, and some forward calls can
still be batched better.

### 5. Point-selection overhead is still visible

The point-data generation path still uses a per-point loop and duplicate zero
amplitude generation.

These are real issues, but they are lower priority than the stage barrier and
transport path.

## Three Required Waves

There are **3 implementation waves** left to close the important execution
gaps in a bounded way.

## Wave 1: Unified Restartable Reducer Backend Model

Introduce an explicit reducer backend abstraction.

Possible backends:

- tiny in-memory only where safe
- local scratch-backed reducer
- durable shared-storage reducer

Goals:

- same logical reducer ownership
- bounded memory use
- restartability in both local and HPC modes
- no assumption that large runs can stay in RAM
- explicit race-safe reducer state transitions

Required output of Wave 1:

- a reducer backend abstraction
- explicit reducer commit/finalize semantics
- explicit statement of which state lives in RAM, local scratch, or durable
  storage
- no change to single-writer chunk ownership
- an explicit policy boundary between:
  - scattering interval artifacts as optional/supported outputs
  - scattering interval artifacts as required execution transport

This wave closes the architecture mistake of treating local optimization as
“all in memory” and cluster optimization as “everything on disk”.

## Wave 2: Local Restartable Fast Path

Build the same-node optimized execution mode on top of Wave 1.

Goals:

- direct forward → inverse handoff when running locally
- local scratch-backed chunk reducers
- fewer hot-path interval/shard writes
- scatter shared arrays once
- sane default worker topology for one GPU / one node
- preserve restartability after local crash
- preserve reducer single-writer semantics

Required output of Wave 2:

- a local restartable mode that is faster but still manifest/reducer-driven
- fewer local transport/checkpoint writes on the hot path
- unchanged final chunk ownership model
- an explicit option to still save scattering/q-space interval artifacts when
  requested for publication or debugging workflows

This is the biggest expected improvement for GPU utilization on workstations and
single-node runs.

## Wave 3: Durable Distributed HPC Path

Refine the distributed checkpointed path separately.

Goals:

- durable restart across workers/jobs
- scratch vs durable storage separation
- explicit checkpointing policy
- reduced unnecessary compression / serialization overhead
- scheduler/resource/locality refinement
- preserve the same reducer ownership and manifest semantics as local mode

Required output of Wave 3:

- a cleaner durable distributed mode for 1–15 node use
- explicit checkpoint policy for when interval/shard durability is required
- race-safe reducer progress and recovery remain intact
- explicit policy for when scattering interval artifacts are:
  - required for restartability
  - optional for inspection/reproducibility
  - safely skippable as transport artifacts

This wave keeps the system cluster-friendly without forcing the local path to
pay the same cost.

## Low-Risk Fixes to Fold Into Stages 1–2

These are still worthwhile, but they are not separate roadmap stages:

- eliminate redundant `q_av`
- batch `q_amp + q_delta` where safe
- replace hot-path `np.savez_compressed` with `np.savez` where checkpoint format
  remains unchanged
- scatter shared arrays once in scattering execution
- cache `inspect.signature` results for mask dispatch
- remove duplicate `_generate_amplitude` calls in point selection

These should be treated as supporting cleanup within the major waves above.

## What Not to Do

- do not design around a single universal path that is equally optimal for
  single-node and distributed HPC
- do not require the full reduced output to stay in RAM for large 3D jobs
- do not make restartability optional only for single-node runs
- do not let worker topology defaults encourage multiple process workers on a
  single GPU unless the user explicitly asks for it

## Success Criteria

This plan is successful when:

### Local restartable runs

- forward and inverse stages avoid unnecessary disk transport
- local restarts are still possible
- GPU utilization rises materially on representative same-node runs
- reducer ownership remains single-writer and race-safe

### Durable distributed runs

- restartability and checkpoint integrity remain intact
- partials and reducers remain single-writer / manifest-driven
- throughput improves without weakening crash recovery
- the plan remains practical for the real target scale of 1–15 nodes

## Bottom Line

The biggest remaining utilization problem is not masking or decoder logic
anymore.

It is that the current default transport and checkpoint path is optimized for
durability first even when running locally.

The correct solution is:

- **same restartable logical reducer model everywhere**
- **different storage locality and checkpoint policy for local vs HPC**
- **3 stages total**

This is the most balanced solution for:

- speed
- safety
- crash/restart behavior
- single-node friendliness
- cluster friendliness
