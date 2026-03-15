# cuFINUFFT Wrapper Improvement Plan

This document describes the next implementation plan for
[core/adapters/cunufft_wrapper.py](../core/adapters/cunufft_wrapper.py).

The goal is to improve:

- GPU utilization
- throughput
- memory stability
- fallback behavior
- confidence in real GPU runs

This plan is intentionally local to the NUFFT adapter layer and its immediate
call sites. It does not reopen the broader scientific-stage architecture.

## Status

This plan is now materially implemented through the first three waves, plus a
first same-geometry inverse super-batch slice.

Implemented:

- Wave 1:
  - allocator reuse improvements
  - valid CuPy exception handling
  - improved chunk-planning observability
- Wave 2:
  - batched inverse execution for shared geometry
  - `n_trans`-aware budgeting
  - batched inverse call-site adoption in residual-field and scattering chunk work
  - bounded validation of `gpu_spreadinterponly`
- Wave 3:
  - internal device-resident inverse batching
  - GPU-resident accumulation where justified
  - centralized launch-parameter tuning hooks
  - optional pinned-host-copy path
  - optional stream-hook plumbing
- Post-Wave Slice:
  - additive `execute_inverse_cunufft_super_batch(...)` helper
  - width backoff before CPU fallback for large `n_trans` batches
  - first production adoption in residual-field interval-batch work over one
    shared `rifft_grid`

Still intentionally deferred:

- true multi-stream upload/compute/download overlap
- benchmark-driven autotuning of launch parameters
- deeper cuFINUFFT hidden-memory calibration from real GPU traces
- wider super-batch adoption in other call sites such as scattering, if
  profiling justifies it
- smarter automatic super-batch width selection based on measured VRAM headroom
- broader real-GPU benchmark and integration coverage beyond bounded validation

The sections below remain useful as the implementation rationale and as the
record of what still remains for later tuning waves.

## Benchmarking

A bounded wrapper benchmark harness is available at
[scripts/benchmark_cunufft_overlap.py](../scripts/benchmark_cunufft_overlap.py).

The harness is:

- GPU-gated
- opt-in
- intended for wrapper-level diagnostics, not scientific validation

Useful environment variables:

- `MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP=1`
- `MOSAIC_NUFFT_PINNED_HOST=1`
- `MOSAIC_NUFFT_GPU_STREAM=current`
- `MOSAIC_NUFFT_BENCH_Q_POINTS=<int>`
- `MOSAIC_NUFFT_BENCH_REAL_POINTS=<int>`

Example:

```bash
MOSAIC_NUFFT_BENCH_REAL_POINTS=134217728 \
MOSAIC_NUFFT_EXPERIMENTAL_OVERLAP=1 \
MOSAIC_NUFFT_PINNED_HOST=1 \
MOSAIC_NUFFT_GPU_STREAM=current \
conda run -n mosaic python scripts/benchmark_cunufft_overlap.py
```

Interpret benchmark results carefully:

- current experimental mode is still diagnostics-oriented
- modest improvements are more likely due to pinned-host-copy behavior and
  launch/timing differences than true asynchronous overlap
- true multi-stream overlap remains deferred

## Scope and Non-Goals

In scope:

- better use of GPU memory and planning
- better reuse of shared inverse-transform geometry
- fewer unnecessary host/device transfers
- more stable fallback and retry behavior
- better observability and tests

Not in scope for this wave:

- changing `eps` defaults
- redesigning the mathematical pipeline
- rewriting the whole adapter stack
- relying on undocumented cuFINUFFT behavior without validation

`eps` should remain unchanged unless explicitly requested in a separate task.

## Historical Context

The following issues motivated the original wrapper work and explain the wave
ordering above. They should not be read as the current literal implementation
state after the completed waves.

The earlier wrapper already did some important things correctly:

- explicit VRAM budgeting
- direction-aware source/target side selection
- CPU fallback when GPU execution is unavailable
- chunk backoff on GPU failure

However, the current implementation still leaves significant performance on the
table and has a few stability gaps:

1. `free_gpu_memory()` is called after every successful chunk.
   - This defeats CuPy pool reuse.
   - It increases allocator churn.
   - It reduces the benefit of any future pinned-memory or stream work.

2. The residual-field task still performs two separate inverse NUFFTs with the
   same geometry.
   - One for `q_amp - q_amp_av`
   - One for `q_amp_av`
   - This repeats expensive setup work.

3. Results are copied back to host per chunk and accumulated on CPU.
   - This is simple and safe.
   - It is not optimal for multi-interval inverse workloads.

4. The wrapper currently depends mostly on mock-driven tests.
   - This is fine for unit-level correctness.
   - It is not enough for high confidence in real GPU behavior.

5. The current wrapper still uses `gpu_spreadinterponly=1` in the GPU launch
   path.
   - This may be harmless for type 3.
   - It may also be a leftover debug-oriented flag.
   - It should be validated explicitly rather than assumed.

## Important cuFINUFFT / CuPy Facts

These facts should be treated as implementation constraints:

### cuFINUFFT Python `Plan` API

For `cufinufft.Plan` in the relevant development API:

- constructor:
  - `Plan(nufft_type, n_modes_or_dim, n_trans=1, eps=..., isign=..., dtype=..., **kwargs)`
- for type 3:
  - `n_modes_or_dim` must be a single integer dimension
  - passing `(0,) * dim` is invalid
- correct type-3 pattern:
  - `cufinufft.Plan(3, dim, n_trans=2, eps=1e-12, isign=-1, dtype="complex128", ...)`
- `setpts(x, y=None, z=None, s=None, t=None, u=None)`:
  - `x,y,z` are source coordinates
  - `s,t,u` are target coordinates
- `execute(data, out=None)`:
  - for `n_trans=2`, `data` must have shape `(2, n_sources)`
  - returns `(2, n_targets)`
- default dtype is `complex64`
  - pass `dtype="complex128"` explicitly

### Valid cuFINUFFT GPU kwargs

Only supported kwargs should be used, for example:

- `upsampfac`
- `gpu_method`
- `gpu_sort`
- `gpu_binsizex`, `gpu_binsizey`, `gpu_binsizez`
- `gpu_obinsizex`, `gpu_obinsizey`, `gpu_obinsizez`
- `gpu_maxsubprobsize`
- `gpu_kerevalmeth`
- `gpu_spreadinterponly`
- `gpu_maxbatchsize`
- `gpu_device_id`
- `gpu_stream`
- `modeord`

Anything else should be treated as invalid and avoided.

### CuPy 13.4.1 exception handling

The wrapper should only catch CuPy exceptions that actually exist.

Use:

- `cp.cuda.memory.OutOfMemoryError`
- `cp.cuda.runtime.CUDARuntimeError`
- `cp.cuda.driver.CUDADriverError`

Do not use:

- `cp.cuda.memory.MemoryError`

`OutOfMemoryError` inherits from Python's builtin `MemoryError`; there is no
CuPy `cp.cuda.memory.MemoryError` class.

This must be fixed everywhere it appears, especially:

- `_as_device(...)`
- the main GPU execution loop
- any future batched GPU execution loop

## Priority Order

The highest-ROI implementation order is:

1. stop per-chunk memory-pool flushing
2. add batched inverse execution for shared geometry
3. use the batched inverse path in both:
   - `core/residual_field/tasks.py`
   - `core/scattering/tasks.py`
4. add GPU-resident accumulation support where it is safe and local
5. add observability and real-GPU validation
6. only then consider streams, pinned memory, and deeper tuning

## Wave 1: Stability and Allocator Reuse

### Objective

Stop wasting GPU resources while preserving current behavior.

### Changes

1. Remove `free_gpu_memory()` from the successful per-chunk `finally` path.
2. Keep `free_gpu_memory()` on:
   - explicit OOM/backoff
   - stage-end cleanup
   - forced CPU fallback transitions
3. Fix exception lists so they only reference valid CuPy exception classes.
   - explicitly remove `cp.cuda.memory.MemoryError`
   - fix all current catch sites, not only the main execution loop
4. Add debug-level logging for:
   - chosen chunk size
   - retry count
   - fallback reason
   - free VRAM at chunk start
   - budget breakdown:
     - `resident_bytes`
     - `grid_bytes`
     - `per_target_bytes`
   - direction and problem size:
     - `n_sources`
     - `n_targets`
     - forward vs inverse

### Expected result

- lower allocator churn
- better reuse of CuPy memory pools
- more stable chunk throughput
- easier diagnosis of GPU fallback behavior

## Wave 2: Batched Inverse Execution

### Objective

Reuse shared inverse geometry where multiple weight vectors use the same
`q_coords` and `rifft_grid`.

### Why this matters

The residual-field path currently computes:

- inverse transform of `q_amp - q_amp_av`
- inverse transform of `q_amp_av`

with the same geometry.

The scattering chunk task has the same dual-inverse pattern and should be
treated as part of the same wave.

### Changes

1. Add a public helper:
   - `execute_inverse_cunufft_batch(...)`
2. Back it with a type-3 plan path using:
   - `cufinufft.Plan(3, dim, n_trans=...)`
   - `dtype="complex128"`
3. Ensure stacked weights have shape:
   - `(n_trans, n_sources)`
4. Ensure type-3 plan construction uses:
   - integer `dim`, not `(0,) * dim`
5. Make the budget model `n_trans`-aware.
   - `_per_target_bytes(...)` should accept `n_trans`
   - chunk planning should pass `n_trans` through
   - resident budgeting should include the stacked weight tensor
6. Update both dual-inverse call sites to use the batched API:
   - [core/residual_field/tasks.py](../core/residual_field/tasks.py)
   - [core/scattering/tasks.py](../core/scattering/tasks.py)
7. Validate whether `gpu_spreadinterponly=1` is appropriate for production
   type-3 execution.
   - compare against `gpu_spreadinterponly=0` on a known-answer case
   - remove the flag if it is redundant or suspicious

### Expected result

- fewer duplicated inverse launches
- better GPU occupancy
- lower repeated setup cost in the residual-field stage

## Wave 3: GPU-Resident Accumulation

### Objective

Reduce repeated GPU-to-host round-trips for multi-interval inverse work.

### Changes

1. Add an explicit internal device-resident API rather than an ambiguous mode.
   For example:
   - `execute_inverse_cunufft_batch(..., device_out=True)`
   - or an internal helper such as `_execute_inverse_cunufft_batch_device(...)`
2. In the residual-field batch task:
   - keep running accumulators on GPU
   - perform one host copy at the end of the interval batch
3. Keep the default public API host-returning unless explicitly changed.

### Expected result

- fewer transfers
- lower CPU accumulation overhead
- better performance for multi-interval shard batches

## Wave 4: Tuning and Overlap

### Objective

Push utilization further only after the core execution path is stable.

### Candidate work

1. benchmark `gpu_method`, `gpu_maxsubprobsize`, and `gpu_maxbatchsize`
2. evaluate pinned host memory for chunk uploads/downloads
3. evaluate CUDA stream overlap for transfer + compute pipelining
4. refine budget heuristics using observed GPU traces

### Important note

These are tuning tasks, not first-line correctness fixes.

They should only be attempted after:

- allocator reuse is fixed
- batched inverse is in place
- real-GPU validation exists

## Testing Plan

### Unit tests

Extend [tests/unit/runtime/test_cunufft_wrapper.py](../tests/unit/runtime/test_cunufft_wrapper.py) for:

- valid exception handling
- no per-success-chunk pool flushing
- batched inverse API shape handling
- fallback reason logging or state signaling
- `n_trans`-aware budget behavior

### Stage-level tests

Add targeted tests around:

- residual-field dual inverse path using batched execution
- scattering dual inverse path using batched execution
- numerical equivalence of batched vs separate inverse calls
  - use the direct CPU fallback as the installation-independent reference path
- `gpu_spreadinterponly` validation on a bounded known-answer case

### Real GPU validation

Add an opt-in integration or benchmark harness that runs only when GPU support is
explicitly enabled.

It should cover:

- forward-heavy asymmetric case
- inverse-heavy asymmetric case
- chunk backoff behavior
- CPU fallback reason capture

## Implementation Notes

### Keep public compatibility where possible

Preferred public surface:

- `execute_cunufft(...)`
- `execute_inverse_cunufft(...)`

Additive APIs are fine if they are narrow and justified, for example:

- `execute_inverse_cunufft_batch(...)`

### Do not silently change scientific behavior

This work is about:

- planning
- memory
- batching
- stability

It should not change:

- transform sign conventions
- `eps`
- output dtype expectations
- downstream scientific-stage contracts

## Recommended First Slice

The smallest defensible first implementation slice is:

1. remove per-success-chunk `free_gpu_memory()` flushing
2. fix invalid CuPy exception catches
3. add `execute_inverse_cunufft_batch(...)`
4. make batch budgeting explicitly `n_trans`-aware
5. switch the residual-field and scattering dual inverse call sites to use it
6. add a bounded validation for `gpu_spreadinterponly`
7. add targeted tests

This provides the best balance of:

- low risk
- meaningful GPU throughput gain
- improved stability
- minimal architectural churn

## Deferred Until Proven Necessary

These items are intentionally deferred:

- changing default `eps`
- unified memory fallback
- full stream pipeline redesign
- broad forward-stage batching redesign
- rewriting the wrapper into multiple modules

They may still be valuable later, but they are not the highest-ROI next step.
