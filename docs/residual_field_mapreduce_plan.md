# Residual-Field Map-Reduce Release Plan

## Goal

Stabilize the residual-field stage for release by fixing the reduction boundary,
removing driver-side accumulation, preserving race safety and restartability,
and keeping the design practical for:

- 1 node with 1-4 GPUs
- up to 10-15 nodes

This plan is for execution architecture only. It does not change decoder
semantics, scattering-stage architecture, or the scientific inverse method.

## Current Bottleneck

The current local path is still architecturally wrong for release:

- a residual task computes full inverse-NUFFT outputs
- the task returns large partial arrays
- the orchestrator-side backend mutates live reducer state

That creates the wrong reduction boundary and keeps the hottest mutable state
off the actual owner.

The current distributed durable path is closer to a correct manifest-driven
model, but it still pays too much write amplification when many partials per
chunk become durable shards before the final reduce.

The release problem is therefore not a single symmetric issue:

- local mode is still too close to a driver-accumulator design
- distributed mode is still too close to a shard-heavy many-producer design

The plan must keep that distinction explicit. It should not imply that the
current implementation is already a unified owner-partition reducer with only
policy differences.

## Non-Goals

- no decoder redesign
- no scattering-stage redesign
- no new inverse method or solver rewrite
- no default shared mutable cross-worker reducer state
- no optimization for 1000-node scale

## Release Invariants

These are non-negotiable.

1. Large residual arrays must never move through Dask task results.
2. The driver must never perform residual-array accumulation.
3. Exactly one owner may hold mutable reducer state for a reducer target at a time.
4. Live mutable state is never durable truth.
5. Durable truth is:
   - immutable checkpoint generations
   - reducer progress
   - final committed chunk artifacts
6. Artifact existence alone never implies durable incorporation.
7. Reconciliation is part of correctness, not optional cleanup.
8. Local and distributed modes may share the same reducer contract, but they do not have to share the same physical storage policy.

## Architecture Comparison

| Option | Strongest appeal | Strongest objection | Release stance |
| --- | --- | --- | --- |
| Current driver-accumulator path | smallest code delta | keeps the wrong reduction boundary and returns large arrays through Dask/task boundaries | reject |
| Shared cross-worker memmap / flock | can look attractive for low-chunk single-node runs | shared mutable state, locking, crash recovery, and filesystem behavior become correctness risks | reject as default |
| Many workers per whole chunk, then final reduce | improves utilization when chunk count is low | creates overlapping truths for one chunk and pushes correctness into a late expensive reduce | reject as default |
| Single-owner partition reducer | clean ownership, status-only returns, strongest restart model | needs explicit partitioning, affinity, and immutable checkpoints to work well | target architecture |
| Delay all subchunking until later phases | keeps Phase 1 smaller | underutilizes low-chunk-count local runs and fails over-budget chunk cases too early | reject |

## Final Target Architecture

Use a single-owner partition reducer.

A reducer target is:

- `(chunk_id, parameter_digest)` when whole-chunk ownership is sufficient
- `(chunk_id, partition_id, parameter_digest)` when bounded partitioning is required

For bounded partitions, `partition_id` is part of the release-safe identity,
not an optional planner detail.

Steady-state contract:

```text
driver/orchestrator:
  build partition plan
  assign one owner per reducer target
  submit work to that owner
  receive status/manifests only
  reconcile durable truth
  finalize committed outputs

owner worker:
  load interval input
  run inverse NUFFT
  mutate only its owned live reducer state
  publish immutable checkpoint generations
```

This is the only design in this project scope that is simultaneously defensible
for correctness, performance, restartability, and release risk.

### Status Contract

Tasks return status/manifests only, never residual arrays.

The return object must remain small and immutable and must identify at least:

- reducer target identity
- incorporated interval or interval-batch identity
- contribution accounting needed for progress
- covered point-range or partition identity when bounded partitioning is active
- success/failure state

Residual payloads are incorporated by the owner reducer, not by the task
result channel.

## What Is Architecture vs Policy vs Heuristic

Architecture:

- single-owner partition reduction
- status-only task returns
- immutable checkpoint generations as durable truth
- reconciliation before finalize and on restart/recovery

Policy:

- local vs distributed checkpoint placement
- transport durability requirements
- affinity requirements for live mutable owners
- cleanup and retention behavior

Tuning heuristics:

- cutover from whole-chunk ownership to bounded partitions
- checkpoint cadence
- RAM and scratch budget thresholds

Future optimization:

- adaptive large-3D partition planning beyond the bounded Phase 1/2 path

## Ownership Model

Mutable reducer state is owned by exactly one owner per reducer target.

That owner may be:

- a whole chunk owner
- a disjoint point-range partition owner

This immediately rules out:

- driver-owned live accumulation
- shared mutable memmap/flock as the default design
- many workers mutating overlapping whole-chunk truth

The driver remains responsible for:

- planning
- dispatch
- progress observation
- reconciliation
- finalize orchestration

The driver is not the reducer.

## Partitioning Strategy

### Default

Use whole-chunk ownership when both are true:

- chunk count is high enough to occupy the available workers
- whole-chunk live state and checkpoint generations fit within release budgets

### Required Phase 1 bounded partitioning

Phase 1 must include basic disjoint point-range partitioning.

This is specifically required for the practical 2D one-chunk / multi-worker
case. If one chunk is the only reducer target and whole-chunk ownership is the
only Phase 1 mode, then multiple local workers or GPUs sit idle by design.

Trigger it when either is true:

- `n_chunks < effective_nufft_workers`
- whole-chunk live state or checkpoint size exceeds allowed RAM / local scratch / shared-storage budget

In the first case, the planner must emit multiple disjoint owner partitions
unless prevented by the same budget constraints that would make whole-chunk
ownership invalid.

Phase 1 partitioning must stay minimal:

- disjoint contiguous point-id ranges
- one owner per range
- no overlapping whole-chunk partials
- finalize by concatenation of disjoint outputs

Phase 1 mechanics are intentionally narrow:

- the interval input stays the same
- the reciprocal-space input stays the same
- only the assigned point-id range and its corresponding `real_coords` subset
  change per partition
- the reducer target therefore needs explicit partition identity, not just
  `chunk_id`

This is the minimum implementation shape needed to make the Phase 1 subchunking
requirement concrete.

Concatenation is valid only for disjoint point-range partitions. Overlapping
whole-chunk late reduction is not part of the release architecture.

This is not full adaptive partition planning. It is the minimum needed so:

- low-chunk-count local runs still use the hardware
- oversized chunks have a bounded release path

The exact cutover rule remains a tuning heuristic and must not be written as if
it were already validated science-stage policy.

### Large 3D note

Phase 3 is where adaptive partitioning matures.

But the document must stay honest: if representative release-scope 3D chunks
already exceed whole-chunk budgets, bounded partitioning becomes an earlier
release requirement rather than a later optimization.

## Checkpoint Truth Model

Durable truth must be immutable.

### Durable truth

- immutable checkpoint generations for each reducer target
- reducer progress that records only durably incorporated coverage
- final committed chunk artifacts

### Not durable truth

- live in-memory accumulators
- live file-backed accumulators
- task success alone
- artifact existence alone

### Consequence

If an owner dies after mutating live state but before publishing a committed
generation, recovery must restore from the last immutable generation and
recompute later work.

Immutable generations are mandatory. Live mutable files are scratch, not truth.
Checkpoint cadence is not part of durable truth semantics; it is a policy and
tuning choice that must be measured against restart cost and write
amplification.

The minimum trigger rule is:

- checkpoint on a policy-driven interval of accepted owner updates
- checkpoint before finalize

The exact cadence is tuning, not architecture.

## Reconciliation Model

Finalize-only reconciliation is not sufficient.

Minimum required reconciliation:

1. on restart / recovery
2. before finalize

Reconciliation must answer:

- what coverage is durably incorporated
- what work was attempted but not durably committed
- what must be re-dispatched

Reconciliation must be driven by expected partition coverage plus durable
progress, not by raw artifact existence.

The plan does not require a heavy always-on reconciliation loop in Phase 1, but
it does require enough reconciliation to prevent:

- missing interval coverage
- duplicate incorporation
- finalize from incomplete truth

## Local vs Distributed Policy

The target reducer contract is shared. The physical policy is not identical,
and the current implementation is not yet a unified realization of that target.

Shared invariants do not imply shared Phase 1 plumbing.

### Local policy

- owner-local live accumulator
- RAM-backed when small enough
- file-backed on local scratch only as a bounded fallback when RAM is
  insufficient
- immutable snapshot generations
- direct interval handoff preferred when safe
- no shard-per-partial default

Direct interval handoff is opportunistic transport, not part of reducer
correctness. If it is disabled for safety or memory reasons, the reducer
contract does not change.

File-backed live accumulation is not assumed cheap. Its checkpoint-copy cost is
part of the release acceptance burden, not an architectural promise.

### Distributed policy

- owner-partition reducer still applies
- durable interval transport only when policy requires it
- immutable checkpoint generations on durable shared storage
- reducer progress durable
- final artifacts durable
- shared storage must not be the hot path for every live accumulation step

### Distributed Durable Truth

In the distributed release path, durable truth is:

- immutable committed per-partition generations
- reducer progress referencing durably incorporated generations
- final committed chunk artifacts

Storage roles are:

- owner-local scratch/live state: mutable, non-truth
- shared durable generations and reducer progress: committed truth
- final published chunk artifacts: committed output truth

It is not:

- live owner-local scratch state
- uncommitted transport
- task success alone

Phase 2 may still be transitional in implementation detail, but it must not be
ambiguous about the durable truth unit.

### Affinity rule

Sticky owner placement is mandatory semantically for live mutable
owner-partitions.

The exact scheduler mechanism is implementation-specific. Placement can relax
only for immutable transport or committed checkpoint artifacts.

On retry or owner loss, work must either return to the same owner or resume
from that owner target's last committed generation before continuing.

### Sync Fallback

If Dask is absent or disabled, the same ownership and durable-truth model must
still hold. Only orchestration and transport mechanics change.

## Release Phases

### Phase 1: Correct Boundary and Bounded Local Throughput

Required outcomes:

- no large residual arrays through Dask/task returns
- no driver-side residual accumulation
- single-owner mutable state
- immutable checkpoint generations
- recovery-time and pre-finalize reconciliation
- basic disjoint point-range partitioning for:
  - the one-chunk / multi-worker local case
  - over-budget whole-chunk cases

Phase 1 is the minimum release-correct architecture, not a placeholder.

### Phase 2: Distributed Durable Policy

Required outcomes:

- explicit owner affinity rules
- durable checkpoint placement policy for shared storage
- measured shared-storage write amplification
- bounded shared-storage bytes written per committed partition
- measured checkpoint generation cost
- bounded recompute window from the last committed generation
- cleanup and retention rules consistent with durable truth

This phase makes the same architecture credible on 10-15 nodes.
It is still a hardening phase, not a claim that the distributed owner-partition
path is already fully realized in current code.

### Phase 3: Adaptive Large-Chunk Planning

Required outcomes only if still needed after Phase 1 and Phase 2:

- adaptive partition sizing
- stronger 3D-oriented budget heuristics
- better cutover policy from whole chunk to bounded partitions

Phase 3 is for planner maturity, not for fixing a broken reduction boundary.

## File Touch Points

Primary implementation surfaces for this plan:

- `core/residual_field/contracts.py`
- `core/residual_field/planning.py`
- `core/residual_field/tasks.py`
- `core/residual_field/execution.py`
- `core/residual_field/backend.py`
- `core/residual_field/local_accumulator.py`
- `core/residual_field/artifacts.py`
- `core/scattering/execution.py`
- `core/runtime/worker_hooks.py`
- `core/runtime/dask_helpers.py`

## Hard Release Criteria

Before this ships, all of the following must be demonstrably true.

1. Residual tasks do not return large residual arrays through Dask.
2. The driver does not accumulate residual arrays.
3. Every reducer target has exactly one mutable owner at a time.
4. Reducer target identity is explicit and stable.
5. Durable truth is immutable checkpoint generations plus reducer progress plus final committed artifacts.
6. Artifact existence alone is not treated as durable incorporation.
7. Restart restores from immutable generations, not live mutable state.
8. Reconciliation runs on restart/recovery and before finalize.
9. The one-chunk / multi-worker local case has an explicit bounded partition path and achieves acceptable utilization.
10. Representative 2D one-chunk runs on 2-4 workers avoid shard-per-batch scratch explosion.
11. Representative multi-node runs show bounded shared-storage write amplification per committed partition, bounded checkpoint-generation cost, and bounded finalize latency.
12. Claims about large 3D support are backed by acceptance runs, not heuristics alone.

## Final Decision

Release the residual-field refactor as a single-owner partition reducer with:

- status-only task returns
- immutable checkpoint generations
- mandatory recovery/pre-finalize reconciliation
- Phase 1 basic disjoint point-range partitioning whenever whole-chunk ownership is insufficient

Anything weaker leaves the reduction boundary, restart truth, or release risk in
the wrong place.
