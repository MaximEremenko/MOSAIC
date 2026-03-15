# M-Decoder Patch Consistency Plan

This note explains the current decoder-size inconsistency risk in displacement
mode, compares it to the manuscript requirements, and proposes the smallest
sound implementation path.

It is complementary to:

- [qspace_equation_mask_plan.md](qspace_equation_mask_plan.md)
- [input_schema.md](input_schema.md)
- the manuscript text under `/mnt/c/Projects/MOSAICPaper/Text`

## Manuscript Requirement

The manuscript is clear that the displacement decoder is built from a **fixed**
local feature operator.

Relevant statements:

- [Methods.tex](../Text/Methods/Methods.tex): fixed neighborhood, fixed
  pre-processing, fixed linear feature map `r_s = L[...]`
- [Methods.tex](../Text/Methods/Methods.tex): one global decoder `M`, or
  optionally a small class-conditional family `{M_alpha}`
- [mainM_decoder.tex](../Text/mainM_decoder.tex): local patches are extracted on
  a fixed neighborhood around each site and mapped through one fixed decoder, or
  a mask-independent class-conditional family

This means:

1. a single decoder `M` is only mathematically coherent if all training and
   inference patches live in the **same feature space**
2. if different site classes need different local patch geometry, the correct
   extension is **multiple decoders** `{M_alpha}`, with assignment fixed
   independently of the mask

It does **not** mean:

- one decoder per mask
- one decoder trained opportunistically on whatever patch shapes happen to be in
  the current run

## Current Code Problem

### Where patch geometry enters

Patch size is determined by point-selection settings:

- `referenceNumber`
- `distFromAtomCenter`
- `stepInAngstrom`

in [core/patch_centers/from_average.py](../core/patch_centers/from_average.py).

Those values are written into `PointData` as:

- `dist_from_atom_center`
- `step_in_frac`

and later used to build the local RIFFT grid in:

- [core/patch_centers/point_data.py](../core/patch_centers/point_data.py)
- [core/patch_centers/local_grid.py](../core/patch_centers/local_grid.py)

### Where feature size is determined

During decoding:

- [core/decoding/grid.py](../core/decoding/grid.py) reconstructs a dense local
  patch from scattered residual-field samples
- [core/decoding/features.py](../core/decoding/features.py) flattens the
  centered odd component of that patch into a feature vector

So feature length depends on:

- local patch shape
- therefore on `dist_from_atom_center`
- and on `step_in_frac`

### What the current code does

[core/decoding/decoder_service.py](../core/decoding/decoder_service.py) builds
one list of feature vectors in `build_feature_sets(...)`, then trains one global
decoder in `train_decoder_from_samples(...)`, and later checks:

- [core/decoding/decoder_service.py](../core/decoding/decoder_service.py): all
  features must match one `feature_dim`

So today:

- mixed patch geometries are allowed into the pipeline
- but they are only rejected later by a generic
  `"Feature dimension mismatch"` error

That is too late and too implicit.

### Why the issue is real

[core/patch_centers/from_average.py](../core/patch_centers/from_average.py)
currently allows multiple `points` entries with the same:

- element
- reference number

but different:

- `distFromAtomCenter`
- `stepInAngstrom`

Those entries are simply concatenated into one `PointData`.

This means the current displacement path can silently mix incompatible patch
operators before eventually failing.

## Correct Interpretation

The problem is **not** masks.

Masks only change the input residual field `R_F(r)`.

The decoder inconsistency comes from changing the **local patch operator**:

- different patch extent
- different patch sampling
- therefore different feature dimension and feature meaning

So if patch volume differs, the correct question is:

- single decoder or decoder family?

not:

- one decoder per mask?

## Implementation Stages

### Stage 1: Explicit PatchSpec and Strict Early Validation

This is the immediate correctness fix and the only blocking phase for the
current single-global-decoder path.

For the current global-decoder path:

- all displacement points in one decoder run must share one patch spec

Define a patch spec as at least:

- dimensionality
- `dist_from_atom_center`
- `step_in_frac`
- fixed feature-construction settings that affect `r_s`
  - `q_window_kind`
  - `q_window_at_db`
  - `edge_guard_frac`
  - `ls_weight_gamma`
  - feature-construction version

Implementation rules:

- collect all patch specs before decoder training or inference proceeds
- if more than one unique patch spec exists, fail **early**
- fail before residual-field reload, feature construction, decoder cache use,
  decoder training, or decoder inference

Extra strict rule for same reference number:

- if the same `referenceNumber` appears with multiple patch specs, fail with a
  dedicated actionable error

Reason:

- same reference number usually means the same crystallographic site class
- silently assigning different patch operators to the same site class is almost
  certainly a configuration mistake

Stage 1 closure criteria:

1. an explicit patch-spec concept exists in the decoding path
2. mixed patch specs fail before the late generic feature-dimension mismatch
   path
3. same-`referenceNumber` mixed specs raise a dedicated actionable error
4. valid single-patch-spec runs are unchanged
5. targeted unit coverage exists for all of the above

### Stage 2: Proper Decoder Family Support

This is the correct long-term extension once mixed patch geometry is genuinely
required.

Implement a **mask-independent decoder family** `{M_alpha}`.

Pragmatic implementation note:

- Stage 2 may be introduced as an explicit decoder-family mode so the current
  single-global-decoder path remains unchanged by default
- Stage 3 remains the place for a full persisted family cache/provenance format

Each decoder key should be based on:

1. site class
2. patch spec

Recommended decoder key:

```text
decoder_key = (
    site_class_key,
    patch_spec_key,
)
```

Where:

- `site_class_key` could default to:
  - `reference_number`
  - or an explicit user-supplied class label later
- `patch_spec_key` includes:
  - dimension
  - `dist_from_atom_center`
  - `step_in_frac`
  - feature-construction settings

Important manuscript constraint:

- class assignment must be fixed independently of the mask
- decoder selection by `reference_number` or explicit site class is valid
- decoder selection based on the masked field is **not** valid if you want to
  preserve additivity

Decoder family behavior:

- training:
  - group samples by `decoder_key`
  - train one `M` per key
- inference:
  - assign each central point the same `decoder_key`
  - apply that decoder only

Immediate Stage 2 closure criteria:

1. mixed patch geometries are supported when decoder-family mode is used
2. decoder keys are formed deterministically from site class + patch spec
3. decoder assignment remains fixed independently of the mask
4. single-decoder runs remain unchanged
5. full persisted family cache/provenance redesign is still deferred to Stage 3

This is exactly the manuscript’s `{M_alpha}` picture.

### Stage 3: Cache / Provenance Support for Decoder Families

This stage is needed only after Stage 2 exists.

Expected work:

- decoder-bundle cache format or one cache file per `decoder_key`
- provenance that records:
  - trained keys
  - patch specs per key
  - site classes per key

Stage 3 is not required to close the immediate correctness gap in the current
single-global-decoder path.

## What Not to Do

### Do not train one decoder per mask

That breaks the intended method structure.

The manuscript explicitly says:

- train on the unmasked field
- reuse the same decoder (or fixed decoder family) for all masks

### Do not force same patch volume by atom type only

That is too coarse.

Different Wyckoff/reference classes can share an element but still need
different local neighborhoods.

If you need a grouping key, `reference_number` is a better default than element
alone.

### Do not silently resample mixed patch shapes into one common vector

That may be possible later, but it is a different modeling choice.

For now it would hide a conceptual inconsistency rather than solve it clearly.

## Cache / Provenance Implication

Current decoder cache paths are built in:

- [core/decoding/decoder_cache.py](../core/decoding/decoder_cache.py)

The current cache key does **not** include patch-spec grouping or decoder-family
structure. That is fine for a single global decoder but insufficient for a
decoder family.

If Phase 2 is implemented, the cache format should become a decoder bundle, for
example:

- one file containing:
  - decoder keys
  - one matrix `M` per key
  - one `feature_dim` per key
- or one cache file per decoder key under one bundle directory

Either way, provenance must record:

- which keys were trained
- which patch specs they correspond to
- which site classes they correspond to

## Recommended Implementation Order

1. Stage 1: early validation and fail-fast correctness for single-global-decoder mode
2. Stage 2: decoder-family support `{M_alpha}` keyed by site class + patch spec if truly needed
3. Stage 3: cache/provenance bundle support for decoder families

## Bottom Line

The immediate correctness gap is closed when Stage 1 is done:

1. a single global decoder can no longer silently consume mixed patch specs
2. same-`referenceNumber` mixed patch specs are rejected clearly and early
3. the old late generic feature-dimension mismatch is no longer the first
   failure mode for this problem

After that, any remaining work is either:

- Stage 2 / Stage 3 decoder-family support for genuinely mixed patch geometry
- or optional extension work that does not block correctness

That is the cleanest bounded closure path that stays faithful to the manuscript
and avoids silent size inconsistency bugs.
