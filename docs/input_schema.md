# Canonical Input Schema

This document defines the canonical release-candidate input schema for MOSAIC.
It matches the `schema_version: 2` format used in
[examples/input_parameters.json](/mnt/c/Projects/MOSAICPaper/MOSAIC/examples/input_parameters.json).

The code still accepts older compatibility payloads such as `structInfo`,
`peakInfo`, `rspace_info`, and `runtime_info`, but those should be treated as
legacy forms. Public-facing examples and new workflows should use the unified
schema below.

## Top-Level Keys

### `schema_version`

- Integer schema marker.
- Release-candidate examples use `2`.

### `paths`

Defines file-system inputs and outputs.

Supported keys:

- `config_root`
  - Base directory for resolving relative structure, coefficient, and point
    selector files.
- `structure_file`
  - Structure/configuration file to load.
- `output_directory`
  - Working directory for generated release outputs.
- `average_structure_file`
  - Optional average-structure file where the selected processing mode requires
    it.

### `structure`

Defines structure-specific inputs.

Supported keys:

- `dimension`
  - `1`, `2`, or `3`.
- `coefficients`
  - Optional coefficient source information.
- `cell_limits`
  - Optional `min` / `max` bounds when a cropped cell window is required.

### `reciprocal_space`

Defines the reciprocal-space regions and masks to evaluate.

Supported keys:

- `intervals`
  - List of reciprocal-space subsets to sample.
  - In the canonical example each interval uses:
    - `limit`
    - `subvolume_step`
- `mask`
  - Reciprocal-space selection logic.
  - Common keys include:
    - `equation`
    - `special_points`
    - `shell_radii`

Conceptually, these windows correspond to the manuscript’s
phase-preserving mask `W_F(Q)` applied to selected diffuse-scattering features.

### `processing`

Defines how MOSAIC converts reciprocal-space selections into site-resolved
outputs.

Supported keys:

- `mode`
  - Scientific interpretation mode.
  - Common values in the current codebase:
    - `displacement`
    - chemical/average-position style workflows through the same compatibility
      layer
- `method`
  - Patch-center / point-selection method.
  - Example: `central`
- `num_chunks`
  - Number of chunk groups for stage accumulation.
- `fresh_start`
  - Whether to remove prior generated output for the run.
- `run_postprocessing`
  - Whether the decoder/output stage should run after residual-field creation.
- `points`
  - Point-selection entries, each with:
    - `selector`
      - e.g. `file`
    - `window`
      - `dist_from_atom_center`
      - `step_in_angstrom`

### `runtime`

Defines runtime and form-factor settings.

Supported keys:

- `dask`
  - Runtime backend configuration.
  - Common keys:
    - `backend`
    - `max_workers`
    - `threads_per_worker`
    - `processes`
    - `worker_wait_timeout`
- `form_factor`
  - Scattering-factor family and calculator choice.
  - Common keys:
    - `family`
    - `calculator`

## Canonical Smoke Example

The release-candidate smoke path uses the following minimal pattern:

```json
{
  "schema_version": 2,
  "paths": {
    "config_root": ".",
    "structure_file": "sample_1d.f1d",
    "output_directory": "./sample_1d_release"
  },
  "structure": {
    "dimension": 1
  },
  "reciprocal_space": {
    "intervals": [
      {
        "limit": [1.0],
        "subvolume_step": [1.0]
      }
    ],
    "mask": {
      "equation": "h**2>=-1"
    }
  },
  "processing": {
    "mode": "displacement",
    "method": "central",
    "num_chunks": 1,
    "fresh_start": true,
    "run_postprocessing": false
  },
  "runtime": {
    "dask": {
      "backend": "local",
      "max_workers": 1,
      "threads_per_worker": 1,
      "processes": false,
      "worker_wait_timeout": "30s"
    },
    "form_factor": {
      "family": "neutron",
      "calculator": "default"
    }
  }
}
```

See [examples/input_parameters.json](/mnt/c/Projects/MOSAICPaper/MOSAIC/examples/input_parameters.json)
for the exact supported smoke-scale example.

## Legacy Compatibility Mapping

The loader in [core/config/schema.py](/mnt/c/Projects/MOSAICPaper/MOSAIC/core/config/schema.py)
still maps older payloads into the canonical internal structure. In particular:

- `structInfo` maps to `paths + structure`
- `peakInfo` maps to `reciprocal_space`
- `rspace_info` maps to `processing`
- `runtime_info` maps to `runtime`

That compatibility exists for historical inputs, but new release-facing
examples should stay on the unified schema.
