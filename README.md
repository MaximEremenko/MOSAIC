# MOSAIC

MOSAIC is a method and software workflow for connecting selected
diffuse-scattering features to site-resolved occupancy and displacement fields
in atomistic configurations. Starting from an atomic configuration, MOSAIC
computes the complex scattering amplitude, isolates reciprocal-space features
with phase-preserving masks, and applies an inverse Fourier transform to
recover atom-centered residual real-space fields that can be interpreted in
chemical or displacement modes.

Release Waves 1-3 provide the current release baseline for the
scientific-stage architecture:

- packaging metadata
- a stable CLI/run path
- one bounded canonical smoke-scale example
- baseline installation and runtime notes
- one CI workflow for smoke and unit validation
- baseline changelog, citation, and contributing metadata

This is not yet the full public-release baseline tracked in
[RELEASE_GAPS.md](RELEASE_GAPS.md); licensing, broader example curation, and
broader governance files remain deferred.

## What MOSAIC Computes

The method described in the manuscript and TeX sources under
`/mnt/c/Projects/MOSAICPaper/Text` follows this scientific workflow:

1. start from an atomistic configuration obtained from experiment or simulation
2. compute the total scattering amplitude and the diffuse scattering amplitude
   `ΔA(Q)` by subtracting the average-structure term
3. apply one or more phase-preserving reciprocal-space masks `W_F(Q)` to select
   halos, rods, superlattice features, wings, or other regions of interest
4. evaluate the inverse Fourier transform on atom-centered points or patches to
   obtain a residual real-space field `R_F(r)`
5. recover site-resolved chemical-order or displacement fields associated with
   the selected diffuse features

This linear, phase-preserving formulation is the reason MOSAIC can decompose
disjoint reciprocal-space selections and accumulate them through the
Map-Reduce-style runtime used by the current codebase.

## Scientific Stages

The installable package mirrors that scientific workflow:

1. `core.structure`: load the configuration, lattice, and optional coefficients
2. `core.patch_centers`: choose the atom-centered origins or average positions
3. `core.qspace`: define reciprocal-space partitions and masks
4. `core.scattering`: compute the scattering or diffuse amplitude on the chosen
   `Q` points
5. `core.residual_field`: apply the inverse Fourier transform and assemble
   atom-centered residual fields
6. `core.decoding`: convert those residual fields into site-resolved
   displacement or occupancy outputs

Runtime, storage, and adapters remain separate support layers:

- `core.runtime`: Dask/client/runtime ownership
- `core.storage`: SQLite and HDF5 persistence
- `core.adapters`: optional GPU/backend-specific helpers
- `core.workflow`: orchestration of the scientific stages above

## Install

Release Wave 1 supports two install stories.

Conda-first baseline:

```bash
conda env create -f core/environment.yml
conda activate MOSAIC
python -m pip install -e .
```

Editable pip install in an existing Python 3.11 environment:

```bash
python -m pip install -e .
```

Documented baseline for this wave:

- Python 3.11
- Linux/WSL-like environments
- CPU-only execution supported and treated as the release baseline
- GPU acceleration optional
- Dask/distributed execution optional

## Quick Start

The canonical release example is a small 1D run under [examples/](examples/).

From the repository root:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 python -m core.main examples/run_parameters.json
```

If installed:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 mosaic examples/run_parameters.json
```

Supported release-candidate validation command:

```bash
/home/f3e/anaconda3/envs/mosaic/bin/python scripts/smoke_example.py
```

Canonical example inputs:

- [examples/run_parameters.json](examples/run_parameters.json)
- [examples/input_parameters.json](examples/input_parameters.json)
- [examples/sample_1d.f1d](examples/sample_1d.f1d)
- [examples/central_points.txt](examples/central_points.txt)

Expected output location:

- `examples/sample_1d_release/processed_point_data/`

Release validation smoke helper:

```bash
python scripts/smoke_example.py
```

The smoke helper is the bounded CI-facing path. It removes any prior
`examples/sample_1d_release/` output, runs the canonical example with a hard
timeout, and checks for the expected artifact set. In the intended `mosaic`
environment this is the supported smoke-scale release-candidate validation
path.

## Runtime Notes

- CPU-only is the documented smoke-scale baseline.
- GPU support is optional and adapter-backed via
  [core/adapters/cunufft_wrapper.py](core/adapters/cunufft_wrapper.py).
- Dask is optional. The canonical example does not require a distributed
  cluster.
- The canonical example is pinned to one local worker and one thread in its
  shipped runtime settings so the smoke path stays bounded and repeatable.
- If `finufft` is unavailable, the smoke-scale path can fall back to a much
  slower direct CPU evaluation. That fallback is for smoke validation only,
  not production-scale runs.
- The canonical smoke example is expected to finish in seconds to low minutes
  on a normal workstation-class CPU, with output under
  `examples/sample_1d_release/processed_point_data/`.
- Memory use scales mainly with reciprocal-space interval size, chunk size, and
  worker concurrency in:
  - `scattering`
  - `residual_field`
  - `decoding`
- Larger 2D/3D examples under `examples/` are not the current release
  reproducibility target.

## Input and Output Overview

The canonical release-candidate input schema is the unified `schema_version: 2`
format used by [examples/input_parameters.json](examples/input_parameters.json).
It is documented in [docs/input_schema.md](docs/input_schema.md). Public docs
should treat that schema as primary; the older `structInfo` / `peakInfo` /
`rspace_info` / `runtime_info` forms remain compatibility inputs only.

Canonical top-level schema keys:

- `paths`
  - configuration root, structure file, and output directory
- `structure`
  - dimensionality and structure-specific inputs
- `reciprocal_space`
  - interval definitions and mask settings
- `processing`
  - mode, patch-center selection, chunking, and whether decoding runs
- `runtime`
  - local/Dask settings and form-factor configuration

Primary output area:

- `<working_directory>/processed_point_data/`

Important artifact families:

- q-space interval metadata and SQLite state
- scattering interval artifacts and chunk accumulations
- residual-field chunk HDF5 artifacts produced by the inverse Fourier
  transform
- decoding outputs such as displacement or site-intensity tables

## Examples

Start with [examples/README.md](examples/README.md). The repository still
contains many legacy and research assets under `examples/`. For this release
candidate, the supported example surface is intentionally narrow:

- the canonical 1D example files listed above
- the smoke script in [scripts/smoke_example.py](scripts/smoke_example.py)
- the documented run commands `python -m core.main ...` and `mosaic ...`

Larger 2D/3D configurations, old `main*.py` example drivers, plotting projects,
and exploratory helper scripts remain research material and are not part of the
supported release-candidate path.
