# MOSAIC

MOSAIC is a scientific-stage pipeline for structure loading, patch-center
selection, q-space partitioning, scattering, residual-field materialization,
and decoding.

Release Wave 1 provides the minimum installable and documented baseline for the
current scientific-stage architecture:

- packaging metadata
- a stable CLI/run path
- one canonical smoke-scale example
- baseline installation and runtime notes

This is not yet the full public-release baseline tracked in
[RELEASE_GAPS.md](RELEASE_GAPS.md); licensing, CI, broader example curation,
and governance files remain deferred.

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
- CPU-only execution supported
- GPU acceleration optional
- Dask/distributed execution optional

## Quick Start

The canonical Release Wave 1 example is a small 1D run under [examples/](examples/).

From the repository root:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 python -m core.main examples/run_parameters.json
```

If installed:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 mosaic examples/run_parameters.json
```

Canonical example inputs:

- [examples/run_parameters.json](examples/run_parameters.json)
- [examples/input_parameters.json](examples/input_parameters.json)
- [examples/sample_1d.f1d](examples/sample_1d.f1d)
- [examples/central_points.txt](examples/central_points.txt)

Expected output location:

- `examples/sample_1d_release/processed_point_data/`

## Runtime Notes

- CPU-only is the documented smoke-scale baseline.
- GPU support is optional and adapter-backed via
  [core/adapters/cunufft_wrapper.py](core/adapters/cunufft_wrapper.py).
- Dask is optional. The canonical example does not require a distributed
  cluster.
- Memory use scales mainly with reciprocal-space interval size, chunk size, and
  worker concurrency in:
  - `scattering`
  - `residual_field`
  - `decoding`
- Larger 2D/3D examples under `examples/` are not the Release Wave 1
  reproducibility target.

## Input and Output Overview

Canonical input schema keys:

- `structInfo`
  - structure dimension, structure file, working directory
- `peakInfo`
  - reciprocal-space interval definitions and mask settings
- `rspace_info`
  - patch-center method, chunking, point-selection inputs, and processing mode
- `runtime_info`
  - runtime and form-factor settings

Primary output area:

- `<working_directory>/processed_point_data/`

Important artifact families:

- q-space interval metadata and SQLite state
- scattering interval artifacts
- residual-field chunk HDF5 artifacts
- decoding outputs such as displacement or site-intensity tables

## Package Layout

The installable package follows the scientific workflow:

- `core.config`
- `core.structure`
- `core.patch_centers`
- `core.qspace`
- `core.scattering`
- `core.residual_field`
- `core.decoding`
- `core.storage`
- `core.runtime`
- `core.adapters`
- `core.workflow`

## Examples

Start with [examples/README.md](examples/README.md). The repository still
contains many legacy and research assets under `examples/`; not every file
there is a supported release example.
