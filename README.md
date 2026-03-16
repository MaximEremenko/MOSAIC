# MOSAIC

MOSAIC is a computational framework for linking selected diffuse-scattering
features to site-resolved chemical-order and displacement fields in atomistic
configurations.

Starting from a structural configuration, MOSAIC computes the complex
scattering amplitude, isolates reciprocal-space features with phase-preserving
masks, applies an inverse Fourier transform to recover atom-centered real-space
fields, and decodes those fields into site-resolved outputs associated with the
chosen diffuse features.

The scientific method is described in the accompanying manuscript:

> **A Filter-Based Approach Linking Diffuse Scattering Features to
> Site-Resolved Real-Space Fields**

## Method Overview

MOSAIC implements a strictly linear, phase-preserving workflow:

1. Compute the total scattering amplitude and the diffuse amplitude by
   subtracting the coherent average-structure contribution.
2. Apply one or more reciprocal-space masks (called *windows* in the
   manuscript) to select diffuse features of interest.
3. Evaluate a restricted inverse Fourier transform (RIFFT) at atom-centered
   points (chemical mode) or local patches (displacement mode) to obtain
   residual real-space fields.
4. Decode those fields into site-resolved observables.

Two output modes are supported:

- **Chemical mode** (also called *average-position mode*): recover a scalar
  site-resolved chemical or occupancy contrast associated with the selected
  reciprocal-space feature.
- **Displacement mode**: recover site-resolved displacement vectors through the
  linear `M`-decoder.

Because the workflow is linear in the diffuse amplitude, contributions from
disjoint reciprocal-space selections are additive. This makes it possible to
analyze diffuse features independently and compare or recombine their
real-space maps in a controlled way.

## Scientific Scope

MOSAIC is intended for atomistic configurations obtained from:

- Reverse Monte Carlo refinements
- molecular dynamics or Monte Carlo simulations
- 2D structural projections from STEM or related imaging workflows

All scattering calculations use the kinematic (single-scattering)
approximation.

The codebase supports 1D, 2D, and 3D structural cases, with the same overall
scientific logic applied across those geometries. The repository currently
ships bounded 1D and 2D runnable examples, plus 3D reference inputs for
research-scale use.

## Installation

Conda is the recommended path:

```bash
conda env create -f core/environment.yml
conda activate mosaic
pip install -e .
```

Minimal editable install in an existing Python environment:

```bash
pip install -e .
```

Practical requirements:

- Python 3.11+
- Linux or WSL
- CPU-only execution works out of the box
- GPU acceleration and Dask-based parallel execution are optional

## Quick Start

Run the bounded 1D displacement example from the repository root:

```bash
python -m core.main examples/config_1D/displacement/run_parameters.json
```

If the console entry point is installed:

```bash
mosaic examples/config_1D/displacement/run_parameters.json
```

Expected output location:

- `examples/config_1D/displacement/output_displacement/processed_point_data/`

Bounded smoke helper:

```bash
python scripts/smoke_example.py
```

## Shipped Examples

Runnable example packages currently include:

| Directory | Dimensionality | Mode |
|-----------|---------------|------|
| `examples/config_1D/displacement/` | 1D | Displacement |
| `examples/config_1D/chemical_ordering/` | 1D | Chemical ordering |
| `examples/config_2D/displacement/` | 2D | Displacement |
| `examples/config_2D/chemical_ordering/` | 2D | Chemical ordering |
| `examples/config_3D/` | 3D | Reference inputs only |

See [examples/README.md](examples/README.md) for the current example surface
and the distinction between runnable examples and larger research material.

## Configuration Model

MOSAIC uses a JSON input model centered on five top-level sections:

- `paths`: structure file, config root, output directory
- `structure`: dimensionality, coefficients, lattice-related inputs
- `reciprocal_space`: interval definitions and mask equations
- `processing`: output mode, patch-center method, chunking, decoder settings
- `runtime`: execution backend, worker settings, and scattering-weight
  configuration

Schema reference:

- [docs/input_schema.md](docs/input_schema.md)

## Package Surface

The scientific workflow is organized by stage:

- `core.structure`: configuration, lattice, and coefficient handling
- `core.patch_centers`: atom-centered evaluation-point selection
- `core.qspace`: reciprocal-space partitioning and masking
- `core.scattering`: forward scattering amplitude (type-3 NUFFT via finufft/cufinufft)
- `core.residual_field`: inverse reconstruction of residual fields
- `core.decoding`: site-resolved decoding of chemical or displacement outputs

Supporting layers:

- `core.runtime`: execution backend and Dask orchestration
- `core.storage`: persistence and intermediate artifacts
- `core.workflow`: stage sequencing
- `core.adapters`: backend-specific integrations

## Runtime Notes

- CPU-only execution can be forced with `MOSAIC_NUFFT_CPU_ONLY=1`.
- The default `synchronous` execution mode runs in-process.
- Local or distributed Dask execution is optional and configured through the
  runtime section of the input JSON.
- Memory use depends strongly on reciprocal-space interval size, chunking, and
  worker concurrency.

## Documentation

Start here:

- [docs/README.md](docs/README.md)
- [docs/input_schema.md](docs/input_schema.md)
- [docs/runtime_guide.md](docs/runtime_guide.md)
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [examples/README.md](examples/README.md)

## Citation

If you use MOSAIC in your work, please cite:

```bibtex
@article{mosaic_paper,
  author  = {Eremenko, Maksim},
  title   = {A Filter-Based Approach Linking Diffuse Scattering Features
             to Site-Resolved Real-Space Fields},
  year    = {2026}
}
```

Software citation:

```bibtex
@software{mosaic,
  author    = {Eremenko, Maksim},
  title     = {MOSAIC},
  version   = {0.1.0},
  url       = {https://github.com/MaximEremenko/MOSAIC}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable metadata.

## License

Apache-2.0. See [LICENSE](LICENSE).
