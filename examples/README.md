# Examples

This directory contains one supported release-candidate example and a larger set
of research, historical, and exploratory material.

## Supported Release-Candidate Example

- [run_parameters.json](run_parameters.json)
- [input_parameters.json](input_parameters.json)
- [sample_1d.f1d](sample_1d.f1d)
- [central_points.txt](central_points.txt)

Run it from the repository root:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 python -m core.main examples/run_parameters.json
```

Release smoke helper:

```bash
python scripts/smoke_example.py
```

This helper is the CI-facing smoke path. It recreates the canonical output
tree and checks for the expected artifact set under a fixed timeout.

Expected output location:

- `examples/sample_1d_release/processed_point_data/`

Canonical smoke-scale runtime assumptions:

- CPU-only baseline
- local backend
- one worker
- one thread per worker
- one chunk
- `run_postprocessing: false`

## Canonical Schema

The supported example uses the unified release schema documented in
[docs/input_schema.md](../docs/input_schema.md).

## Research and Historical Material

The rest of `examples/` is not part of the supported release-candidate surface.
That includes, for example:

- `config_2D/` and `config_3D/`
- legacy `main*.py` drivers
- cluster helper scripts such as `run_slurm.sh`, `run_sge.sh`, and
  `run_local_gpu.sh`
- `.opju` plotting/project files
- exploratory SymPy and plotting helpers
- large `.rmc6f` research inputs

Those files remain useful reference material, but they are not validated by the
canonical smoke path and should not be treated as release-grade examples.
