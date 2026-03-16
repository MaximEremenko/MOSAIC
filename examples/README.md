# Examples

This folder contains the current shipped example surface.

## Start Here

If you want the smallest runnable case, start with:

- `examples/config_1D/displacement/`

Then move to:

- `examples/config_1D/chemical_ordering/`
- `examples/config_2D/displacement/`
- `examples/config_2D/chemical_ordering/`

## Example Families

### `config_1D`

Small CPU-friendly examples for first runs and smoke validation.

Included cases:

- `config_1D/displacement/`
- `config_1D/chemical_ordering/`

Run them with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
conda run -n mosaic python -m core.main examples/config_1D/chemical_ordering/run_parameters.json
```

### `config_2D`

Larger examples with richer masks and larger outputs.

Included cases:

- `config_2D/displacement/`
- `config_2D/chemical_ordering/`

Run them with:

```bash
conda run -n mosaic python -m core.main examples/config_2D/displacement/run_parameters.json
conda run -n mosaic python -m core.main examples/config_2D/chemical_ordering/run_parameters.json
```

The 2D displacement example is configured for `local` Dask execution. The 2D
chemical-ordering example is configured for `cuda-local`.

### `config_3D`

Research-scale 3D examples with large structure files. These require more
memory and compute time than the 1D and 2D cases.

Included cases:

- `config_3D/displacement/` (CaTiO3, `.rmc6f`)
- `config_3D/chemical_ordering/` (LiFeO2, `.rmc6f`)

Run them with:

```bash
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters.json
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters.json
```

Both 3D examples are configured for `local` Dask execution. This folder is not
the recommended starting point for first runs.

## What Each Example Folder Contains

Each runnable example directory contains:

- `run_parameters.json`
- `input_parameters.json`
- a structure file such as `.f1d`, `.f2d`, or `.rmc6f`

## Outputs

Outputs are written to the `paths.output_directory` configured in each
`input_parameters.json`.

Current output roots in the repo:

- `examples/config_1D/displacement/output_displacement/`
- `examples/config_1D/chemical_ordering/output_chemical_ordering/`
- `examples/config_2D/displacement/output_displacement/`
- `examples/config_2D/displacement/output_displacement_decoder_full/`
- `examples/config_2D/chemical_ordering/output_chemical_ordering/`
- `examples/config_3D/displacement/output_displacement/`
- `examples/config_3D/chemical_ordering/output_chemical_ordering/`

Most generated artifacts live under `processed_point_data/`.

## Smoke Check

For the canonical bounded smoke path:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 conda run -n mosaic python scripts/smoke_example.py
```

## Related Docs

- [../docs/input_schema.md](../docs/input_schema.md)
- [../docs/runtime_guide.md](../docs/runtime_guide.md)
