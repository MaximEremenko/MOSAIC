# 1D Examples

This directory contains the two current 1D example cases:

- `displacement/`
- `chemical_ordering/`

Both are small, synchronous examples intended for first runs and CPU-oriented
validation.

## `displacement/`

- structure file: `displacement/slice_displacement_1d.f1d`
- mode: `displacement`
- backend: `synchronous`
- output root: `displacement/output_displacement`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

## `chemical_ordering/`

- structure file: `chemical_ordering/slice_chemical_ordering_1d.f1d`
- mode: `chemical`
- backend: `synchronous`
- output root: `chemical_ordering/output_chemical_ordering`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/chemical_ordering/run_parameters.json
```
