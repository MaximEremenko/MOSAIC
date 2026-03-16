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
- mask: superlattice sphere at 1/2(h), r = 0.25 rlu
- output root: `displacement/output_displacement`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

### Linearity demonstration

Two additional mask variants demonstrate that contributions from disjoint
reciprocal-space selections are additive: `R(mask) + R(rest) = R(all)`.

| Config | Mask | Description |
|--------|------|-------------|
| `input_parameters.json` | Feature | Superlattice sphere `(Mod(h,1)−0.5)² ≤ 0.25²` |
| `input_parameters_rest.json` | ~Feature | Everything outside the sphere |
| `input_parameters_all.json` | Full | All reciprocal-space points |

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters_rest.json
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters_all.json
```

## `chemical_ordering/`

- structure file: `chemical_ordering/slice_chemical_ordering_1d.f1d`
- mode: `chemical`
- backend: `synchronous`
- mask: superlattice sphere at 1/2(h), r = 0.198 rlu
- output root: `chemical_ordering/output_chemical_ordering`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/chemical_ordering/run_parameters.json
```

### Linearity demonstration

| Config | Mask | Description |
|--------|------|-------------|
| `input_parameters.json` | Feature | Superlattice sphere `(Mod(h,1)−0.5)² ≤ 0.198²` |
| `input_parameters_rest.json` | ~Feature | Everything outside the sphere |
| `input_parameters_all.json` | Full | All reciprocal-space points |

```bash
conda run -n mosaic python -m core.main examples/config_1D/chemical_ordering/run_parameters_rest.json
conda run -n mosaic python -m core.main examples/config_1D/chemical_ordering/run_parameters_all.json
```
