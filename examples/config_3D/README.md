# 3D Examples

This directory contains two 3D example cases:

- [displacement/](displacement)
- [chemical_ordering/](chemical_ordering)

These are research-scale examples with large structure files. They require
more memory and compute time than the 1D and 2D cases and are not recommended
as a starting point for first runs.

## `displacement/`

- structure file: `displacement/Catio3.rmc6f`
- average structure: `displacement/Catio3_average.rmc6f`
- mode: `displacement`
- backend: `local`
- checked-in runtime: `max_workers: 2`, `threads_per_worker: 16`, `processes: false`
- output root: `displacement/output_displacement`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters.json
```

### Linearity demonstration (displacement)

Four mask variants partition the reciprocal space around 1/2(111) superlattice
features into disjoint regions. The rod mask selects a cylinder of radius
r1 = 0.1876 rlu in the (h,k) plane along l; the sphere mask selects the
interior of radius r2 = 0.2501 rlu. These are constructed so that
`u(rod\sphere) + u(sphere) + u(rest) = u(all)`.

| Config | Mask | Description |
|--------|------|-------------|
| `input_parameters_rod.json` | Rod \ Sphere | Annular rod along (1/2, 1/2, l) excluding sphere (tilt/rotation modes) |
| `input_parameters_sphere.json` | Sphere | Interior of r2 sphere at 1/2(111) (breathing modes) |
| `input_parameters_rest.json` | ~(Rod ∪ Sphere) | Outside both rod and sphere (background) |
| `input_parameters_all.json` | Full volume | All reciprocal-space points (unmasked reference) |

Run each variant:

```bash
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters_rod.json
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters_sphere.json
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters_rest.json
conda run -n mosaic python -m core.main examples/config_3D/displacement/run_parameters_all.json
```

The linearity check: summing per-site displacement vectors from rod, sphere,
and rest outputs should match the all output exactly (up to floating-point
precision).

## `chemical_ordering/`

- structure file: `chemical_ordering/LiFeO2.rmc6f`
- mode: `chemical`
- backend: `local`
- checked-in runtime: `max_workers: 2`, `threads_per_worker: 16`, `processes: false`
- output root: `chemical_ordering/output_chemical_ordering`

Default run (superlattice spheres mask):

```bash
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters.json
```

### Linearity demonstration

Four additional mask variants are provided to demonstrate that contributions
from disjoint reciprocal-space selections are additive. The masks are
constructed so that `R(combined) = R(spheres) + R(cos)` and
`R(full) = R(combined) + R(rest)`.

| Config | Mask | Description |
|--------|------|-------------|
| `input_parameters_spheres.json` | S | Superlattice 1/2(111) spheres (nanodomain filter, r = 0.1576 rlu) |
| `input_parameters_cos.json` | C \ S | Cosine iso-surface `cos(pi*h)+cos(pi*k)+cos(pi*l) ~ 0` excluding spheres (SRO filter) |
| `input_parameters_combined.json` | S ∪ C | Union of spheres and cosine surface |
| `input_parameters_rest.json` | ~(S ∪ C) | Everything not in spheres or cosine surface (residual) |

Run each variant with its corresponding `run_parameters_*.json`:

```bash
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters_spheres.json
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters_cos.json
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters_combined.json
conda run -n mosaic python -m core.main examples/config_3D/chemical_ordering/run_parameters_rest.json
```

The linearity check: summing per-site scalars from spheres and cos outputs
should match the combined output exactly (up to floating-point precision).
