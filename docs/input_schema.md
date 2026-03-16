# Input Schema

MOSAIC runs use two JSON files:

1. `run_parameters.json`
2. `input_parameters.json`

The shipped examples all follow that pattern.

## `run_parameters.json`

This is the small entry file passed to the CLI. In the examples it contains:

```json
{
  "input_parameters_path": "./input_parameters.json"
}
```

Run an example from the repository root with:

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

If MOSAIC is installed as a CLI, the equivalent command is:

```bash
conda run -n mosaic mosaic examples/config_1D/displacement/run_parameters.json
```

## `input_parameters.json`

The current public schema uses these top-level sections:

- `schema_version`
- `paths`
- `structure`
- `reciprocal_space`
- `processing`
- `runtime`

## Section Summary

### `schema_version`

- Integer schema marker.
- All shipped examples use `2`.

### `paths`

Controls where inputs are read from and where outputs are written.

Common keys in the shipped examples:

- `config_root`
- `structure_file`
- `output_directory`

Optional key used by some workflows:

- `average_structure_file`

### `structure`

Describes the structure input.

Common keys:

- `dimension`
- `coefficients`
- `cell_limits`

Notes:

- The examples use `dimension: 1` or `dimension: 2`.
- `structure.coefficients` is optional.
- `structure.coefficients` provides per-site chemical labels or grouping used
  during coefficient-based postprocessing. The loader accepts either `scheme`
  or `source` inside `structure.coefficients`. The shipped examples use both
  forms.

### `reciprocal_space`

Defines what MOSAIC samples in reciprocal space.

Common keys:

- `intervals`
- `mask`

Common `mask` entries:

- `equation`: a symbolic expression evaluated per Q-point (used by 1D and 2D
  examples). In the manuscript this is called a *window* W_F(Q).
- `special_points`: list of named reciprocal-space points with symmetry and
  shape parameters (used by the 3D example)
- `shell_radii`: radial bounds (`r1`, `r2`) for shell-based masking around
  special points (used with `special_points` in the 3D example)

### `processing`

Controls the scientific workflow and postprocessing surface.

Common keys:

- `mode`
- `method`
- `num_chunks`
- `fresh_start`
- `run_postprocessing`
- `points`

The shipped examples use:

- `mode: "displacement"`
- `mode: "chemical"`
- `method: "from_average"`

Optional keys present in some examples:

- `decoder`: configures the `M`-decoder pipeline for displacement mode. The
  `M`-decoder is a trained linear operator that maps local residual-field
  patches to site-resolved displacement vectors. The 2D displacement example
  includes this section and writes an additional decoder output directory.
- `chemical_filtered_ordering`: boolean. When `true`, enables filtered
  chemical-ordering output (scalar occupancy contrast per site). Used by the
  chemical-ordering examples.
- `coefficients`: controls how structure coefficients are applied during
  postprocessing. Keys include `use` (boolean) and `center_by` (centering
  strategy, e.g. `"global"`).

### `runtime`

Controls execution settings.

Common keys:

- `dask`
- `scattering_weights`
- `progress`

Common `runtime.dask` keys:

- `backend`
- `max_workers`
- `threads_per_worker`
- `processes`
- `worker_wait_timeout`

`runtime.scattering_weights` selects the type of scattering factor used in the
amplitude calculation. This is distinct from `structure.coefficients`, which
provides per-site chemical labels for coefficient-based postprocessing.

Supported `kind` values for `runtime.scattering_weights`:

- `ones`: unit weights (default)
- `atomic_number`: Z-based weights
- `neutron`: neutron scattering lengths
- `xray`: X-ray form factors
- `electron`: electron scattering factors

## Example References

Use the shipped examples as the concrete schema reference:

- [../examples/config_1D/displacement/input_parameters.json](../examples/config_1D/displacement/input_parameters.json)
- [../examples/config_1D/chemical_ordering/input_parameters.json](../examples/config_1D/chemical_ordering/input_parameters.json)
- [../examples/config_2D/displacement/input_parameters.json](../examples/config_2D/displacement/input_parameters.json)
- [../examples/config_2D/chemical_ordering/input_parameters.json](../examples/config_2D/chemical_ordering/input_parameters.json)

## Compatibility Note

The loader still accepts older compatibility forms such as:

- `structInfo`
- `peakInfo`
- `rspace_info`
- `runtime_info`

New configs should use the unified schema shown above.
