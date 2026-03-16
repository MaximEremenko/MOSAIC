# Runtime Guide

This guide covers the runtime settings used by the shipped examples.

## How To Run Examples

From the repository root:

```bash
conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

Installed CLI equivalent:

```bash
conda run -n mosaic mosaic examples/config_1D/displacement/run_parameters.json
```

## Backends Used By The Examples

The checked-in examples use three Dask backend modes:

- `synchronous`: in-process execution
- `local`: local `dask.distributed.LocalCluster`
- `cuda-local`: local `dask_cuda.LocalCUDACluster`

Current example runtime profiles:

- `examples/config_1D/displacement`: `synchronous`, `max_workers: 1`
- `examples/config_1D/chemical_ordering`: `synchronous`, `max_workers: 1`
- `examples/config_2D/displacement`: `local`, `max_workers: 6`, `processes: true`
- `examples/config_2D/chemical_ordering`: `cuda-local`, `max_workers: 1`

## Choosing A Starting Point

Use the examples in this order:

1. `examples/config_1D/displacement`
2. `examples/config_1D/chemical_ordering`
3. `examples/config_2D/displacement`
4. `examples/config_2D/chemical_ordering`

The 1D examples are the smallest and best for installation checks.

## NUFFT Backend

MOSAIC uses a type-3 non-uniform FFT (NUFFT) for both forward and inverse
transforms. The underlying library is [finufft](https://finufft.readthedocs.io/)
on CPU and cufinufft on GPU. The type-3 transform maps non-uniform source
points (atom positions) to non-uniform target points (reciprocal-space samples
or atom-centered evaluation grids).

## CPU-Only Runs

To force the NUFFT layer onto CPU:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

This is the safest baseline when you want to separate environment issues from
workflow issues.

## GPU Runs

Only the shipped 2D chemical-ordering example is configured for `cuda-local`.
A typical single-GPU launch is:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mosaic python -m core.main examples/config_2D/chemical_ordering/run_parameters.json
```

Notes:

- `cuda-local` requires `dask-cuda` in the active environment.
- GPU VRAM pressure and host RAM pressure are separate limits.
- If a GPU run is unstable, reduce worker count first and then retry.

## Worker Count Guidance

Practical rules:

- Keep `synchronous` for small validation runs.
- Use modest worker counts for `local` because each worker can hold large arrays.
- Do not assume more workers will help on a single GPU.

If you see Dask memory pause or resume warnings, reduce `runtime.dask.max_workers`
before changing anything else.

## Output Locations

Outputs are written under `paths.output_directory` from the active
`input_parameters.json`.

Current example outputs:

- `examples/config_1D/displacement/output_displacement`
- `examples/config_1D/chemical_ordering/output_chemical_ordering`
- `examples/config_2D/displacement/output_displacement`
- `examples/config_2D/displacement/output_displacement_decoder_full`
- `examples/config_2D/chemical_ordering/output_chemical_ordering`

Most generated artifacts live under `processed_point_data/`. Some modes also
produce `residual_shards/` for intermediate accumulation data.
