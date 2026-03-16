# Troubleshooting

This page covers the most common issues with the shipped examples.

## Run The Smallest Example First

Start from the repository root with:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 conda run -n mosaic python -m core.main examples/config_1D/displacement/run_parameters.json
```

If that works, move up through:

1. `examples/config_1D/chemical_ordering`
2. `examples/config_2D/displacement`
3. `examples/config_2D/chemical_ordering`

## Dask Worker Memory Warnings

Typical warning:

```text
distributed.worker.memory - WARNING - Worker is at 80% memory usage. Pausing worker.
```

What it usually means:

- Dask is warning about host RAM, not GPU VRAM.
- `local` runs with multiple processes can duplicate large arrays across workers.

First checks:

- reduce `runtime.dask.max_workers`
- retry with `synchronous` for a bounded run
- use `MOSAIC_NUFFT_CPU_ONLY=1` to separate GPU issues from general workflow issues

## `cuda-local` Fails To Start

Typical symptoms:

- `ImportError` for `dask_cuda`
- CUDA initialization failures
- no visible GPU devices

Checks:

- confirm the active environment includes `dask-cuda`
- verify `CUDA_VISIBLE_DEVICES`
- retry the same case on CPU if you only need to validate the workflow surface

## File Not Found Or Wrong Output Location

Checks:

- confirm `paths.config_root`
- confirm `paths.structure_file`
- confirm `paths.output_directory`
- run from the repository root when using the example commands as written

Each shipped example `run_parameters.json` points to `./input_parameters.json`,
so relative paths are resolved from that example directory after loading.

## Runs Are Slow Or Unstable

The main drivers are:

- reciprocal-space interval size
- worker count
- point window size
- whether the run uses CPU-only, `local`, or `cuda-local`

Simplify one variable at a time:

- reduce worker count
- switch to `synchronous`
- force CPU-only
- start from a 1D example instead of a 2D example

## Useful References

- [runtime_guide.md](runtime_guide.md)
- [input_schema.md](input_schema.md)
- [../examples/README.md](../examples/README.md)
