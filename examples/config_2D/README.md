# 2D Examples

This directory contains the two current 2D example cases:

- [displacement/](displacement)
- [chemical_ordering/](chemical_ordering)

## `displacement/`

- structure file: `displacement/sample_2d.f2d`
- mode: `displacement`
- backend: `local`
- checked-in runtime: `max_workers: 6`, `threads_per_worker: 1`, `processes: true`
- output roots:
  - `displacement/output_displacement`
  - `displacement/output_displacement_decoder_full`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_2D/displacement/run_parameters.json
```

## `chemical_ordering/`

- structure file: `chemical_ordering/sample_2d_chem_PMN_2Dcol10.f2d`
- mode: `chemical`
- backend: `cuda-local`
- checked-in runtime: `max_workers: 1`, `threads_per_worker: 1`, `processes: false`
- output root: `chemical_ordering/output_chemical_ordering`

Run it with:

```bash
conda run -n mosaic python -m core.main examples/config_2D/chemical_ordering/run_parameters.json
```

If you want to pin a specific GPU for the chemical-ordering example:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mosaic python -m core.main examples/config_2D/chemical_ordering/run_parameters.json
```
