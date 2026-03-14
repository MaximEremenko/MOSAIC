# Examples

Release Wave 1 supports one canonical example:

- [run_parameters.json](run_parameters.json)
- [input_parameters.json](input_parameters.json)
- [sample_1d.f1d](sample_1d.f1d)
- [central_points.txt](central_points.txt)

Run it from the repository root:

```bash
MOSAIC_NUFFT_CPU_ONLY=1 python -m core.main examples/run_parameters.json
```

Expected output location:

- `examples/sample_1d_release/processed_point_data/`

The rest of `examples/` currently contains larger legacy, research, and
historical assets. They are not yet curated as release-grade examples for this
wave.
