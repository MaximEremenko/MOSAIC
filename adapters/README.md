# Adapters

This directory contains optional backend-specific helpers used by the current
scientific-stage workflow.

Release-candidate expectations:

- adapters are support code, not the primary user-facing API
- the canonical smoke path does not require direct adapter use
- GPU acceleration is optional and adapter-backed
- CPU-only execution remains the supported release baseline

Current examples include:

- `core/adapters/cunufft_wrapper.py`
  - optional NUFFT backend selection and CPU fallback behavior
- `core/adapters/mask_plot_helper.py`
  - plotting/visualization support, not part of the canonical smoke path
