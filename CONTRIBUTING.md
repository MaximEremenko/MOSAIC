# Contributing

MOSAIC is currently maintained as a scientific-stage codebase with a small
release surface. Keep changes narrow, behavior-preserving where possible, and
aligned with the current stage boundaries:

- `core.scattering`
- `core.residual_field`
- `core.decoding`

## Development Setup

Conda-first baseline:

```bash
conda env create -f core/environment.yml
conda activate mosaic
python -m pip install -e .
```

## Before Opening a Change

Run the canonical smoke path:

```bash
python scripts/smoke_example.py
```

Run the unit suite:

```bash
conda run -n mosaic python -m pytest -q tests/unit
```

## Change Scope

- Do not reopen package layout or scientific-stage naming without an explicit
  refactor plan update.
- Keep storage/runtime changes local unless they are required for correctness.
- Prefer stage-local contracts and models over new global abstraction files.
- Keep the documented run path stable:
  - `python -m core.main <run_parameters.json>`
  - `mosaic <run_parameters.json>`

## Pull Requests

- Explain the user-visible or release-visible effect of the change.
- Note any validation you ran.
- Call out deferred follow-up work instead of folding unrelated cleanup into the
  same change.
