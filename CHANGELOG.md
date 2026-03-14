# Changelog

All notable changes to MOSAIC will be documented in this file.

The format is based on Keep a Changelog, with a lightweight project-specific
baseline starting from the current scientific-stage release layout.

## [Unreleased]

### Changed

- Aligned the release-facing README, example docs, and schema docs to the
  manuscript method vocabulary: diffuse scattering amplitude, phase-preserving
  masks, inverse Fourier transform, and residual real-space fields.
- Clarified the supported release-candidate surface versus research-only
  example material.
- Replaced placeholder docs in `docs/`, `adapters/`, and `plugins/` with
  release-facing guidance.

## [0.1.0] - 2026-03-14

### Added

- Installable packaging metadata and a stable `mosaic` CLI entrypoint.
- A canonical bounded 1D smoke example under `examples/`.
- Baseline CI for import/version smoke, CLI help/version, canonical smoke, and
  `pytest -q tests/unit`.
- Release metadata baselines for changelog, citation, and contributing.

### Changed

- Reorganized the codebase around scientific stages:
  `config`, `structure`, `patch_centers`, `qspace`, `scattering`,
  `residual_field`, `decoding`, `storage`, `runtime`, `adapters`, and
  `workflow`.
- Stabilized stage boundaries, stage-local contracts, and the release-facing
  run path around `python -m core.main ...` and `mosaic ...`.
- Hardened the canonical smoke path for CPU-only validation environments.

### Known Limitations

- No repository `LICENSE` has been selected yet.
- Broader 2D/3D example curation remains deferred.
- Governance and release automation remain intentionally minimal at this stage.
